"""Independently audit the Stage 22 structural-state coverage QUBO screen."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    distance_matrix,
    file_sha256,
    load_target,
    maxmin_seeded,
    maxsum_greedy,
    read_csv,
    read_json,
    rooted,
)
from scripts.run_stage22_structural_state_coverage_qubo import (
    BASELINE_METHODS,
    CANDIDATE_METHOD,
    assignment_for_subset,
    build_auxiliary_qubo,
    build_coverage_terms,
    direct_greedy,
    exact_oracle,
    objective_components,
    parse_subset,
    qubo_energy,
    qubo_hash,
)


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def checked_subset(value: str, ids: list[str], k: int, label: str) -> tuple[str, ...]:
    subset = parse_subset(value)
    if len(subset) != k or len(set(subset)) != k or not set(subset).issubset(ids):
        raise ValueError(f"{label}: invalid subset")
    return subset


def check_metrics(
    row: dict[str, str],
    subset: tuple[str, ...],
    ids: list[str],
    matrix: Any,
    terms: dict[str, Any],
    diversity_weight: float,
    label: str,
) -> dict[str, float]:
    metrics = objective_components(subset, ids, matrix, terms, diversity_weight)
    for key in (
        "coverage_fraction",
        "mean_pair_distance_normalized",
        "minimum_pair_distance_normalized",
        "composite_objective",
    ):
        close(float(row[key]), float(metrics[key]), f"{label}/{key}")
    return metrics


def jaccard(first: tuple[str, ...], second: tuple[str, ...]) -> float:
    left, right = set(first), set(second)
    return len(left & right) / len(left | right)


def audit(config_path: Path, root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    result_path = rooted(root, config["outputs"]["result_json"])
    result = read_json(result_path)
    if result.get("status") != "stage22_structural_state_coverage_qubo_complete":
        raise ValueError("unexpected Stage 22 result status")
    if result["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("result identifies another config")
    implementation_path = rooted(root, result["implementation"]["path"])
    if result["implementation"]["sha256"] != file_sha256(implementation_path):
        raise ValueError("implementation hash differs")

    output_paths: dict[str, Path] = {}
    for key, descriptor in result["outputs"].items():
        path = rooted(root, descriptor["path"])
        if file_sha256(path) != descriptor["sha256"]:
            raise ValueError(f"output hash differs: {key}")
        output_paths[key] = path
    for target_inputs in result["inputs"].values():
        for descriptor in target_inputs.values():
            if file_sha256(rooted(root, descriptor["path"])) != descriptor["sha256"]:
                raise ValueError(f"input hash differs: {descriptor['path']}")

    selection_rows = read_csv(output_paths["selection_csv"])
    restart_rows = read_csv(output_paths["restart_csv"])
    model = read_json(output_paths["model_record_json"])
    diagnostic = config["diagnostic"]
    fractions = [float(value) for value in diagnostic["neighborhood_fractions"]]
    k_values = [int(value) for value in diagnostic["k_values"]]
    restart_count = int(diagnostic["restart_count"])
    diversity_weight = float(diagnostic["diversity_weight"])
    exact_state_limit = int(diagnostic["exact_state_limit"])
    cardinality_penalty = float(diagnostic["cardinality_penalty"])
    constraint_penalty = float(diagnostic["constraint_penalty"])
    expected_selection = (
        len(config["targets"])
        * len(fractions)
        * len(k_values)
        * (1 + len(BASELINE_METHODS))
    )
    expected_restarts = (
        len(config["targets"]) * len(fractions) * len(k_values) * restart_count
    )
    if len(selection_rows) != expected_selection:
        raise ValueError(f"selection row count differs: {len(selection_rows)}")
    if len(restart_rows) != expected_restarts:
        raise ValueError(f"restart row count differs: {len(restart_rows)}")

    selection_index = {
        (
            row["target_id"],
            float(row["neighborhood_fraction"]),
            int(row["k"]),
            row["method"],
        ): row
        for row in selection_rows
    }
    if len(selection_index) != len(selection_rows):
        raise ValueError("duplicate selection row")
    restart_index: dict[tuple[str, float, int], list[dict[str, str]]] = {}
    for row in restart_rows:
        key = (
            row["target_id"],
            float(row["neighborhood_fraction"]),
            int(row["k"]),
        )
        restart_index.setdefault(key, []).append(row)

    checked_selection = 0
    checked_restarts = 0
    checked_swaps = 0
    exact_states_checked = 0
    target_checks: dict[str, Any] = {}
    posthoc_stability: dict[str, Any] = {}
    for target_id, spec in config["targets"].items():
        target = load_target(root, target_id, spec)
        ids = target["ids"]
        matrix = distance_matrix(ids, target["distances"])
        reference_id = str(spec["reference_id"])
        target_checks[target_id] = {
            "candidate_count": len(ids),
            "hard_gate_excluded_count": len(target["excluded_hard_gate"]),
            "fraction_checks": {},
        }
        posthoc_stability[target_id] = {}
        for fraction in fractions:
            terms = build_coverage_terms(ids, matrix, fraction)
            fraction_key = f"{fraction:.6f}"
            target_checks[target_id]["fraction_checks"][fraction_key] = {
                "neighbor_count": terms["neighbor_count"],
                "k_checks": {},
            }
            for k in k_values:
                label = f"{target_id}/{fraction:.6f}/{k}"
                key = (target_id, fraction, k)
                rows = {
                    method: selection_index[(*key, method)]
                    for method in (CANDIDATE_METHOD, *BASELINE_METHODS)
                }
                subsets = {
                    method: checked_subset(row["selected_subset"], ids, k, f"{label}/{method}")
                    for method, row in rows.items()
                }
                metrics = {
                    method: check_metrics(
                        rows[method], subset, ids, matrix, terms,
                        diversity_weight, f"{label}/{method}"
                    )
                    for method, subset in subsets.items()
                }
                expected_baselines = {
                    "direct_greedy": direct_greedy(
                        ids, matrix, terms, k, diversity_weight
                    ),
                    "maxmin_seeded": maxmin_seeded(ids, matrix, k, reference_id),
                    "maxsum_greedy": maxsum_greedy(ids, matrix, k),
                }
                for method, expected in expected_baselines.items():
                    if subsets[method] != expected:
                        raise ValueError(f"{label}: {method} differs")
                greedy_value = metrics["direct_greedy"]["composite_objective"]
                for method, row in rows.items():
                    close(
                        float(row["delta_composite_vs_direct_greedy"]),
                        metrics[method]["composite_objective"] - greedy_value,
                        f"{label}/{method}/greedy-delta",
                    )

                restarts = restart_index[key]
                if len(restarts) != restart_count:
                    raise ValueError(f"{label}: restart count differs")
                restart_values: list[float] = []
                restart_subsets: list[tuple[str, ...]] = []
                for restart in restarts:
                    subset = checked_subset(
                        restart["selected_subset"], ids, k, f"{label}/restart"
                    )
                    current = check_metrics(
                        restart, subset, ids, matrix, terms,
                        diversity_weight, f"{label}/restart"
                    )
                    current_value = float(current["composite_objective"])
                    selected = set(subset)
                    for outgoing in subset:
                        for incoming in ids:
                            if incoming in selected:
                                continue
                            candidate = tuple(sorted((selected - {outgoing}) | {incoming}))
                            candidate_value = objective_components(
                                candidate, ids, matrix, terms, diversity_weight
                            )["composite_objective"]
                            if candidate_value > current_value + 1e-12:
                                raise ValueError(
                                    f"{label}: restart is not one-swap local optimum"
                                )
                            checked_swaps += 1
                    restart_values.append(current_value)
                    restart_subsets.append(subset)
                    checked_restarts += 1

                exact = exact_oracle(
                    ids, matrix, terms, k, diversity_weight, exact_state_limit
                )
                candidate_subset = subsets[CANDIDATE_METHOD]
                candidate_value = metrics[CANDIDATE_METHOD]["composite_objective"]
                if exact is not None:
                    exact_states_checked += int(exact["state_count"])
                    if candidate_subset != tuple(exact["selected_subset"]):
                        raise ValueError(f"{label}: candidate differs from exact oracle")
                    close(
                        candidate_value,
                        float(exact["metrics"]["composite_objective"]),
                        f"{label}/exact-objective",
                    )
                else:
                    close(candidate_value, max(restart_values), f"{label}/best-restart")

                qubo = build_auxiliary_qubo(
                    ids, matrix, terms, k, diversity_weight,
                    cardinality_penalty, constraint_penalty
                )
                assignment = assignment_for_subset(candidate_subset, terms, qubo)
                energy = qubo_energy(qubo, assignment)
                close(energy, -candidate_value, f"{label}/qubo-equivalence")
                recorded = model["target_models"][target_id]["fraction_models"][
                    fraction_key
                ]["k_models"][str(k)]
                if recorded["qubo"]["sha256"] != qubo_hash(qubo):
                    raise ValueError(f"{label}: QUBO hash differs")
                if recorded["selected_subset"] != list(candidate_subset):
                    raise ValueError(f"{label}: model subset differs")

                best_frequency = sum(subset == candidate_subset for subset in restart_subsets)
                if int(recorded["best_restart_frequency"]) != best_frequency:
                    raise ValueError(f"{label}: best frequency differs")
                target_checks[target_id]["fraction_checks"][fraction_key][
                    "k_checks"
                ][str(k)] = {
                    "state_count": math.comb(len(ids), k),
                    "exact_oracle_checked": exact is not None,
                    "restart_count": len(restarts),
                    "unique_restart_solution_count": len(set(restart_subsets)),
                    "best_restart_frequency": best_frequency,
                    "qubo_sha256": qubo_hash(qubo),
                }
                checked_selection += 1 + len(BASELINE_METHODS)

                if k == int(config["go_no_go"]["required_common_k"]):
                    regrets = [candidate_value - value for value in restart_values]
                    posthoc_stability[target_id][fraction_key] = {
                        "label": "posthoc_descriptive_only_not_a_gate_revision",
                        "best_objective": candidate_value,
                        "exact_subset_frequency": best_frequency / restart_count,
                        "unique_subset_count": len(set(restart_subsets)),
                        "median_objective_regret": statistics.median(regrets),
                        "maximum_objective_regret": max(regrets),
                        "within_1e-4_objective_fraction": sum(
                            regret <= 1e-4 + 1e-12 for regret in regrets
                        ) / restart_count,
                        "within_1e-3_objective_fraction": sum(
                            regret <= 1e-3 + 1e-12 for regret in regrets
                        ) / restart_count,
                        "median_jaccard_to_selected": statistics.median(
                            jaccard(candidate_subset, subset)
                            for subset in restart_subsets
                        ),
                    }

    primary_fraction = float(diagnostic["primary_neighborhood_fraction"])
    required_k = int(config["go_no_go"]["required_common_k"])
    minimum_gain = float(config["go_no_go"]["minimum_objective_gain"])
    minimum_frequency = float(
        config["go_no_go"]["minimum_best_restart_fraction"]
    )
    strict_targets: list[str] = []
    sensitivity: dict[str, Any] = {}
    for target_id in sorted(config["targets"]):
        primary = selection_index[
            (target_id, primary_fraction, required_k, CANDIDATE_METHOD)
        ]
        if (
            float(primary["delta_composite_vs_direct_greedy"]) > minimum_gain
            and float(primary["best_restart_fraction"]) >= minimum_frequency
        ):
            strict_targets.append(target_id)
        positive = sum(
            float(
                selection_index[(target_id, fraction, required_k, CANDIDATE_METHOD)][
                    "delta_composite_vs_direct_greedy"
                ]
            )
            > minimum_gain
            for fraction in fractions
        )
        sensitivity[target_id] = {
            "positive_fraction_count": positive,
            "fraction_count": len(fractions),
            "passed": positive
            >= int(config["go_no_go"]["minimum_positive_sensitivity_fractions"]),
        }
    gate = strict_targets == sorted(config["targets"]) and all(
        value["passed"] for value in sensitivity.values()
    )
    expected_decision = result["decision"]
    if expected_decision["strict_primary_targets"] != strict_targets:
        raise ValueError("strict primary target decision differs")
    if expected_decision["sensitivity"] != sensitivity:
        raise ValueError("sensitivity decision differs")
    if bool(expected_decision["structural_coverage_gate_passed"]) != gate:
        raise ValueError("structural coverage gate decision differs")

    if any(int(value) != 0 for value in result["data_boundary"].values()):
        raise ValueError("Stage 22 data boundary is nonzero")
    protected = ("fresh_validation", "locked_test", "bace1", "docking_score")
    input_paths = [
        descriptor["path"]
        for target in result["inputs"].values()
        for descriptor in target.values()
    ]
    if any(marker in path.lower() for path in input_paths for marker in protected):
        raise ValueError("protected data path entered Stage 22")

    audit_result = {
        "schema_version": "1.0",
        "status": "stage22_structural_state_coverage_qubo_audit_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "result": {
            "path": result_path.relative_to(root).as_posix(),
            "sha256": file_sha256(result_path),
        },
        "coverage": {
            "selection_rows_recomputed": checked_selection,
            "restart_rows_recomputed": checked_restarts,
            "one_swap_transitions_checked": checked_swaps,
            "exact_subset_states_enumerated": exact_states_checked,
            "target_count": len(config["targets"]),
            "fraction_count": len(fractions),
            "k_count": len(k_values),
        },
        "target_checks": target_checks,
        "decision_recomputed": {
            "strict_primary_targets": strict_targets,
            "sensitivity": sensitivity,
            "structural_coverage_gate_passed": gate,
        },
        "posthoc_stability_diagnostic": posthoc_stability,
        "checks": {
            "input_hashes_verified": True,
            "output_hashes_verified": True,
            "all_metrics_recomputed": True,
            "classical_baselines_recomputed": True,
            "exact_oracles_recomputed_where_feasible": True,
            "all_restarts_are_one_swap_local_optima": True,
            "auxiliary_qubo_hashes_and_energies_recomputed": True,
            "go_no_go_decision_recomputed": True,
            "data_boundary_zero": True,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
    }
    output = rooted(root, output.as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit_result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(audit_result, indent=2, sort_keys=True))
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage22_structural_state_coverage_qubo.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage22_structural_state_coverage_qubo_audit.json"),
    )
    args = parser.parse_args()
    audit(args.config, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
