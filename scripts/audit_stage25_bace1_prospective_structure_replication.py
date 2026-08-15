"""Independently audit the Stage25 BACE1 prospective replication."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage22_beam_baseline import beam_search
from scripts.run_stage21_structure_aware_qubo import (
    distance_matrix,
    file_sha256,
    load_target,
    read_csv,
    read_json,
    rooted,
)
from scripts.run_stage22_structural_state_coverage_qubo import (
    assignment_for_subset,
    build_auxiliary_qubo,
    build_coverage_terms,
    direct_greedy,
    improve_by_swaps,
    objective_components,
    qubo_energy,
    qubo_hash,
)


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def subset(value: str) -> tuple[str, ...]:
    values = tuple(sorted(part for part in value.split("+") if part))
    if len(values) != len(set(values)):
        raise ValueError("duplicate subset member")
    return values


def audit(config_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    result_path = rooted(root, config["outputs"]["result_json"])
    result = read_json(result_path)
    if result.get("status") != "stage25_bace1_prospective_structure_replication_complete":
        raise ValueError("unexpected Stage25 status")
    if result["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("config hash differs")
    implementation = rooted(root, result["implementation"]["path"])
    if result["implementation"]["sha256"] != file_sha256(implementation):
        raise ValueError("implementation hash differs")
    paths: dict[str, Path] = {}
    for key, descriptor in result["outputs"].items():
        path = rooted(root, descriptor["path"])
        if descriptor["sha256"] != file_sha256(path):
            raise ValueError(f"output hash differs: {key}")
        paths[key] = path
    reads = read_csv(paths["read_csv"])
    batches = read_csv(paths["batch_csv"])
    baselines = read_csv(paths["baseline_csv"])
    model = read_json(paths["model_record_json"])
    target_spec = {
        "reference_id": config["target"]["reference_id"],
        "inputs": config["target"]["inputs"],
    }
    target = load_target(root, "BACE1", target_spec)
    ids = target["ids"]
    matrix = distance_matrix(ids, target["distances"])
    objective = config["objective"]
    sampler = config["sampler"]
    gate = config["gate"]
    target_k = int(objective["k"])
    diversity_weight = float(objective["diversity_weight"])
    terms = build_coverage_terms(
        ids, matrix, float(objective["neighborhood_fraction"])
    )
    expected_counts = (
        int(sampler["batch_count"]) * int(sampler["reads_per_batch"]),
        int(sampler["batch_count"]),
        1 + len(config["classical_baselines"]["beam_widths"]),
    )
    if (len(reads), len(batches), len(baselines)) != expected_counts:
        raise ValueError("Stage25 row counts differ")
    baseline_index = {
        (row["method"], int(row["beam_width"])): row for row in baselines
    }
    greedy = direct_greedy(ids, matrix, terms, target_k, diversity_weight)
    greedy_subset, greedy_metrics, _ = improve_by_swaps(
        greedy, ids, matrix, terms, target_k, diversity_weight
    )
    recorded = baseline_index[("direct_greedy_plus_swap", 0)]
    if subset(recorded["selected_subset"]) != greedy_subset:
        raise ValueError("direct baseline subset differs")
    close(
        float(recorded["composite_objective"]),
        float(greedy_metrics["composite_objective"]),
        "direct baseline",
    )
    baseline_values = [float(greedy_metrics["composite_objective"])]
    for width_value in config["classical_baselines"]["beam_widths"]:
        width = int(width_value)
        beam, _ = beam_search(
            ids, matrix, terms, target_k, diversity_weight, width
        )
        refined, metrics, _ = improve_by_swaps(
            beam, ids, matrix, terms, target_k, diversity_weight
        )
        recorded = baseline_index[("beam_plus_swap", width)]
        if subset(recorded["selected_subset"]) != refined:
            raise ValueError(f"beam {width} subset differs")
        close(
            float(recorded["composite_objective"]),
            float(metrics["composite_objective"]),
            f"beam {width}",
        )
        baseline_values.append(float(metrics["composite_objective"]))
    strong = max(baseline_values)
    read_groups: dict[int, list[dict[str, str]]] = {}
    for row in reads:
        read_groups.setdefault(int(row["batch"]), []).append(row)
    batch_index = {int(row["batch"]): row for row in batches}
    checked_reads = 0
    batch_values: list[float] = []
    batch_deltas: list[float] = []
    for batch in range(int(sampler["batch_count"])):
        group = read_groups[batch]
        if len(group) != int(sampler["reads_per_batch"]):
            raise ValueError(f"batch {batch} read count differs")
        values = []
        for row in group:
            selected = subset(row["selected_subset"])
            if len(selected) != target_k or not set(selected).issubset(ids):
                raise ValueError("invalid sampled subset")
            metrics = objective_components(
                selected, ids, matrix, terms, diversity_weight
            )
            close(
                float(row["composite_objective"]),
                float(metrics["composite_objective"]),
                f"batch {batch} objective",
            )
            close(
                float(row["delta_vs_strong_classical"]),
                float(metrics["composite_objective"]) - strong,
                f"batch {batch} delta",
            )
            values.append(float(metrics["composite_objective"]))
            checked_reads += 1
        best = max(values)
        close(float(batch_index[batch]["best_objective"]), best, f"batch {batch} best")
        close(
            float(batch_index[batch]["delta_vs_strong_classical"]),
            best - strong,
            f"batch {batch} best delta",
        )
        batch_values.append(best)
        batch_deltas.append(best - strong)
    best = max(batch_values)
    within = sum(
        value >= best - float(gate["objective_tolerance"])
        for value in batch_values
    ) / len(batch_values)
    above = sum(
        value > float(gate["minimum_gain_vs_strong_classical"])
        for value in batch_deltas
    ) / len(batch_deltas)
    target_record = result["target_record"]
    close(float(target_record["strong_classical_objective"]), strong, "strong")
    close(float(target_record["best_sampler_objective"]), best, "sampler")
    close(float(target_record["within_tolerance_batch_fraction"]), within, "stable")
    close(float(target_record["above_strong_classical_batch_fraction"]), above, "winning")
    selected = tuple(target_record["best_sampler_subset"])
    metrics = objective_components(selected, ids, matrix, terms, diversity_weight)
    qubo = build_auxiliary_qubo(
        ids,
        matrix,
        terms,
        target_k,
        diversity_weight,
        float(objective["cardinality_penalty"]),
        float(objective["coverage_constraint_penalty"]),
    )
    close(
        qubo_energy(qubo, assignment_for_subset(selected, terms, qubo)),
        -float(metrics["composite_objective"]),
        "QUBO energy",
    )
    if model["qubo"]["sha256"] != qubo_hash(qubo):
        raise ValueError("QUBO hash differs")
    gate_passed = (
        within >= float(gate["minimum_batch_fraction_within_tolerance"])
        and above >= float(gate["minimum_batch_fraction_above_strong_classical"])
    )
    if result["decision"]["prospective_structure_replication_gate_passed"] != gate_passed:
        raise ValueError("gate decision differs")
    if any(int(value) != 0 for value in result["data_boundary"].values()):
        raise ValueError("nonzero data boundary")
    audit_result = {
        "schema_version": "1.0",
        "status": "stage25_bace1_prospective_structure_replication_audit_ok",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "result": {"path": result_path.relative_to(root).as_posix(), "sha256": file_sha256(result_path)},
        "coverage": {
            "read_rows_recomputed": checked_reads,
            "batch_rows_recomputed": len(batches),
            "classical_baselines_recomputed": len(baselines),
            "candidate_count": len(ids),
        },
        "decision_recomputed": {
            "prospective_structure_replication_gate_passed": gate_passed,
            "within_tolerance_batch_fraction": within,
            "above_strong_classical_batch_fraction": above,
        },
        "checks": {
            "all_output_hashes_verified": True,
            "all_read_objectives_recomputed": True,
            "all_classical_baselines_recomputed": True,
            "qubo_hash_and_energy_recomputed": True,
            "data_boundary_zero": True,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
    }
    output = rooted(root, output_path.as_posix())
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
        default=Path("configs/stage25_bace1_prospective_structure_replication.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage25_bace1_prospective_structure_replication_audit.json"),
    )
    args = parser.parse_args()
    audit(args.config, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
