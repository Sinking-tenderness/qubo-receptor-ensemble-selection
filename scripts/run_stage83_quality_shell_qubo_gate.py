"""Screen a quadratic quality-shell QUBO for non-convex frontier recovery."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import dimod
import numpy as np
from dwave.samplers import TabuSampler

try:
    import scripts.run_stage75_explicit_variable_k_cqm as s75
    import scripts.run_stage81_dirac_global_qubo_formulation_gate as s81
    import scripts.run_stage82_lagrangian_fixed_k_qubo_gate as s82
except ImportError:
    import run_stage75_explicit_variable_k_cqm as s75
    import run_stage81_dirac_global_qubo_formulation_gate as s81
    import run_stage82_lagrangian_fixed_k_qubo_gate as s82


def build_shell_bqm(
    cell: dict[str, Any],
    k: int,
    shell_center: float,
    shell_strength: float,
    multiplier: float,
) -> tuple[dimod.BinaryQuadraticModel, dict[str, float]]:
    model = cell["model"]
    reward = float(cell["reward"])
    shifted = np.asarray(model["raw_coefficients"], dtype=float) - reward
    pair_scale = max(float(np.max(np.abs(shifted))), 1e-12)
    threshold = int(cell["frontiers"][k]["quality_threshold"])
    quality = np.asarray(model["deficits"], dtype=float) / max(threshold, 1)
    linear = {
        f"x{index:03d}": float(shell_strength)
        * (float(value) ** 2 - 2 * float(shell_center) * float(value))
        for index, value in enumerate(quality)
    }
    quadratic = {
        (f"x{left:03d}", f"x{right:03d}"): float(value) / pair_scale
        + 2
        * float(shell_strength)
        * float(quality[left])
        * float(quality[right])
        for (left, right), value in zip(model["pairs"], shifted)
    }
    base = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
    penalized, penalty, flip_bound = s82.add_cardinality_penalty(
        base, k, multiplier
    )
    return penalized, {
        "pair_scale": pair_scale,
        "quality_threshold": float(threshold),
        "cardinality_penalty": penalty,
        "base_maximum_flip_bound": flip_bound,
    }


def reference_supported(
    cell: dict[str, Any], k: int, center: float, strength: float
) -> bool:
    model = cell["model"]
    reward = float(cell["reward"])
    shifted = np.asarray(model["raw_coefficients"], dtype=float) - reward
    pair_scale = max(float(np.max(np.abs(shifted))), 1e-12)
    frontier = cell["frontiers"][k]
    threshold = int(frontier["quality_threshold"])
    reference = tuple(frontier["reference_subset"])
    chosen = set(reference)
    reference_objective = s75.variable_energy(model, reference, reward) / pair_scale
    reference_quality = s75.subset_deficit(model, reference) / max(threshold, 1)
    reference_energy = reference_objective + float(strength) * (
        reference_quality - float(center)
    ) ** 2
    for outgoing in reference:
        for incoming in range(model["count"]):
            if incoming in chosen:
                continue
            neighbor = tuple(sorted((chosen - {outgoing}) | {incoming}))
            objective = s75.variable_energy(model, neighbor, reward) / pair_scale
            quality = s75.subset_deficit(model, neighbor) / max(threshold, 1)
            energy = objective + float(strength) * (quality - float(center)) ** 2
            if energy < reference_energy - 1e-12:
                return False
    return True


def sample_condition(
    config: dict[str, Any],
    cell: dict[str, Any],
    k: int,
    condition: dict[str, float],
    seed: int,
) -> tuple[dict[str, Any], tuple[int, ...] | None]:
    protocol = config["quality_shell_screen"]
    center = float(condition["center"])
    strength = float(condition["strength"])
    bqm, encoding = build_shell_bqm(
        cell,
        k,
        center,
        strength,
        float(protocol["cardinality_penalty_multiplier"]),
    )
    quantized, precision = s81.fixed_precision_bqm(
        bqm, int(protocol["signed_precision_bits"])
    )
    samples = TabuSampler().sample(
        quantized,
        num_reads=int(protocol["tabu_reads"]),
        timeout=int(protocol["tabu_timeout_milliseconds"]),
        seed=seed,
    )
    model = cell["model"]
    threshold = int(cell["frontiers"][k]["quality_threshold"])
    total = 0
    cardinality_valid = 0
    quality_valid = 0
    best_subset: tuple[int, ...] | None = None
    best_objective = math.inf
    for datum in samples.data(fields=["sample", "num_occurrences"]):
        occurrences = int(datum.num_occurrences)
        subset = tuple(
            index
            for index in range(model["count"])
            if int(datum.sample[f"x{index:03d}"]) == 1
        )
        total += occurrences
        if len(subset) != int(k):
            continue
        cardinality_valid += occurrences
        if s75.subset_deficit(model, subset) > threshold:
            continue
        quality_valid += occurrences
        objective = s75.variable_energy(model, subset, float(cell["reward"]))
        if objective < best_objective - 1e-12 or (
            math.isclose(objective, best_objective, abs_tol=1e-12)
            and (best_subset is None or subset < best_subset)
        ):
            best_objective = objective
            best_subset = subset
    baseline_subset = tuple(cell["frontiers"][k]["reference_subset"])
    baseline = s75.variable_energy(model, baseline_subset, float(cell["reward"]))
    record = model["record"]
    row = {
        "target_id": str(record["target_id"]),
        "outer_fold": int(record["outer_fold"]),
        "k": int(k),
        "shell_center": center,
        "shell_strength": strength,
        "reference_one_swap_supported": reference_supported(
            cell, k, center, strength
        ),
        "candidate_receptor_count": int(model["count"]),
        "bqm_variable_count": int(bqm.num_variables),
        "bqm_interaction_count": int(bqm.num_interactions),
        "qci_total_binary_levels": 2 * int(bqm.num_variables),
        "qci_level_limit_ok": 2 * int(bqm.num_variables)
        <= int(config["experiment"]["qci_total_level_limit"]),
        "cardinality_penalty": encoding["cardinality_penalty"],
        "base_maximum_flip_bound": encoding["base_maximum_flip_bound"],
        "penalty_dominates_flip_bound": encoding["cardinality_penalty"]
        > encoding["base_maximum_flip_bound"],
        "quantized_dynamic_range": precision["quantized_dynamic_range"],
        "coefficient_retention_fraction": precision["retained_nonzero_bias_count"]
        / precision["source_nonzero_bias_count"],
        "read_count": total,
        "cardinality_valid_read_fraction": cardinality_valid / total,
        "quality_valid_read_fraction": quality_valid / total,
        "baseline_objective": baseline,
        "best_feasible_objective": ""
        if best_subset is None
        else best_objective,
        "frontier_competitive": best_subset is not None
        and best_objective <= baseline + 1e-8,
        "best_subset": ""
        if best_subset is None
        else s75.subset_name(model, best_subset),
    }
    return row, best_subset


def run_screen(
    config: dict[str, Any], cells: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = config["quality_shell_screen"]
    trials: list[dict[str, Any]] = []
    candidates: dict[
        tuple[str, int, int], list[tuple[float, tuple[int, ...], float, float]]
    ] = defaultdict(list)
    condition_index = 0
    for cell in cells:
        model = cell["model"]
        target = str(model["record"]["target_id"])
        fold = int(model["record"]["outer_fold"])
        for k in cell["frontiers"]:
            for condition in protocol["conditions"]:
                row, subset = sample_condition(
                    config,
                    cell,
                    int(k),
                    condition,
                    int(protocol["seed_base"]) + condition_index,
                )
                trials.append(row)
                if subset is not None:
                    candidates[(target, fold, int(k))].append(
                        (
                            float(row["best_feasible_objective"]),
                            subset,
                            float(condition["center"]),
                            float(condition["strength"]),
                        )
                    )
                condition_index += 1
    cell_rows: list[dict[str, Any]] = []
    cell_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in cells:
        model = cell["model"]
        target = str(model["record"]["target_id"])
        fold = int(model["record"]["outer_fold"])
        cell_lookup[(target, fold)] = cell
        for k, frontier in cell["frontiers"].items():
            local = candidates[(target, fold, int(k))]
            reference = tuple(frontier["reference_subset"])
            baseline = s75.variable_energy(model, reference, float(cell["reward"]))
            if local:
                objective, subset, center, strength = min(
                    local, key=lambda item: (item[0], item[1], item[2], item[3])
                )
            else:
                objective, subset, center, strength = (
                    math.inf,
                    tuple(),
                    math.nan,
                    math.nan,
                )
            support = any(
                bool(row["reference_one_swap_supported"])
                for row in trials
                if row["target_id"] == target
                and int(row["outer_fold"]) == fold
                and int(row["k"]) == int(k)
            )
            cell_rows.append(
                {
                    "target_id": target,
                    "outer_fold": fold,
                    "k": int(k),
                    "candidate_receptor_count": int(model["count"]),
                    "reference_supported_by_grid": support,
                    "baseline_objective": baseline,
                    "best_feasible_objective": ""
                    if math.isinf(objective)
                    else objective,
                    "frontier_competitive": objective <= baseline + 1e-8,
                    "best_quality_weight": ""
                    if math.isnan(center)
                    else f"{center:g}|{strength:g}",
                    "best_shell_center": "" if math.isnan(center) else center,
                    "best_shell_strength": "" if math.isnan(strength) else strength,
                    "best_subset": ""
                    if not subset
                    else s75.subset_name(model, subset),
                }
            )
    variable_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        grouped[(str(row["target_id"]), int(row["outer_fold"]))].append(row)
    for key, rows in sorted(grouped.items()):
        cell = cell_lookup[key]
        references = {
            int(k): s75.variable_energy(
                cell["model"],
                tuple(frontier["reference_subset"]),
                float(cell["reward"]),
            )
            for k, frontier in cell["frontiers"].items()
        }
        reference_k, reference_objective = min(references.items(), key=lambda item: item[1])
        feasible = [row for row in rows if row["best_feasible_objective"] != ""]
        best = (
            min(
                feasible,
                key=lambda row: (
                    float(row["best_feasible_objective"]),
                    int(row["k"]),
                    str(row["best_subset"]),
                ),
            )
            if feasible
            else None
        )
        variable_rows.append(
            {
                "target_id": key[0],
                "outer_fold": key[1],
                "reference_k": int(reference_k),
                "reference_objective": float(reference_objective),
                "candidate_k": "" if best is None else int(best["k"]),
                "candidate_objective": ""
                if best is None
                else float(best["best_feasible_objective"]),
                "frontier_competitive": best is not None
                and float(best["best_feasible_objective"])
                <= float(reference_objective) + 1e-8,
                "candidate_subset": "" if best is None else str(best["best_subset"]),
            }
        )
    return trials, cell_rows, variable_rows


def summarize(
    config: dict[str, Any],
    trials: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    variable: list[dict[str, Any]],
) -> dict[str, Any]:
    target_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        target_groups[str(row["target_id"])].append(row)
    fixed_fraction = sum(bool(row["frontier_competitive"]) for row in cells) / len(cells)
    support_fraction = sum(bool(row["reference_supported_by_grid"]) for row in cells) / len(cells)
    variable_fraction = sum(bool(row["frontier_competitive"]) for row in variable) / len(variable)
    target_fractions = {
        target: sum(bool(row["frontier_competitive"]) for row in rows) / len(rows)
        for target, rows in sorted(target_groups.items())
    }
    gate = config["decision_gate"]
    checks = {
        "all_level_limits_ok": all(bool(row["qci_level_limit_ok"]) for row in trials),
        "all_penalties_dominate_flip_bound": all(
            bool(row["penalty_dominates_flip_bound"]) for row in trials
        ),
        "minimum_coefficient_retention": min(
            float(row["coefficient_retention_fraction"]) for row in trials
        )
        >= float(gate["minimum_coefficient_retention_fraction"]),
        "reference_support": support_fraction
        >= float(gate["minimum_reference_support_fraction"]),
        "fixed_k_competitiveness": fixed_fraction
        >= float(gate["minimum_fixed_k_frontier_competitive_fraction"]),
        "per_target_competitiveness": min(target_fractions.values())
        >= float(gate["minimum_per_target_frontier_competitive_fraction"]),
        "variable_k_competitiveness": variable_fraction
        >= float(gate["minimum_variable_k_frontier_competitive_fraction"]),
    }
    return {
        "trial_condition_count": len(trials),
        "fixed_k_cell_count": len(cells),
        "variable_k_model_count": len(variable),
        "reference_supported_by_grid_fraction": support_fraction,
        "fixed_k_frontier_competitive_fraction": fixed_fraction,
        "per_target_frontier_competitive_fraction": target_fractions,
        "variable_k_frontier_competitive_fraction": variable_fraction,
        "minimum_cardinality_valid_read_fraction": min(
            float(row["cardinality_valid_read_fraction"]) for row in trials
        ),
        "mean_cardinality_valid_read_fraction": sum(
            float(row["cardinality_valid_read_fraction"]) for row in trials
        ) / len(trials),
        "minimum_coefficient_retention_fraction": min(
            float(row["coefficient_retention_fraction"]) for row in trials
        ),
        "maximum_bqm_variable_count": max(int(row["bqm_variable_count"]) for row in trials),
        "maximum_bqm_interaction_count": max(
            int(row["bqm_interaction_count"]) for row in trials
        ),
        "maximum_quantized_dynamic_range": max(
            float(row["quantized_dynamic_range"]) for row in trials
        ),
        "best_condition_counts": dict(
            sorted(
                Counter(
                    str(row["best_quality_weight"])
                    for row in cells
                    if row["best_quality_weight"] != ""
                ).items()
            )
        ),
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    text = f"""# Stage83 Quality-shell QUBO Gate

The fixed quality-shell grid made the historical reference a one-swap local
minimum in `{summary['reference_supported_by_grid_fraction']:.4f}` of fixed-k
cells. Cold local emulation recovered a frontier-competitive candidate in
`{summary['fixed_k_frontier_competitive_fraction']:.4f}` of fixed-k cells and
`{summary['variable_k_frontier_competitive_fraction']:.4f}` of variable-k
models.

All candidates were checked against the original integer deficit threshold.
Maximum size was `{summary['maximum_bqm_variable_count']}` variables and
`{summary['maximum_bqm_interaction_count']}` interactions; minimum coefficient
retention was `{summary['minimum_coefficient_retention_fraction']:.6f}`.

Limited Dirac-3 calibration authorized:
`{result['decision']['limited_qci_calibration_authorized']}`. This remains a
post-hoc hybrid candidate-generation study, not an exact encoding, efficacy
validation, speedup result, or quantum-advantage claim.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = s82.read_json(config_path)
    for name in ("stage82_result", "stage82_audit", "stage79_physical_audit"):
        s82.verified(root, config["inputs"][name], name)
    stage82 = s82.read_json(root / config["inputs"]["stage82_audit"]["path"])
    if stage82["status"] != "stage82_lagrangian_fixed_k_qubo_independent_audit_ok":
        raise ValueError("Stage83 requires the passing Stage82 audit")
    source_cells = s81.canonical_cells(config, root)
    trials, cells, variable = run_screen(config, source_cells)
    summary = summarize(config, trials, cells, variable)
    outputs = config["outputs"]
    paths = {
        "trial_metrics_csv": root / outputs["trial_metrics_csv"],
        "fixed_k_metrics_csv": root / outputs["fixed_k_metrics_csv"],
        "variable_k_metrics_csv": root / outputs["variable_k_metrics_csv"],
    }
    s82.write_csv(paths["trial_metrics_csv"], trials)
    s82.write_csv(paths["fixed_k_metrics_csv"], cells)
    s82.write_csv(paths["variable_k_metrics_csv"], variable)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage83_quality_shell_qubo_gate_complete",
        "summary": summary,
        "decision": {
            "quality_shell_hybrid_formulation_frozen": True,
            "limited_qci_calibration_authorized": bool(summary["gate_passed"]),
            "full_qci_production_authorized": False,
            "quantum_advantage_claim_authorized": False,
            "remaining_qci_trial_seconds_before_calibration": 166,
        },
        "data_boundary": {
            "historical_development_models_read": len(source_cells),
            "fixed_k_cells_read": len(cells),
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "qci_cloud_queries": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {},
    }
    for label, path in paths.items():
        result["outputs"][label] = {
            "path": outputs[label],
            "sha256": s82.sha256(path),
            "size_bytes": path.stat().st_size,
        }
    result_path = root / outputs["result_json"]
    s82.write_json(result_path, result)
    report_path = root / outputs["report_md"]
    write_report(report_path, result)
    result["outputs"]["report_md"] = {
        "path": outputs["report_md"],
        "sha256": s82.sha256(report_path),
        "size_bytes": report_path.stat().st_size,
    }
    s82.write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/stage83_quality_shell_qubo_gate.json"
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = run((root / args.config).resolve(), root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
