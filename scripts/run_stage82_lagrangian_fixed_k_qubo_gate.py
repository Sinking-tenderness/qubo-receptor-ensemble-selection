"""Screen a constraint-preserving hybrid fixed-k QUBO decomposition."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
except ImportError:
    import run_stage75_explicit_variable_k_cqm as s75
    import run_stage81_dirac_global_qubo_formulation_gate as s81


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Stage82 refuses to write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verified(root: Path, descriptor: dict[str, Any], label: str) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file() or sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage82 {label} identity differs: {path}")
    if path.stat().st_size != int(descriptor["size_bytes"]):
        raise ValueError(f"Stage82 {label} size differs: {path}")
    return path


def add_cardinality_penalty(
    bqm: dimod.BinaryQuadraticModel, k: int, multiplier: float
) -> tuple[dimod.BinaryQuadraticModel, float, float]:
    flip_bounds = []
    for variable in bqm.variables:
        bound = abs(float(bqm.get_linear(variable)))
        bound += sum(
            abs(float(value))
            for _, value in bqm.iter_neighborhood(variable)
        )
        flip_bounds.append(bound)
    maximum_flip_bound = max(flip_bounds, default=0.0)
    penalty = max(float(multiplier) * maximum_flip_bound, 1e-6)
    output = bqm.copy()
    variables = list(output.variables)
    for variable in variables:
        output.add_linear(variable, penalty * (1 - 2 * int(k)))
    for left, right in itertools.combinations(variables, 2):
        output.add_quadratic(left, right, 2 * penalty)
    output.offset += penalty * int(k) ** 2
    return output, penalty, maximum_flip_bound


def build_bqm(
    cell: dict[str, Any], k: int, quality_weight: float, multiplier: float
) -> tuple[dimod.BinaryQuadraticModel, dict[str, float]]:
    model = cell["model"]
    reward = float(cell["reward"])
    shifted = np.asarray(model["raw_coefficients"], dtype=float) - reward
    pair_scale = max(float(np.max(np.abs(shifted))), 1e-12)
    threshold = int(cell["frontiers"][k]["quality_threshold"])
    linear = {
        f"x{index:03d}": float(quality_weight)
        * float(model["deficits"][index])
        / max(threshold, 1)
        for index in range(model["count"])
    }
    quadratic = {
        (f"x{left:03d}", f"x{right:03d}"): float(value) / pair_scale
        for (left, right), value in zip(model["pairs"], shifted)
    }
    base = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
    penalized, penalty, flip_bound = add_cardinality_penalty(
        base, k, multiplier
    )
    return penalized, {
        "pair_scale": pair_scale,
        "quality_threshold": float(threshold),
        "cardinality_penalty": penalty,
        "base_maximum_flip_bound": flip_bound,
    }


def sample_condition(
    config: dict[str, Any],
    cell: dict[str, Any],
    k: int,
    quality_weight: float,
    seed: int,
) -> tuple[dict[str, Any], tuple[int, ...] | None]:
    protocol = config["lagrangian_screen"]
    bqm, encoding = build_bqm(
        cell,
        k,
        quality_weight,
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
    row = {
        "target_id": str(model["record"]["target_id"]),
        "outer_fold": int(model["record"]["outer_fold"]),
        "k": int(k),
        "quality_weight": float(quality_weight),
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
    protocol = config["lagrangian_screen"]
    trials: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    candidates: dict[tuple[str, int, int], list[tuple[float, tuple[int, ...], float]]] = (
        defaultdict(list)
    )
    cell_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    condition_index = 0
    for cell in cells:
        model = cell["model"]
        target = str(model["record"]["target_id"])
        fold = int(model["record"]["outer_fold"])
        cell_by_key[(target, fold)] = cell
        for k in cell["frontiers"]:
            for weight in protocol["quality_weights"]:
                row, subset = sample_condition(
                    config,
                    cell,
                    int(k),
                    float(weight),
                    int(protocol["seed_base"]) + condition_index,
                )
                trials.append(row)
                if subset is not None:
                    candidates[(target, fold, int(k))].append(
                        (
                            float(row["best_feasible_objective"]),
                            subset,
                            float(weight),
                        )
                    )
                condition_index += 1
    for cell in cells:
        model = cell["model"]
        target = str(model["record"]["target_id"])
        fold = int(model["record"]["outer_fold"])
        for k, frontier in cell["frontiers"].items():
            local = candidates[(target, fold, int(k))]
            reference = tuple(frontier["reference_subset"])
            baseline = s75.variable_energy(model, reference, float(cell["reward"]))
            if local:
                best_objective, best_subset, best_weight = min(
                    local, key=lambda item: (item[0], item[1], item[2])
                )
            else:
                best_objective, best_subset, best_weight = math.inf, tuple(), math.nan
            cell_rows.append(
                {
                    "target_id": target,
                    "outer_fold": fold,
                    "k": int(k),
                    "candidate_receptor_count": int(model["count"]),
                    "baseline_objective": baseline,
                    "best_feasible_objective": ""
                    if math.isinf(best_objective)
                    else best_objective,
                    "frontier_competitive": best_objective <= baseline + 1e-8,
                    "best_quality_weight": ""
                    if math.isnan(best_weight)
                    else best_weight,
                    "best_subset": ""
                    if not best_subset
                    else s75.subset_name(model, best_subset),
                }
            )
    variable_rows: list[dict[str, Any]] = []
    grouped_cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        grouped_cells[(str(row["target_id"]), int(row["outer_fold"]))].append(row)
    for key, rows in sorted(grouped_cells.items()):
        cell = cell_by_key[key]
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
        best = min(
            feasible,
            key=lambda row: (
                float(row["best_feasible_objective"]),
                int(row["k"]),
                str(row["best_subset"]),
            ),
        ) if feasible else None
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
    variable_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    target_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        target_groups[str(row["target_id"])].append(row)
    target_fractions = {
        target: sum(bool(row["frontier_competitive"]) for row in rows) / len(rows)
        for target, rows in sorted(target_groups.items())
    }
    fixed_fraction = sum(bool(row["frontier_competitive"]) for row in cells) / len(cells)
    variable_fraction = sum(
        bool(row["frontier_competitive"]) for row in variable_rows
    ) / len(variable_rows)
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
        "variable_k_model_count": len(variable_rows),
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
        "best_quality_weight_counts": dict(
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
    decision = result["decision"]
    text = f"""# Stage82 Lagrangian Fixed-k QUBO Gate

## Formulation

Stage82 decomposes the Stage75 variable-k CQM into fixed-k QUBOs. A frozen
quality-weight grid generates candidates, an analytically bounded cardinality
penalty enforces k, and a classical guard rejects every candidate that violates
the original conditional-quality threshold. The original Stage75 objective is
used for all final comparisons across weights and k values.

## Local gate result

- Fixed-k frontier-competitive cells: `{summary['fixed_k_frontier_competitive_fraction']:.4f}`.
- Variable-k frontier-competitive models: `{summary['variable_k_frontier_competitive_fraction']:.4f}`.
- Maximum logical variables: `{summary['maximum_bqm_variable_count']}`.
- Maximum QUBO interactions: `{summary['maximum_bqm_interaction_count']}`.
- Minimum coefficient retention: `{summary['minimum_coefficient_retention_fraction']:.6f}`.

## Decision

Limited Dirac-3 calibration authorized: `{decision['limited_qci_calibration_authorized']}`.
The route is a hybrid candidate-generation protocol, not an exact unconstrained
encoding and not evidence of quantum advantage. A physical run remains limited
to frozen positive and negative controls until hardware fidelity is measured.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    for name in ("stage81_result", "stage81_audit", "stage79_physical_audit"):
        verified(root, config["inputs"][name], name)
    stage81 = read_json(root / config["inputs"]["stage81_audit"]["path"])
    if stage81["status"] != "stage81_dirac_global_qubo_formulation_independent_audit_ok":
        raise ValueError("Stage82 requires the passing Stage81 audit")
    cells = s81.canonical_cells(config, root)
    trials, cell_rows, variable_rows = run_screen(config, cells)
    summary = summarize(config, trials, cell_rows, variable_rows)
    outputs = config["outputs"]
    trial_path = root / outputs["trial_metrics_csv"]
    cell_path = root / outputs["fixed_k_metrics_csv"]
    variable_path = root / outputs["variable_k_metrics_csv"]
    write_csv(trial_path, trials)
    write_csv(cell_path, cell_rows)
    write_csv(variable_path, variable_rows)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage82_lagrangian_fixed_k_qubo_gate_complete",
        "summary": summary,
        "decision": {
            "lagrangian_hybrid_formulation_frozen": True,
            "limited_qci_calibration_authorized": bool(summary["gate_passed"]),
            "full_qci_production_authorized": False,
            "quantum_advantage_claim_authorized": False,
            "remaining_qci_trial_seconds_before_calibration": 166,
        },
        "data_boundary": {
            "historical_development_models_read": len(cells),
            "fixed_k_cells_read": len(cell_rows),
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "qci_cloud_queries": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {},
    }
    for label, local in (
        ("trial_metrics_csv", trial_path),
        ("fixed_k_metrics_csv", cell_path),
        ("variable_k_metrics_csv", variable_path),
    ):
        result["outputs"][label] = {
            "path": outputs[label],
            "sha256": sha256(local),
            "size_bytes": local.stat().st_size,
        }
    result_path = root / outputs["result_json"]
    write_json(result_path, result)
    report_path = root / outputs["report_md"]
    write_report(report_path, result)
    result["outputs"]["report_md"] = {
        "path": outputs["report_md"],
        "sha256": sha256(report_path),
        "size_bytes": report_path.stat().st_size,
    }
    write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/stage82_lagrangian_fixed_k_qubo_gate.json"
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = run((root / args.config).resolve(), root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
