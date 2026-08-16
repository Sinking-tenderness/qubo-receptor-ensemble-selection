"""Adjudicate float32-safe global variable-k QUBO routes for Dirac-3."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json  # noqa: F401 (deduped)
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import dimod
import numpy as np
from dwave.samplers import SteepestDescentSolver, TabuSampler

try:
    import scripts.run_stage75_explicit_variable_k_cqm as s75
except ImportError:
    import run_stage75_explicit_variable_k_cqm as s75


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()




def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Stage81 refuses to write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verified(root: Path, descriptor: dict[str, Any], label: str) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file() or sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage81 {label} identity differs: {path}")
    if path.stat().st_size != int(descriptor["size_bytes"]):
        raise ValueError(f"Stage81 {label} size differs: {path}")
    return path


def canonical_cells(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    inputs = config["inputs"]
    stage77 = read_json(verified(root, inputs["stage77_config"], "Stage77 config"))
    source = read_json(verified(root, inputs["stage72_model_record"], "Stage72 model"))
    workloads = s75.read_csv(
        verified(root, inputs["stage74_workload_metrics"], "Stage74 workloads")
    )
    comparisons = s75.read_csv(
        verified(root, inputs["stage74_cell_comparison"], "Stage74 comparisons")
    )
    trials = s75.read_csv(
        verified(root, inputs["stage74_solver_trials"], "Stage74 trials")
    )
    quantile = float(config["experiment"]["canonical_reward_quantile"])
    cells: list[dict[str, Any]] = []
    for record in source["models"]:
        model = s75.load_model(record)
        frontiers = s75.source_frontiers(
            model,
            workloads,
            comparisons,
            trials,
            stage77["frozen_cqm"]["quality_regime"],
        )
        reward = float(s75.reward_order_statistic(model, quantile)["reward"])
        cells.append(
            {
                "model": model,
                "frontiers": frontiers,
                "reward": reward,
                "cqm": s75.build_cqm(model, frontiers, reward),
            }
        )
    return cells


def nonzero_biases(bqm: dimod.BinaryQuadraticModel) -> list[float]:
    values = [abs(float(value)) for value in bqm.linear.values()]
    values.extend(abs(float(value)) for value in bqm.quadratic.values())
    return [value for value in values if value > 1e-15]


def fixed_precision_bqm(
    bqm: dimod.BinaryQuadraticModel, bits: int
) -> tuple[dimod.BinaryQuadraticModel, dict[str, Any]]:
    biases = nonzero_biases(bqm)
    full_scale = max(biases)
    levels = 2 ** (bits - 1) - 1

    def quantize(value: float) -> float:
        return float(np.float32(round(float(value) / full_scale * levels) / levels))

    output = dimod.BinaryQuadraticModel(
        {name: quantize(value) for name, value in bqm.linear.items()},
        {pair: quantize(value) for pair, value in bqm.quadratic.items()},
        quantize(float(bqm.offset)),
        dimod.BINARY,
    )
    retained = nonzero_biases(output)
    return output, {
        "source_full_scale": full_scale,
        "retained_nonzero_bias_count": len(retained),
        "source_nonzero_bias_count": len(biases),
        "quantized_dynamic_range": max(retained) / min(retained),
    }


def encoded_feasible_start(
    bqm: dimod.BinaryQuadraticModel, assignment: dict[str, int]
) -> dict[Any, int]:
    fixed = {name: int(value) for name, value in assignment.items() if name in bqm.variables}
    reduced = bqm.copy()
    reduced.fix_variables(fixed)
    if len(reduced.variables) <= 20:
        auxiliary = dimod.ExactSolver().sample(reduced).first.sample
    else:
        zero = {name: 0 for name in reduced.variables}
        auxiliary = SteepestDescentSolver().sample(
            reduced, initial_states=[zero]
        ).first.sample
    output: dict[Any, int] = dict(fixed)
    output.update({name: int(value) for name, value in auxiliary.items()})
    return output


def evaluate_samples(
    cqm: dimod.ConstrainedQuadraticModel,
    inverter: Any,
    sampleset: dimod.SampleSet,
    baseline: float,
) -> dict[str, Any]:
    total = 0
    feasible = 0
    best_feasible = math.inf
    raw_best_feasible = False
    for index, datum in enumerate(
        sampleset.data(fields=["sample", "num_occurrences"])
    ):
        sample = inverter(datum.sample)
        occurrences = int(datum.num_occurrences)
        is_feasible = bool(cqm.check_feasible(sample, atol=1e-6, rtol=1e-6))
        if index == 0:
            raw_best_feasible = is_feasible
        total += occurrences
        feasible += occurrences * int(is_feasible)
        if is_feasible:
            best_feasible = min(best_feasible, float(cqm.objective.energy(sample)))
    return {
        "read_count": total,
        "feasible_read_fraction": feasible / total,
        "raw_best_feasible": raw_best_feasible,
        "best_feasible_objective": None if math.isinf(best_feasible) else best_feasible,
        "frontier_competitive": best_feasible <= baseline + 1e-8,
    }


def direct_rows(config: dict[str, Any], cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    protocol = config["direct_penalty_screen"]
    rows: list[dict[str, Any]] = []
    for factor in protocol["penalty_factors"]:
        for bits in protocol["signed_precision_bits"]:
            for cell_index, cell in enumerate(cells):
                cqm = cell["cqm"]
                objective_scale = max(
                    [abs(float(value)) for value in cqm.objective.quadratic.values()]
                    + [1e-12]
                )
                bqm, inverter = dimod.cqm_to_bqm(
                    cqm, lagrange_multiplier=float(factor) * objective_scale
                )
                quantized, precision = fixed_precision_bqm(bqm, int(bits))
                model = cell["model"]
                frontiers = cell["frontiers"]
                reward = float(cell["reward"])
                baseline_k, baseline_frontier = min(
                    frontiers.items(),
                    key=lambda item: s75.variable_energy(
                        model, tuple(item[1]["reference_subset"]), reward
                    ),
                )
                baseline = s75.variable_energy(
                    model, tuple(baseline_frontier["reference_subset"]), reward
                )
                initial = encoded_feasible_start(
                    bqm,
                    s75.assignment(
                        model,
                        frontiers,
                        tuple(baseline_frontier["reference_subset"]),
                    ),
                )
                cold = TabuSampler().sample(
                    quantized,
                    num_reads=int(protocol["cold_tabu_reads"]),
                    timeout=int(protocol["tabu_timeout_milliseconds"]),
                    seed=int(protocol["seed_base"]) + cell_index,
                )
                warm = TabuSampler().sample(
                    quantized,
                    num_reads=int(protocol["warm_tabu_reads"]),
                    timeout=int(protocol["tabu_timeout_milliseconds"]),
                    initial_states=[initial],
                    initial_states_generator="tile",
                    seed=int(protocol["seed_base"]) + 1000 + cell_index,
                )
                cold_metrics = evaluate_samples(cqm, inverter, cold, baseline)
                warm_metrics = evaluate_samples(cqm, inverter, warm, baseline)
                record = model["record"]
                rows.append(
                    {
                        "target_id": str(record["target_id"]),
                        "outer_fold": int(record["outer_fold"]),
                        "penalty_factor": float(factor),
                        "signed_precision_bits": int(bits),
                        "candidate_receptor_count": int(model["count"]),
                        "cqm_variable_count": int(cqm.num_variables()),
                        "bqm_variable_count": int(bqm.num_variables),
                        "bqm_interaction_count": int(bqm.num_interactions),
                        "qci_total_binary_levels": 2 * int(bqm.num_variables),
                        "qci_level_limit_ok": 2 * int(bqm.num_variables)
                        <= int(config["experiment"]["qci_total_level_limit"]),
                        "quantized_dynamic_range": precision["quantized_dynamic_range"],
                        "coefficient_retention_fraction": precision[
                            "retained_nonzero_bias_count"
                        ]
                        / precision["source_nonzero_bias_count"],
                        "baseline_k": int(baseline_k),
                        "baseline_objective": baseline,
                        "cold_raw_best_feasible": cold_metrics["raw_best_feasible"],
                        "cold_feasible_read_fraction": cold_metrics[
                            "feasible_read_fraction"
                        ],
                        "cold_frontier_competitive": cold_metrics[
                            "frontier_competitive"
                        ],
                        "warm_raw_best_feasible": warm_metrics["raw_best_feasible"],
                        "warm_feasible_read_fraction": warm_metrics[
                            "feasible_read_fraction"
                        ],
                        "warm_frontier_competitive": warm_metrics[
                            "frontier_competitive"
                        ],
                    }
                )
    return rows


def conservative_prefilter_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        model = cell["model"]
        record = model["record"]
        deficits = model["deficits"]
        for k, frontier in cell["frontiers"].items():
            per_receptor_limit = int(frontier["quality_threshold"]) // int(k)
            pool = {
                index
                for index, value in enumerate(deficits)
                if int(value) <= per_receptor_limit
            }
            reference = set(int(value) for value in frontier["reference_subset"])
            rows.append(
                {
                    "target_id": str(record["target_id"]),
                    "outer_fold": int(record["outer_fold"]),
                    "k": int(k),
                    "quality_threshold": int(frontier["quality_threshold"]),
                    "per_receptor_limit": per_receptor_limit,
                    "conservative_pool_count": len(pool),
                    "pool_can_select_k": len(pool) >= int(k),
                    "reference_fully_retained": reference <= pool,
                    "reference_retained_fraction": len(reference & pool) / int(k),
                }
            )
    return rows


def aggregate_direct(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["penalty_factor"]), int(row["signed_precision_bits"]))].append(row)
    output: list[dict[str, Any]] = []
    gate = config["decision_gate"]
    for (factor, bits), local in sorted(grouped.items()):
        count = len(local)
        cold_raw = sum(bool(row["cold_raw_best_feasible"]) for row in local) / count
        cold_competitive = sum(
            bool(row["cold_frontier_competitive"]) for row in local
        ) / count
        output.append(
            {
                "penalty_factor": factor,
                "signed_precision_bits": bits,
                "cell_count": count,
                "cold_raw_best_feasible_fraction": cold_raw,
                "cold_mean_feasible_read_fraction": sum(
                    float(row["cold_feasible_read_fraction"]) for row in local
                )
                / count,
                "cold_frontier_competitive_fraction": cold_competitive,
                "warm_raw_best_feasible_fraction": sum(
                    bool(row["warm_raw_best_feasible"]) for row in local
                )
                / count,
                "warm_mean_feasible_read_fraction": sum(
                    float(row["warm_feasible_read_fraction"]) for row in local
                )
                / count,
                "warm_frontier_competitive_fraction": sum(
                    bool(row["warm_frontier_competitive"]) for row in local
                )
                / count,
                "maximum_quantized_dynamic_range": max(
                    float(row["quantized_dynamic_range"]) for row in local
                ),
                "condition_passed": cold_raw
                >= float(gate["minimum_cold_raw_best_feasible_fraction"])
                and cold_competitive
                >= float(gate["minimum_cold_frontier_competitive_fraction"]),
            }
        )
    return output


def write_report(path: Path, result: dict[str, Any]) -> None:
    direct = result["direct_penalty_summary"]
    prefilter = result["conservative_prefilter_summary"]
    best = max(
        direct,
        key=lambda row: (
            float(row["cold_frontier_competitive_fraction"]),
            float(row["cold_raw_best_feasible_fraction"]),
        ),
    )
    text = f"""# Stage81 Dirac Global QUBO Formulation Gate

## Direct CQM-to-BQM route

The screen evaluated `{len(direct)}` penalty/precision conditions over 16
canonical variable-k protein models. The best cold-start condition reached a
raw-best feasible fraction of
`{best['cold_raw_best_feasible_fraction']:.4f}` and a frontier-competitive
fraction of `{best['cold_frontier_competitive_fraction']:.4f}`. No condition
passed the frozen feasibility and competitiveness gate.

## Conservative quality prefilter

The per-receptor inner approximation retained enough receptors in
`{prefilter['pool_can_select_k_count']}/{prefilter['cell_count']}` fixed-k cells.
It fully retained the historical frontier in
`{prefilter['reference_fully_retained_count']}/{prefilter['cell_count']}` cells;
the mean retained fraction was
`{prefilter['mean_reference_retained_fraction']:.4f}`. This route changes the
scientific feasible set too aggressively and is rejected.

## Decision

No additional Dirac-3 global-QUBO submission is authorized. The remaining
trial allocation is preserved. A future physical run requires a formulation
that preserves the original conditional-quality constraint without a
float32-dominating penalty and without discarding the historical frontier.

This is an encoding no-go result, not evidence against the Stage75 scientific
objective or against quantum optimization in general.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    for name in ("stage80_result", "stage80_audit", "stage79_physical_audit"):
        verified(root, config["inputs"][name], name)
    stage80 = read_json(root / config["inputs"]["stage80_audit"]["path"])
    if stage80["status"] != "stage80_local_move_hardness_independent_audit_ok":
        raise ValueError("Stage81 requires the passing Stage80 audit")
    cells = canonical_cells(config, root)
    direct = direct_rows(config, cells)
    prefilter = conservative_prefilter_rows(cells)
    direct_summary = aggregate_direct(config, direct)
    prefilter_summary = {
        "cell_count": len(prefilter),
        "pool_can_select_k_count": sum(
            bool(row["pool_can_select_k"]) for row in prefilter
        ),
        "reference_fully_retained_count": sum(
            bool(row["reference_fully_retained"]) for row in prefilter
        ),
        "mean_reference_retained_fraction": sum(
            float(row["reference_retained_fraction"]) for row in prefilter
        )
        / len(prefilter),
        "minimum_pool_count": min(
            int(row["conservative_pool_count"]) for row in prefilter
        ),
        "maximum_pool_count": max(
            int(row["conservative_pool_count"]) for row in prefilter
        ),
    }
    direct_pass = any(bool(row["condition_passed"]) for row in direct_summary)
    prefilter_pass = (
        prefilter_summary["pool_can_select_k_count"] == len(prefilter)
        and prefilter_summary["reference_fully_retained_count"]
        / len(prefilter)
        >= float(config["decision_gate"]["minimum_prefilter_reference_retention_fraction"])
    )
    outputs = config["outputs"]
    direct_path = root / outputs["direct_metrics_csv"]
    prefilter_path = root / outputs["prefilter_metrics_csv"]
    write_csv(direct_path, direct)
    write_csv(prefilter_path, prefilter)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage81_dirac_global_qubo_formulation_gate_complete",
        "direct_penalty_summary": direct_summary,
        "conservative_prefilter_summary": prefilter_summary,
        "decision": {
            "direct_float32_global_bqm_authorized": direct_pass,
            "conservative_prefilter_global_bqm_authorized": prefilter_pass,
            "additional_qci_hardware_submission_authorized": direct_pass or prefilter_pass,
            "remaining_qci_trial_seconds_preserved": 166,
            "constraint_preserving_reformulation_required": not (direct_pass or prefilter_pass),
        },
        "data_boundary": {
            "historical_development_models_read": len(cells),
            "fixed_k_prefilter_cells_read": len(prefilter),
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "qci_cloud_queries": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {
            "direct_metrics_csv": {
                "path": outputs["direct_metrics_csv"],
                "sha256": sha256(direct_path),
                "size_bytes": direct_path.stat().st_size,
            },
            "prefilter_metrics_csv": {
                "path": outputs["prefilter_metrics_csv"],
                "sha256": sha256(prefilter_path),
                "size_bytes": prefilter_path.stat().st_size,
            },
        },
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
        "--config", default="configs/stage81_dirac_global_qubo_formulation_gate.json"
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = run((root / args.config).resolve(), root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
