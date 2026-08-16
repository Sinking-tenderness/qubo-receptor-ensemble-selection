"""Compress Stage68 QUBO precision while auditing optimum preservation."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json, write_json  # noqa: F401 (deduped)
import argparse
import csv
import hashlib
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_stage42d_bace1_large_pool_qubo_screen import (
    bedroc_metrics,
    rank_cube,
)
from scripts.run_stage64_cross_target_uncertainty_shrunk_qubo import (
    TOLERANCE,
    jackknife_pair_statistics,
    load_target,
)
from scripts.run_stage68_quality_plateau_portfolio_qubo import (
    PortfolioMilp,
    expanded_qubo_summary,
    factorized_energy,
    integerize_quality,
    quality_plateau,
    stable_redundancy,
    subset_jaccard,
    subset_name,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()




def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))




def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verified(root: Path, value: dict[str, Any], label: str) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage69 frozen {label} identity differs: {path}")
    return path


def model_id(scale: int) -> str:
    return f"quality_scale_{scale}"


def selected_indices(value: str, receptor_ids: list[str]) -> tuple[int, ...]:
    lookup = {item: index for index, item in enumerate(receptor_ids)}
    return tuple(sorted(lookup[item] for item in value.split("+") if item))


def report_text(result: dict[str, Any]) -> str:
    selected = result["selected_compression"]
    near_miss = result["best_uniform_near_miss"]
    displayed = selected or near_miss
    direct = result["direct_qpu_gate"]
    return rf"""# Stage69 QUBO precision compression

## Question

How far can the Stage68 quality-floor QUBO coefficient precision be reduced without changing its feasible baseline, selected subsets, or held-out screening behavior?

## Frozen screen

The conservative integer quality constraint was evaluated at scales

$$
q\in\{{31,63,127,255,511,1023,2047,4095\}}.
$$

For each receptor deficit $d_i$, Stage69 uses $c_i=\lceil qd_i\rceil$ and accepts only states satisfying $\sum_i c_i x_i\le D$. This rounding direction guarantees that every integer-feasible state also satisfies the original continuous Stage68 quality floor.

## Result

- Compression gate passed: `{bool(selected)}`.
- Smallest uniformly feasible scale: `{displayed.get('quality_integer_scale', 'none')}`.
- Feasible cells: `{displayed.get('feasible_cell_count', 0)}/80`.
- Exact subset matches: `{displayed.get('exact_subset_match_count', 0)}/80`.
- Mean subset Jaccard versus continuous Stage68: `{displayed.get('mean_subset_jaccard_vs_continuous', float('nan')):.6f}`.
- Mean absolute BEDROC20 gap: `{displayed.get('mean_absolute_holdout_bedroc_gap', float('nan')):.6f}`.
- Maximum coefficient dynamic range: `{displayed.get('maximum_coefficient_dynamic_range', float('nan')):.6g}`.
- Compression factor versus scale 4095: `{displayed.get('dynamic_range_compression_factor_vs_4095', float('nan')):.3f}x`.

## Hardware boundary

- Direct-QPU precision gate: `{direct['direct_qpu_precision_gate_passed']}`.
- Direct-QPU execution authorized: `{result['decision']['direct_qpu_execution_authorized']}`.
- Compact hybrid or gate-model prototype authorized: `{result['decision']['compact_solver_prototype_authorized']}`.

Stage69 freezes a smaller logical QUBO only when all 80 historical development cells remain feasible and fidelity gates pass. It does not establish embedding feasibility, hardware sampling quality, solver speedup, or quantum advantage.
"""


def compute(config: dict[str, Any], root: Path) -> dict[str, Any]:
    implementation_paths = {
        key: verified(root, value, key)
        for key, value in config["implementation"].items()
    }
    input_paths = {
        key: verified(root, value, key) for key, value in config["inputs"].items()
    }
    stage68_config = read_json(input_paths["stage68_config"])
    stage68_result = read_json(input_paths["stage68_result"])
    stage68_audit = read_json(input_paths["stage68_audit"])
    if not stage68_result["route_gate"][
        "quality_plateau_qubo_freeze_authorized"
    ]:
        raise ValueError("Stage69 requires a frozen Stage68 objective")
    if stage68_audit.get("status") != (
        "stage68_quality_plateau_portfolio_qubo_independent_audit_ok"
    ):
        raise ValueError("Stage69 requires the Stage68 independent audit")
    selected_multiplier = float(
        stage68_result["selected_candidate"]["uncertainty_multiplier"]
    )
    if selected_multiplier != float(config["development"]["uncertainty_multiplier"]):
        raise ValueError("Stage69 multiplier differs from frozen Stage68")
    stage64_config_path = verified(
        root, stage68_config["inputs"]["stage64_config"], "Stage64 config"
    )
    stage64_config = read_json(stage64_config_path)
    reference_rows = read_csv(input_paths["stage68_qubo_fidelity"])
    reference = {
        (
            row["target_id"],
            int(row["outer_fold"]),
            int(row["subset_size"]),
        ): row
        for row in reference_rows
    }
    development = config["development"]
    targets = [str(value) for value in development["target_order"]]
    subset_sizes = [int(value) for value in development["candidate_k_values"]]
    scales = [int(value) for value in development["quality_integer_scales"]]
    alpha = float(development["bedroc_alpha"])
    qubo = config["qubo_encoding"]
    loaded = {
        target: load_target(root, target, stage64_config["targets"][target])
        for target in targets
    }
    rows: list[dict[str, Any]] = []
    internal_models: dict[tuple[int, str, int, int], dict[str, Any]] = {}
    continuous_certificate_count = 0
    quantized_certificate_count = 0
    for target_id in targets:
        target = loaded[target_id]
        ligand_ids = target["ligand_ids"]
        labels = target["labels"]
        receptor_ids = target["receptor_ids"]
        for outer_fold in range(int(development["outer_fold_count"])):
            train_mask = np.asarray(
                [target["outer"][ligand_id] != outer_fold for ligand_id in ligand_ids]
            )
            holdout_mask = ~train_mask
            ranks = rank_cube(target["scores"], train_mask)
            train_rows = [
                row for row, keep in zip(target["ligands"], train_mask) if keep
            ]
            statistics_ = jackknife_pair_statistics(
                target["scores"][:, train_mask, :],
                labels[train_mask],
                train_rows,
                alpha,
                int(development["jackknife_block_count"]),
                int(development["jackknife_seed_base"]) + outer_fold,
            )
            utility = statistics_["full_singleton"]
            spread = statistics_["singleton_spread"]
            redundancy = stable_redundancy(ranks, train_mask)
            workspace = PortfolioMilp(redundancy)
            for subset_size in subset_sizes:
                plateau = quality_plateau(
                    utility, spread, subset_size, selected_multiplier
                )
                continuous, continuous_record = workspace.solve_lower_quality(
                    utility,
                    subset_size,
                    subset_size * plateau["quality_floor"],
                    float(development["milp_time_limit_seconds"]),
                )
                continuous_certificate_count += 1
                continuous_holdout = bedroc_metrics(
                    ranks[:, holdout_mask, :],
                    labels[holdout_mask],
                    continuous,
                    alpha,
                )
                pair_off = plateau["baseline_subset"]
                pair_off_holdout = bedroc_metrics(
                    ranks[:, holdout_mask, :],
                    labels[holdout_mask],
                    pair_off,
                    alpha,
                )
                reference_row = reference[(target_id, outer_fold, subset_size)]
                if subset_name(continuous, receptor_ids) != reference_row[
                    "continuous_subset"
                ]:
                    raise ValueError("Stage69 continuous subset differs from Stage68")
                for scale in scales:
                    base = {
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "subset_size": subset_size,
                        "model_id": model_id(scale),
                        "quality_integer_scale": scale,
                        "continuous_subset": subset_name(
                            continuous, receptor_ids
                        ),
                        "pair_off_subset": subset_name(pair_off, receptor_ids),
                        "continuous_holdout_robust_bedroc": continuous_holdout[
                            "robust_bedroc_composite"
                        ],
                        "pair_off_holdout_robust_bedroc": pair_off_holdout[
                            "robust_bedroc_composite"
                        ],
                        "continuous_milp_gap": continuous_record["mip_gap"],
                    }
                    try:
                        integerized = integerize_quality(
                            utility,
                            subset_size,
                            plateau["quality_floor"],
                            scale,
                        )
                    except ValueError:
                        rows.append(
                            {
                                **base,
                                "status": "pair_off_infeasible_after_conservative_rounding",
                                "quantized_subset": "",
                            }
                        )
                        continue
                    quantized, solver_record = workspace.solve_upper_deficit(
                        integerized["deficits"],
                        subset_size,
                        integerized["maximum_deficit"],
                        utility,
                        float(development["milp_time_limit_seconds"]),
                    )
                    quantized_certificate_count += 1
                    quantized_holdout = bedroc_metrics(
                        ranks[:, holdout_mask, :],
                        labels[holdout_mask],
                        quantized,
                        alpha,
                    )
                    actual_quality = float(np.mean(utility[list(quantized)]))
                    scale_summary = expanded_qubo_summary(
                        redundancy,
                        integerized["deficits"],
                        integerized["maximum_deficit"],
                        integerized["slack_weights"],
                        subset_size,
                        float(qubo["cardinality_penalty"]),
                        float(qubo["quality_penalty"]),
                    )
                    integer_deficit = int(
                        np.sum(integerized["deficits"][list(quantized)])
                    )
                    slack_value = integerized["maximum_deficit"] - integer_deficit
                    energy = factorized_energy(
                        quantized,
                        slack_value,
                        redundancy,
                        integerized["deficits"],
                        integerized["maximum_deficit"],
                        subset_size,
                        float(qubo["cardinality_penalty"]),
                        float(qubo["quality_penalty"]),
                    )
                    redundancy_sum = float(
                        sum(
                            redundancy[left, right]
                            for left, right in itertools.combinations(quantized, 2)
                        )
                    )
                    residual = abs(energy - redundancy_sum)
                    rows.append(
                        {
                            **base,
                            "status": "ok",
                            "quantized_subset": subset_name(
                                quantized, receptor_ids
                            ),
                            "exact_subset_match": quantized == continuous,
                            "subset_jaccard_vs_continuous": subset_jaccard(
                                quantized, continuous
                            ),
                            "quantized_holdout_robust_bedroc": quantized_holdout[
                                "robust_bedroc_composite"
                            ],
                            "quantized_minus_continuous_holdout_bedroc": quantized_holdout[
                                "robust_bedroc_composite"
                            ]
                            - continuous_holdout["robust_bedroc_composite"],
                            "quantized_minus_pair_off_holdout_bedroc": quantized_holdout[
                                "robust_bedroc_composite"
                            ]
                            - pair_off_holdout["robust_bedroc_composite"],
                            "actual_quality_floor_margin": actual_quality
                            - plateau["quality_floor"],
                            "integer_deficit": integer_deficit,
                            "maximum_integer_deficit": integerized[
                                "maximum_deficit"
                            ],
                            "slack_bit_count": len(
                                integerized["slack_weights"]
                            ),
                            "logical_variable_count": scale_summary[
                                "logical_variable_count"
                            ],
                            "quadratic_coefficient_count": scale_summary[
                                "quadratic_coefficient_count"
                            ],
                            "coefficient_dynamic_range": scale_summary[
                                "coefficient_dynamic_range"
                            ],
                            "factorized_energy_residual": residual,
                            "quantized_milp_gap": solver_record["mip_gap"],
                        }
                    )
                    internal_models[(scale, target_id, outer_fold, subset_size)] = {
                        "receptor_ids": receptor_ids,
                        "selected_subset": quantized,
                        "redundancy": redundancy,
                        "integerized": integerized,
                        "scale_summary": scale_summary,
                        "slack_value": slack_value,
                        "energy": energy,
                        "redundancy_sum": redundancy_sum,
                        "residual": residual,
                    }
            print(
                json.dumps(
                    {"target_id": target_id, "outer_fold": outer_fold},
                    sort_keys=True,
                ),
                flush=True,
            )
    summaries: list[dict[str, Any]] = []
    gates = config["compression_gate"]
    for scale in scales:
        scale_rows = [row for row in rows if int(row["quality_integer_scale"]) == scale]
        valid = [row for row in scale_rows if row["status"] == "ok"]
        summary: dict[str, Any] = {
            "model_id": model_id(scale),
            "quality_integer_scale": scale,
            "cell_count": len(scale_rows),
            "feasible_cell_count": len(valid),
            "pair_off_infeasible_cell_count": len(scale_rows) - len(valid),
        }
        if valid:
            summary.update(
                {
                    "exact_subset_match_count": sum(
                        str(row["exact_subset_match"]).lower() == "true"
                        for row in valid
                    ),
                    "mean_subset_jaccard_vs_continuous": statistics.fmean(
                        float(row["subset_jaccard_vs_continuous"])
                        for row in valid
                    ),
                    "minimum_subset_jaccard_vs_continuous": min(
                        float(row["subset_jaccard_vs_continuous"])
                        for row in valid
                    ),
                    "mean_absolute_holdout_bedroc_gap": statistics.fmean(
                        abs(
                            float(
                                row[
                                    "quantized_minus_continuous_holdout_bedroc"
                                ]
                            )
                        )
                        for row in valid
                    ),
                    "maximum_absolute_holdout_bedroc_gap": max(
                        abs(
                            float(
                                row[
                                    "quantized_minus_continuous_holdout_bedroc"
                                ]
                            )
                        )
                        for row in valid
                    ),
                    "mean_holdout_gain_over_pair_off": statistics.fmean(
                        float(row["quantized_minus_pair_off_holdout_bedroc"])
                        for row in valid
                    ),
                    "minimum_actual_quality_floor_margin": min(
                        float(row["actual_quality_floor_margin"])
                        for row in valid
                    ),
                    "maximum_logical_variable_count": max(
                        int(row["logical_variable_count"]) for row in valid
                    ),
                    "maximum_quadratic_coefficient_count": max(
                        int(row["quadratic_coefficient_count"])
                        for row in valid
                    ),
                    "maximum_coefficient_dynamic_range": max(
                        float(row["coefficient_dynamic_range"])
                        for row in valid
                    ),
                    "maximum_factorized_energy_residual": max(
                        float(row["factorized_energy_residual"])
                        for row in valid
                    ),
                }
            )
        summary["compression_gate_passed"] = bool(
            len(valid) >= int(gates["minimum_feasible_cell_count"])
            and valid
            and int(summary["exact_subset_match_count"])
            >= int(gates["minimum_exact_subset_match_count"])
            and float(summary["mean_subset_jaccard_vs_continuous"])
            >= float(gates["minimum_mean_subset_jaccard"])
            and float(summary["minimum_subset_jaccard_vs_continuous"])
            >= float(gates["minimum_subset_jaccard"])
            and float(summary["mean_absolute_holdout_bedroc_gap"])
            <= float(gates["maximum_mean_absolute_holdout_bedroc_gap"])
            and float(summary["maximum_absolute_holdout_bedroc_gap"])
            <= float(gates["maximum_absolute_holdout_bedroc_gap"])
            and float(summary["minimum_actual_quality_floor_margin"])
            >= -float(gates["maximum_quality_floor_violation"])
            and float(summary["maximum_coefficient_dynamic_range"])
            <= float(gates["maximum_compressed_dynamic_range"])
            and float(summary["maximum_factorized_energy_residual"])
            <= float(gates["maximum_factorized_energy_residual"])
        )
        summaries.append(summary)
    reference_summary = next(
        row
        for row in summaries
        if int(row["quality_integer_scale"])
        == int(development["reference_quality_integer_scale"])
    )
    for summary in summaries:
        if "maximum_coefficient_dynamic_range" in summary:
            summary["dynamic_range_compression_factor_vs_4095"] = float(
                reference_summary["maximum_coefficient_dynamic_range"]
            ) / float(summary["maximum_coefficient_dynamic_range"])
        else:
            summary["dynamic_range_compression_factor_vs_4095"] = 0.0
    eligible = [row for row in summaries if row["compression_gate_passed"]]
    selected = (
        min(eligible, key=lambda row: int(row["quality_integer_scale"]))
        if eligible
        else {}
    )
    uniformly_feasible = [
        row
        for row in summaries
        if int(row["feasible_cell_count"])
        >= int(gates["minimum_feasible_cell_count"])
    ]
    near_miss = (
        min(
            uniformly_feasible,
            key=lambda row: int(row["quality_integer_scale"]),
        )
        if uniformly_feasible
        else {}
    )
    selected_scale = int(selected["quality_integer_scale"]) if selected else 0
    selected_target_rows: list[dict[str, Any]] = []
    if selected:
        selected_cells = [
            row
            for row in rows
            if int(row["quality_integer_scale"]) == selected_scale
            and row["status"] == "ok"
        ]
        for target_id in targets:
            current = [
                row for row in selected_cells if row["target_id"] == target_id
            ]
            gains = [
                float(row["quantized_minus_pair_off_holdout_bedroc"])
                for row in current
            ]
            selected_target_rows.append(
                {
                    "target_id": target_id,
                    "quality_integer_scale": selected_scale,
                    "cell_count": len(current),
                    "mean_holdout_gain_over_pair_off": statistics.fmean(gains),
                    "minimum_holdout_gain_over_pair_off": min(gains),
                    "cell_count_within_0p01_of_pair_off": sum(
                        value >= -0.01 - TOLERANCE for value in gains
                    ),
                    "exact_subset_match_count": sum(
                        str(row["exact_subset_match"]).lower() == "true"
                        for row in current
                    ),
                    "mean_subset_jaccard_vs_continuous": statistics.fmean(
                        float(row["subset_jaccard_vs_continuous"])
                        for row in current
                    ),
                }
            )
    model_records: list[dict[str, Any]] = []
    model_scale = int((selected or near_miss).get("quality_integer_scale", 0))
    if model_scale:
        reference_k = int(qubo["reference_model_k"])
        for target_id in targets:
            for outer_fold in range(int(development["outer_fold_count"])):
                model = internal_models[
                    (model_scale, target_id, outer_fold, reference_k)
                ]
                integerized = model["integerized"]
                model_records.append(
                    {
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "reference_k": reference_k,
                        "quality_integer_scale": model_scale,
                        "receptor_ids": model["receptor_ids"],
                        "selected_subset": subset_name(
                            model["selected_subset"], model["receptor_ids"]
                        ),
                        "integer_deficits": [
                            int(value) for value in integerized["deficits"]
                        ],
                        "maximum_integer_deficit": integerized[
                            "maximum_deficit"
                        ],
                        "slack_weights": integerized["slack_weights"],
                        "selected_slack_value": model["slack_value"],
                        "selected_factorized_energy": model["energy"],
                        "selected_redundancy_sum": model["redundancy_sum"],
                        "energy_residual": model["residual"],
                        "stable_redundancy_upper_triangle": [
                            float(model["redundancy"][left, right])
                            for left, right in itertools.combinations(
                                range(len(model["receptor_ids"])), 2
                            )
                        ],
                        "qubo_scale": model["scale_summary"],
                    }
                )
    direct_gate = {
        "maximum_permitted_dynamic_range": float(
            config["direct_qpu_gate"]["maximum_coefficient_dynamic_range"]
        ),
        "observed_maximum_dynamic_range": (
            float(
                (selected or near_miss).get(
                    "maximum_coefficient_dynamic_range", 0.0
                )
            )
        ),
        "direct_qpu_precision_gate_passed": bool(selected)
        and float(selected["maximum_coefficient_dynamic_range"])
        <= float(
            config["direct_qpu_gate"]["maximum_coefficient_dynamic_range"]
        ),
    }
    compression_authorized = bool(selected)
    output_paths = {
        key: root / str(value) for key, value in config["outputs"].items()
    }
    write_csv(output_paths["cell_metrics_csv"], rows)
    write_csv(output_paths["scale_summary_csv"], summaries)
    if selected_target_rows:
        write_csv(output_paths["target_summary_csv"], selected_target_rows)
    model_record = {
        "schema_version": "1.0",
        "algorithm_id": "stage68-quality-floor-qubo-optimum-preserving-precision-compression-v1",
        "selected_quality_integer_scale": selected_scale,
        "diagnostic_model_quality_integer_scale": model_scale,
        "model_role": (
            "selected_compression"
            if selected
            else "diagnostic_uniform_near_miss"
        ),
        "cardinality_penalty": float(qubo["cardinality_penalty"]),
        "quality_penalty": float(qubo["quality_penalty"]),
        "model_count": len(model_records),
        "models": model_records,
    }
    write_json(output_paths["model_record_json"], model_record)
    payload = {
        "scale_summaries": summaries,
        "selected_compression": selected,
        "best_uniform_near_miss": near_miss,
        "selected_target_summary": selected_target_rows,
        "direct_qpu_gate": direct_gate,
    }
    result = {
        "schema_version": "1.0",
        "status": "stage69_qubo_precision_compression_complete",
        "experiment_class": "post-hoc hardware-encoding compression on frozen historical development models",
        "config": descriptor(root, root / "configs/stage69_qubo_precision_compression.json"),
        "implementation": {
            key: descriptor(root, path) for key, path in implementation_paths.items()
        },
        "inputs": {key: descriptor(root, path) for key, path in input_paths.items()},
        "scale_count": len(scales),
        "cell_metric_count": len(rows),
        "continuous_milp_certificate_count": continuous_certificate_count,
        "quantized_milp_certificate_count": quantized_certificate_count,
        "selected_compression": selected,
        "best_uniform_near_miss": near_miss,
        "direct_qpu_gate": direct_gate,
        "compression_gate": {
            "compressed_qubo_freeze_authorized": compression_authorized
        },
        "decision": {
            "compact_solver_prototype_authorized": compression_authorized,
            "direct_qpu_execution_authorized": compression_authorized
            and direct_gate["direct_qpu_precision_gate_passed"],
            "new_target_preregistration_remains_authorized": stage68_result[
                "decision"
            ]["future_new_target_preregistration_authorized"],
            "quantum_advantage_claim_authorized": False,
            "next_action": (
                "freeze the compressed model for embedding/noise simulation and preregister one new target"
                if compression_authorized
                else "retain scale 4095 and redesign the quality constraint encoding"
            ),
        },
        "data_boundary": {
            "historical_development_targets_read": len(targets),
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "analysis_payload_sha256": canonical_sha256(payload),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    result["outputs"] = {
        "cell_metrics_csv": descriptor(root, output_paths["cell_metrics_csv"]),
        "scale_summary_csv": descriptor(root, output_paths["scale_summary_csv"]),
        "model_record_json": descriptor(root, output_paths["model_record_json"]),
    }
    if selected_target_rows:
        result["outputs"]["target_summary_csv"] = descriptor(
            root, output_paths["target_summary_csv"]
        )
    write_json(output_paths["result_json"], result)
    output_paths["report_md"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["report_md"].write_text(
        report_text(result), encoding="utf-8", newline="\n"
    )
    result["outputs"]["report_md"] = descriptor(root, output_paths["report_md"])
    write_json(output_paths["result_json"], result)
    return result


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    expected = root / "configs/stage69_qubo_precision_compression.json"
    if config_path != expected.resolve():
        raise ValueError("Stage69 must run from its frozen repository config")
    config = read_json(config_path)
    result_path = root / str(config["outputs"]["result_json"])
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"Stage69 result exists: {result_path}")
    result = compute(config, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage69_qubo_precision_compression.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
