"""Build a tighter exact-penalty encoding for the frozen Stage68 QUBO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_stage42d_bace1_large_pool_qubo_screen import rank_cube
from scripts.run_stage64_cross_target_uncertainty_shrunk_qubo import (
    TOLERANCE,
    jackknife_pair_statistics,
    load_target,
)
from scripts.run_stage68_quality_plateau_portfolio_qubo import (
    integerize_quality,
    quality_plateau,
    redundancy_sum,
    stable_redundancy,
    subset_name,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


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
        raise ValueError(f"Stage70 frozen {label} identity differs: {path}")
    return path


def selected_indices(value: str, receptor_ids: list[str]) -> tuple[int, ...]:
    lookup = {item: index for index, item in enumerate(receptor_ids)}
    selected = tuple(sorted(lookup[item] for item in value.split("+") if item))
    if not selected:
        raise ValueError("Stage70 selected subset is empty")
    return selected


def bounded_slack_weights(maximum: int, cap: int) -> list[int]:
    if maximum < 0 or cap < 1:
        raise ValueError("Stage70 slack bounds must be nonnegative")
    weights: list[int] = []
    covered = 0
    power = 1
    while power <= cap and covered + power <= maximum:
        weights.append(power)
        covered += power
        power *= 2
    while covered < maximum:
        weight = min(cap, maximum - covered)
        if weight > covered + 1:
            raise ValueError("Stage70 slack encoding contains a representability gap")
        weights.append(weight)
        covered += weight
    if sum(weights) != maximum:
        raise ValueError("Stage70 slack encoding does not reach its tight bound")
    return weights


def slack_interval_is_complete(weights: list[int], maximum: int) -> bool:
    covered = 0
    for weight in sorted(weights):
        if weight > covered + 1:
            return False
        covered += weight
    return covered == maximum


def expanded_centered_qubo_summary(
    redundancy: np.ndarray,
    deficits: np.ndarray,
    maximum_deficit: int,
    slack_weights: list[int],
    subset_size: int,
    center: int,
    cardinality_penalty: float,
    quality_penalty: float,
    include_hash: bool = False,
) -> dict[str, Any]:
    receptor_count = len(deficits)
    centered = deficits.astype(float) - int(center)
    slack = np.asarray(slack_weights, dtype=float)
    centered_rhs = int(maximum_deficit) - int(subset_size) * int(center)
    linear_parts = [
        cardinality_penalty * (1 - 2 * subset_size)
        + quality_penalty
        * (centered * centered - 2 * centered_rhs * centered)
    ]
    if len(slack):
        linear_parts.append(
            quality_penalty * (slack * slack - 2 * centered_rhs * slack)
        )
    linear = np.concatenate(linear_parts)
    receptor_upper = np.triu_indices(receptor_count, 1)
    quadratic_parts = [
        2 * cardinality_penalty
        + redundancy[receptor_upper]
        + 2
        * quality_penalty
        * (centered[:, None] * centered[None, :])[receptor_upper]
    ]
    if len(slack):
        quadratic_parts.append(
            (2 * quality_penalty * centered[:, None] * slack[None, :]).ravel()
        )
        slack_upper = np.triu_indices(len(slack), 1)
        quadratic_parts.append(
            (2 * quality_penalty * slack[:, None] * slack[None, :])[
                slack_upper
            ]
        )
    quadratic = np.concatenate(quadratic_parts)
    coefficients = np.abs(np.concatenate([linear, quadratic]))
    coefficients = coefficients[coefficients > TOLERANCE]
    if not len(coefficients):
        raise ValueError("Stage70 QUBO has no nonzero coefficient")
    result: dict[str, Any] = {
        "logical_variable_count": int(receptor_count + len(slack_weights)),
        "receptor_variable_count": int(receptor_count),
        "slack_variable_count": int(len(slack_weights)),
        "linear_coefficient_count": int(np.sum(np.abs(linear) > TOLERANCE)),
        "quadratic_coefficient_count": int(
            np.sum(np.abs(quadratic) > TOLERANCE)
        ),
        "constant": float(
            cardinality_penalty * subset_size**2
            + quality_penalty * centered_rhs**2
        ),
        "minimum_absolute_nonzero_coefficient": float(np.min(coefficients)),
        "maximum_absolute_coefficient": float(np.max(coefficients)),
        "coefficient_dynamic_range": float(
            np.max(coefficients) / np.min(coefficients)
        ),
        "centered_rhs": int(centered_rhs),
    }
    if include_hash:
        result["qubo_sha256"] = canonical_sha256(
            {
                "center": int(center),
                "centered_rhs": int(centered_rhs),
                "linear": [float(value) for value in linear],
                "quadratic": [float(value) for value in quadratic],
                "constant": result["constant"],
            }
        )
    return result


def choose_center(
    redundancy: np.ndarray,
    deficits: np.ndarray,
    maximum_deficit: int,
    slack_weights: list[int],
    subset_size: int,
    cardinality_penalty: float,
    quality_penalty: float,
) -> tuple[int, dict[str, Any]]:
    candidates: list[tuple[tuple[float, float, int], int]] = []
    for center in range(int(np.max(deficits)) + 1):
        summary = expanded_centered_qubo_summary(
            redundancy,
            deficits,
            maximum_deficit,
            slack_weights,
            subset_size,
            center,
            cardinality_penalty,
            quality_penalty,
        )
        candidates.append(
            (
                (
                    float(summary["coefficient_dynamic_range"]),
                    float(summary["maximum_absolute_coefficient"]),
                    center,
                ),
                center,
            )
        )
    center = min(candidates)[1]
    return center, expanded_centered_qubo_summary(
        redundancy,
        deficits,
        maximum_deficit,
        slack_weights,
        subset_size,
        center,
        cardinality_penalty,
        quality_penalty,
        include_hash=True,
    )


def centered_energy(
    subset: tuple[int, ...],
    slack_value: int,
    redundancy: np.ndarray,
    deficits: np.ndarray,
    maximum_deficit: int,
    subset_size: int,
    center: int,
    cardinality_penalty: float,
    quality_penalty: float,
) -> float:
    centered_sum = int(np.sum(deficits[list(subset)])) - center * len(subset)
    centered_rhs = maximum_deficit - center * subset_size
    return float(
        redundancy_sum(subset, redundancy)
        + cardinality_penalty * (len(subset) - subset_size) ** 2
        + quality_penalty * (centered_sum + slack_value - centered_rhs) ** 2
    )


def candidate_id(cap: int) -> str:
    return f"tight_cap{cap}_centered_pair_upper"


def report_text(result: dict[str, Any]) -> str:
    selected = result["selected_encoding"]
    direct = result["direct_qpu_gate"]
    return rf"""# Stage70 constraint-aware QUBO encoding

## Question

Can the frozen Stage68 quality-floor objective be represented with a materially smaller coefficient range without changing the Stage69 scale-511 feasible problem?

## Encoding

For a fixed subset size $k$, Stage70 uses

$$
E(x,s)=R(x)+P_k\left(\sum_i x_i-k\right)^2
+P_q\left[\sum_i(d_i-c)x_i+\sum_jw_js_j-(D-kc)\right]^2.
$$

The slack range is tightened to $S_{{\max}}=D-\sum_{{r=1}}^k d_{{(r)}}$. Both penalties are set to the known feasible pair-off redundancy upper bound plus one. The integer center $c$ is selected only by the expanded QUBO coefficient range; no holdout metric or selected subset enters this choice.

## Result

- Selected encoding: `{selected.get('candidate_id', 'none')}`.
- Maximum coefficient dynamic range: `{selected.get('maximum_coefficient_dynamic_range', float('nan')):.6g}`.
- Improvement versus Stage69 scale 511: `{selected.get('dynamic_range_improvement_factor_vs_stage69', float('nan')):.3f}x`.
- Maximum logical variables: `{selected.get('maximum_logical_variable_count', 0)}`.
- Maximum quadratic coefficients: `{selected.get('maximum_quadratic_coefficient_count', 0)}`.
- Analytic exact-penalty certificates: `{selected.get('analytic_exact_penalty_certificate_count', 0)}/80`.
- Source scale-511 exact subset matches versus continuous Stage68: `{selected.get('exact_subset_match_count_vs_continuous', 0)}/80`.

## Decision boundary

- Compact logical QUBO freeze authorized: `{result['encoding_gate']['compact_logical_qubo_freeze_authorized']}`.
- Coefficient-noise simulation authorized: `{result['decision']['coefficient_noise_simulation_authorized']}`.
- Direct-QPU precision gate: `{direct['direct_qpu_precision_gate_passed']}`.
- Direct-QPU execution authorized: `{result['decision']['direct_qpu_execution_authorized']}`.

This is a post-hoc encoding result on four consumed development targets. It does not establish new-target efficacy, hardware sampling quality, speedup, or quantum advantage.
"""


def compute(config: dict[str, Any], root: Path) -> dict[str, Any]:
    implementation_paths = {
        key: verified(root, value, key)
        for key, value in config["implementation"].items()
    }
    input_paths = {
        key: verified(root, value, key) for key, value in config["inputs"].items()
    }
    stage64_config = read_json(input_paths["stage64_config"])
    stage68_result = read_json(input_paths["stage68_result"])
    stage69_result = read_json(input_paths["stage69_result"])
    stage69_audit = read_json(input_paths["stage69_audit"])
    if not stage68_result["route_gate"]["quality_plateau_qubo_freeze_authorized"]:
        raise ValueError("Stage70 requires the frozen Stage68 objective")
    if stage69_audit.get("status") != (
        "stage69_qubo_precision_compression_independent_audit_ok"
    ):
        raise ValueError("Stage70 requires the Stage69 independent audit")
    near_miss = stage69_result["best_uniform_near_miss"]
    encoding = config["encoding_screen"]
    scale = int(encoding["quality_integer_scale"])
    if int(near_miss["quality_integer_scale"]) != scale:
        raise ValueError("Stage70 quality scale differs from the Stage69 near miss")
    stage69_rows = read_csv(input_paths["stage69_cell_metrics"])
    references = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in stage69_rows
        if int(row["quality_integer_scale"]) == scale and row["status"] == "ok"
    }
    if len(references) != int(config["encoding_gate"]["required_cell_count"]):
        raise ValueError("Stage70 does not have all Stage69 scale-511 cells")
    development = config["development"]
    targets = [str(value) for value in development["target_order"]]
    subset_sizes = [int(value) for value in development["candidate_k_values"]]
    caps = [int(value) for value in encoding["slack_weight_caps"]]
    loaded = {
        target: load_target(root, target, stage64_config["targets"][target])
        for target in targets
    }
    rows: list[dict[str, Any]] = []
    internal_models: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for target_id in targets:
        target = loaded[target_id]
        ligand_ids = target["ligand_ids"]
        labels = target["labels"]
        receptor_ids = target["receptor_ids"]
        for outer_fold in range(int(development["outer_fold_count"])):
            train_mask = np.asarray(
                [target["outer"][ligand_id] != outer_fold for ligand_id in ligand_ids]
            )
            ranks = rank_cube(target["scores"], train_mask)
            train_rows = [
                row for row, keep in zip(target["ligands"], train_mask) if keep
            ]
            statistics_ = jackknife_pair_statistics(
                target["scores"][:, train_mask, :],
                labels[train_mask],
                train_rows,
                float(development["bedroc_alpha"]),
                int(development["jackknife_block_count"]),
                int(development["jackknife_seed_base"]) + outer_fold,
            )
            utility = statistics_["full_singleton"]
            spread = statistics_["singleton_spread"]
            redundancy = stable_redundancy(ranks, train_mask)
            for subset_size in subset_sizes:
                reference = references[(target_id, outer_fold, subset_size)]
                plateau = quality_plateau(
                    utility,
                    spread,
                    subset_size,
                    float(development["uncertainty_multiplier"]),
                )
                integerized = integerize_quality(
                    utility,
                    subset_size,
                    plateau["quality_floor"],
                    scale,
                )
                deficits = integerized["deficits"]
                maximum_deficit = int(integerized["maximum_deficit"])
                selected = selected_indices(reference["quantized_subset"], receptor_ids)
                if len(selected) != subset_size:
                    raise ValueError("Stage70 source subset size differs")
                if subset_name(plateau["baseline_subset"], receptor_ids) != reference[
                    "pair_off_subset"
                ]:
                    raise ValueError("Stage70 pair-off identity differs from Stage69")
                selected_deficit = int(np.sum(deficits[list(selected)]))
                if selected_deficit > maximum_deficit:
                    raise ValueError("Stage70 source subset violates integer quality")
                minimum_fixed_k_deficit = int(
                    np.sum(np.sort(deficits)[:subset_size])
                )
                tight_slack_maximum = maximum_deficit - minimum_fixed_k_deficit
                if tight_slack_maximum < 0:
                    raise ValueError("Stage70 tight slack bound is negative")
                selected_slack = maximum_deficit - selected_deficit
                pair_off_objective = redundancy_sum(
                    plateau["baseline_subset"], redundancy
                )
                selected_objective = redundancy_sum(selected, redundancy)
                if selected_objective > pair_off_objective + TOLERANCE:
                    raise ValueError("Stage70 source optimum exceeds its feasible bound")
                penalty = pair_off_objective + float(encoding["penalty_margin"])
                if penalty <= pair_off_objective:
                    raise ValueError("Stage70 exact penalty is not above its upper bound")
                for cap in caps:
                    current_id = candidate_id(cap)
                    slack_weights = bounded_slack_weights(tight_slack_maximum, cap)
                    if not slack_interval_is_complete(
                        slack_weights, tight_slack_maximum
                    ):
                        raise ValueError("Stage70 slack interval is incomplete")
                    center, qubo_summary = choose_center(
                        redundancy,
                        deficits,
                        maximum_deficit,
                        slack_weights,
                        subset_size,
                        penalty,
                        penalty,
                    )
                    energy = centered_energy(
                        selected,
                        selected_slack,
                        redundancy,
                        deficits,
                        maximum_deficit,
                        subset_size,
                        center,
                        penalty,
                        penalty,
                    )
                    residual = abs(energy - selected_objective)
                    invalid_gap = penalty - selected_objective
                    row = {
                        "candidate_id": current_id,
                        "slack_weight_cap": cap,
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "subset_size": subset_size,
                        "selected_subset": subset_name(selected, receptor_ids),
                        "pair_off_subset": subset_name(
                            plateau["baseline_subset"], receptor_ids
                        ),
                        "integer_center": center,
                        "center_search_candidate_count": int(np.max(deficits)) + 1,
                        "maximum_integer_deficit": maximum_deficit,
                        "minimum_fixed_k_deficit": minimum_fixed_k_deficit,
                        "tight_slack_maximum": tight_slack_maximum,
                        "selected_slack_value": selected_slack,
                        "slack_interval_complete": True,
                        "pair_off_redundancy_upper_bound": pair_off_objective,
                        "cardinality_penalty": penalty,
                        "quality_penalty": penalty,
                        "selected_redundancy_sum": selected_objective,
                        "selected_factorized_energy": energy,
                        "factorized_energy_residual": residual,
                        "analytic_invalid_state_gap_lower_bound": invalid_gap,
                        "analytic_exact_penalty_certificate": invalid_gap
                        >= float(encoding["penalty_margin"]) - TOLERANCE,
                        "source_exact_subset_match_vs_continuous": reference[
                            "exact_subset_match"
                        ],
                        "source_subset_jaccard_vs_continuous": reference[
                            "subset_jaccard_vs_continuous"
                        ],
                        "source_absolute_holdout_bedroc_gap": abs(
                            float(
                                reference[
                                    "quantized_minus_continuous_holdout_bedroc"
                                ]
                            )
                        ),
                        "source_actual_quality_floor_margin": reference[
                            "actual_quality_floor_margin"
                        ],
                        "stage69_coefficient_dynamic_range": reference[
                            "coefficient_dynamic_range"
                        ],
                        "cell_dynamic_range_improvement_factor": float(
                            reference["coefficient_dynamic_range"]
                        )
                        / float(qubo_summary["coefficient_dynamic_range"]),
                        **qubo_summary,
                    }
                    rows.append(row)
                    internal_models[
                        (current_id, target_id, outer_fold, subset_size)
                    ] = {
                        "row": row,
                        "receptor_ids": receptor_ids,
                        "deficits": deficits,
                        "slack_weights": slack_weights,
                        "redundancy": redundancy,
                    }
            print(json.dumps({"target_id": target_id, "outer_fold": outer_fold}))
    gates = config["encoding_gate"]
    summaries: list[dict[str, Any]] = []
    stage69_maximum = float(near_miss["maximum_coefficient_dynamic_range"])
    for cap in caps:
        current_id = candidate_id(cap)
        current = [row for row in rows if row["candidate_id"] == current_id]
        summary = {
            "candidate_id": current_id,
            "slack_weight_cap": cap,
            "cell_count": len(current),
            "analytic_exact_penalty_certificate_count": sum(
                bool(row["analytic_exact_penalty_certificate"]) for row in current
            ),
            "exact_subset_match_count_vs_continuous": sum(
                str(row["source_exact_subset_match_vs_continuous"]).lower()
                == "true"
                for row in current
            ),
            "mean_subset_jaccard_vs_continuous": statistics.fmean(
                float(row["source_subset_jaccard_vs_continuous"])
                for row in current
            ),
            "minimum_subset_jaccard_vs_continuous": min(
                float(row["source_subset_jaccard_vs_continuous"])
                for row in current
            ),
            "mean_absolute_holdout_bedroc_gap": statistics.fmean(
                float(row["source_absolute_holdout_bedroc_gap"])
                for row in current
            ),
            "maximum_absolute_holdout_bedroc_gap": max(
                float(row["source_absolute_holdout_bedroc_gap"])
                for row in current
            ),
            "minimum_actual_quality_floor_margin": min(
                float(row["source_actual_quality_floor_margin"])
                for row in current
            ),
            "minimum_analytic_invalid_state_gap_lower_bound": min(
                float(row["analytic_invalid_state_gap_lower_bound"])
                for row in current
            ),
            "maximum_factorized_energy_residual": max(
                float(row["factorized_energy_residual"]) for row in current
            ),
            "maximum_logical_variable_count": max(
                int(row["logical_variable_count"]) for row in current
            ),
            "maximum_quadratic_coefficient_count": max(
                int(row["quadratic_coefficient_count"]) for row in current
            ),
            "maximum_slack_variable_count": max(
                int(row["slack_variable_count"]) for row in current
            ),
            "maximum_coefficient_dynamic_range": max(
                float(row["coefficient_dynamic_range"]) for row in current
            ),
            "maximum_absolute_coefficient": max(
                float(row["maximum_absolute_coefficient"]) for row in current
            ),
        }
        summary["dynamic_range_improvement_factor_vs_stage69"] = (
            stage69_maximum
            / float(summary["maximum_coefficient_dynamic_range"])
        )
        summary["encoding_gate_passed"] = bool(
            int(summary["cell_count"]) >= int(gates["required_cell_count"])
            and int(summary["analytic_exact_penalty_certificate_count"])
            >= int(gates["required_exact_penalty_certificate_count"])
            and int(summary["exact_subset_match_count_vs_continuous"])
            >= int(gates["minimum_exact_subset_match_count_vs_continuous"])
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
            and float(summary["minimum_analytic_invalid_state_gap_lower_bound"])
            >= float(gates["minimum_analytic_invalid_state_gap"])
            and float(summary["maximum_factorized_energy_residual"])
            <= float(gates["maximum_factorized_energy_residual"])
            and int(summary["maximum_logical_variable_count"])
            <= int(gates["maximum_logical_variable_count"])
            and int(summary["maximum_quadratic_coefficient_count"])
            <= int(gates["maximum_quadratic_coefficient_count"])
            and float(summary["maximum_coefficient_dynamic_range"])
            <= float(gates["maximum_coefficient_dynamic_range"])
            and float(summary["dynamic_range_improvement_factor_vs_stage69"])
            >= float(gates["minimum_dynamic_range_improvement_factor"])
        )
        summaries.append(summary)
    eligible = [row for row in summaries if row["encoding_gate_passed"]]
    selected_summary = (
        min(
            eligible,
            key=lambda row: (
                float(row["maximum_coefficient_dynamic_range"]),
                int(row["maximum_logical_variable_count"]),
                int(row["maximum_quadratic_coefficient_count"]),
                int(row["slack_weight_cap"]),
            ),
        )
        if eligible
        else {}
    )
    selected_id = str(selected_summary.get("candidate_id", ""))
    reference_k = int(encoding["reference_model_k"])
    model_records: list[dict[str, Any]] = []
    if selected_id:
        for target_id in targets:
            for outer_fold in range(int(development["outer_fold_count"])):
                model = internal_models[
                    (selected_id, target_id, outer_fold, reference_k)
                ]
                row = model["row"]
                redundancy = model["redundancy"]
                model_records.append(
                    {
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "reference_k": reference_k,
                        "candidate_id": selected_id,
                        "receptor_ids": model["receptor_ids"],
                        "selected_subset": row["selected_subset"],
                        "integer_deficits": [
                            int(value) for value in model["deficits"]
                        ],
                        "maximum_integer_deficit": row[
                            "maximum_integer_deficit"
                        ],
                        "minimum_fixed_k_deficit": row[
                            "minimum_fixed_k_deficit"
                        ],
                        "tight_slack_maximum": row["tight_slack_maximum"],
                        "slack_weights": model["slack_weights"],
                        "integer_center": row["integer_center"],
                        "centered_rhs": row["centered_rhs"],
                        "cardinality_penalty": row["cardinality_penalty"],
                        "quality_penalty": row["quality_penalty"],
                        "pair_off_redundancy_upper_bound": row[
                            "pair_off_redundancy_upper_bound"
                        ],
                        "selected_slack_value": row["selected_slack_value"],
                        "selected_redundancy_sum": row[
                            "selected_redundancy_sum"
                        ],
                        "selected_factorized_energy": row[
                            "selected_factorized_energy"
                        ],
                        "factorized_energy_residual": row[
                            "factorized_energy_residual"
                        ],
                        "analytic_invalid_state_gap_lower_bound": row[
                            "analytic_invalid_state_gap_lower_bound"
                        ],
                        "stable_redundancy_upper_triangle": [
                            float(redundancy[left, right])
                            for left, right in itertools.combinations(
                                range(len(model["receptor_ids"])), 2
                            )
                        ],
                        "qubo_summary": {
                            key: row[key]
                            for key in (
                                "logical_variable_count",
                                "receptor_variable_count",
                                "slack_variable_count",
                                "linear_coefficient_count",
                                "quadratic_coefficient_count",
                                "constant",
                                "minimum_absolute_nonzero_coefficient",
                                "maximum_absolute_coefficient",
                                "coefficient_dynamic_range",
                                "qubo_sha256",
                            )
                        },
                    }
                )
    direct_gate = {
        "maximum_permitted_dynamic_range": float(
            config["direct_qpu_gate"]["maximum_coefficient_dynamic_range"]
        ),
        "observed_maximum_dynamic_range": float(
            selected_summary.get("maximum_coefficient_dynamic_range", 0.0)
        ),
        "direct_qpu_precision_gate_passed": bool(selected_summary)
        and float(selected_summary["maximum_coefficient_dynamic_range"])
        <= float(config["direct_qpu_gate"]["maximum_coefficient_dynamic_range"]),
    }
    output_paths = {
        key: root / str(value) for key, value in config["outputs"].items()
    }
    write_csv(output_paths["cell_metrics_csv"], rows)
    write_csv(output_paths["candidate_summary_csv"], summaries)
    model_record = {
        "schema_version": "1.0",
        "algorithm_id": config["encoding_screen"]["algorithm_id"],
        "selected_candidate_id": selected_id,
        "reference_k": reference_k,
        "model_count": len(model_records),
        "models": model_records,
    }
    write_json(output_paths["model_record_json"], model_record)
    payload = {
        "candidate_summaries": summaries,
        "selected_encoding": selected_summary,
        "direct_qpu_gate": direct_gate,
    }
    freeze_authorized = bool(selected_summary)
    result = {
        "schema_version": "1.0",
        "status": "stage70_constraint_aware_qubo_encoding_complete",
        "experiment_class": "post-hoc exact-penalty encoding development on frozen historical models",
        "config": descriptor(root, root / "configs/stage70_constraint_aware_qubo_encoding.json"),
        "implementation": {
            key: descriptor(root, path) for key, path in implementation_paths.items()
        },
        "inputs": {key: descriptor(root, path) for key, path in input_paths.items()},
        "candidate_count": len(caps),
        "cell_metric_count": len(rows),
        "selected_encoding": selected_summary,
        "encoding_gate": {
            "compact_logical_qubo_freeze_authorized": freeze_authorized
        },
        "direct_qpu_gate": direct_gate,
        "decision": {
            "coefficient_noise_simulation_authorized": freeze_authorized,
            "direct_qpu_execution_authorized": freeze_authorized
            and direct_gate["direct_qpu_precision_gate_passed"],
            "new_target_preregistration_remains_authorized": stage69_result[
                "decision"
            ]["new_target_preregistration_remains_authorized"],
            "quantum_advantage_claim_authorized": False,
            "next_action": (
                "freeze this encoding for coefficient-noise simulation while preserving the new-target preregistration route"
                if freeze_authorized
                else "retain Stage69 and redesign the quality constraint outside a direct BQM"
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
        "candidate_summary_csv": descriptor(
            root, output_paths["candidate_summary_csv"]
        ),
        "model_record_json": descriptor(root, output_paths["model_record_json"]),
    }
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
    expected = root / "configs/stage70_constraint_aware_qubo_encoding.json"
    if config_path != expected.resolve():
        raise ValueError("Stage70 must run from its frozen repository config")
    config = read_json(config_path)
    result_path = root / str(config["outputs"]["result_json"])
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"Stage70 result exists: {result_path}")
    result = compute(config, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage70_constraint_aware_qubo_encoding.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
