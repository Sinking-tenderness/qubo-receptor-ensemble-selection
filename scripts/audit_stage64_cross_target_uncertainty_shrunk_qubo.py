"""Independently audit Stage64 uncertainty-shrunk QUBO outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
from pathlib import Path
from typing import Any


TOLERANCE = 1e-12
K_VALUES = tuple(range(1, 7))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def checked_hash(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage64 identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage64 size differs: {path}")
    return path


def close(left: Any, right: Any) -> bool:
    return abs(float(left) - float(right)) <= TOLERANCE


def subset_set(value: str) -> set[str]:
    result = {item for item in value.split("+") if item}
    if not result:
        raise ValueError("Stage64 selected subset is empty")
    return result


def pairwise_jaccard(values: list[str]) -> float:
    sets = [subset_set(value) for value in values]
    pairs = list(itertools.combinations(sets, 2))
    return (
        statistics.fmean(len(left & right) / len(left | right) for left, right in pairs)
        if pairs
        else 1.0
    )


def compare_fields(
    observed: dict[str, Any], expected: dict[str, Any], float_fields: set[str]
) -> None:
    for key, value in expected.items():
        if key in float_fields:
            if not close(observed[key], value):
                raise ValueError(f"Stage64 numeric summary differs: {key}")
        elif str(observed[key]) != str(value):
            raise ValueError(f"Stage64 summary differs: {key}")


def recompute_target_summaries(
    metrics: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    target_order: list[str],
) -> list[dict[str, Any]]:
    cells = {
        (
            row["target_id"],
            int(row["outer_fold"]),
            row["candidate_id"],
            int(row["subset_size"]),
        ): row
        for row in metrics
    }
    output: list[dict[str, Any]] = []
    for target_id in target_order:
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            rows = [
                cells[(target_id, fold, candidate_id, subset_size)]
                for fold in range(4)
                for subset_size in range(2, 7)
            ]
            values = [float(row["holdout_robust_bedroc"]) for row in rows]
            baseline_gains = [
                value
                - float(
                    cells[
                        (
                            target_id,
                            int(row["outer_fold"]),
                            "baseline_v1",
                            int(row["subset_size"]),
                        )
                    ]["holdout_robust_bedroc"]
                )
                for row, value in zip(rows, values)
            ]
            pair_off_gains = [
                value
                - float(
                    cells[
                        (
                            target_id,
                            int(row["outer_fold"]),
                            "pair_off",
                            int(row["subset_size"]),
                        )
                    ]["holdout_robust_bedroc"]
                )
                for row, value in zip(rows, values)
            ]
            stability = statistics.fmean(
                pairwise_jaccard(
                    [
                        cells[(target_id, fold, candidate_id, subset_size)][
                            "selected_subset"
                        ]
                        for fold in range(4)
                    ]
                )
                for subset_size in range(2, 7)
            )
            output.append(
                {
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "fixed_k_cell_count": 20,
                    "mean_fixed_k_holdout_robust_bedroc": statistics.fmean(values),
                    "mean_gain_over_baseline_v1": statistics.fmean(baseline_gains),
                    "minimum_fold_k_gain_over_baseline_v1": min(baseline_gains),
                    "positive_fold_k_gain_count": sum(
                        value > TOLERANCE for value in baseline_gains
                    ),
                    "mean_gain_over_pair_off": statistics.fmean(pair_off_gains),
                    "minimum_fold_k_gain_over_pair_off": min(pair_off_gains),
                    "positive_fold_k_gain_over_pair_off_count": sum(
                        value > TOLERANCE for value in pair_off_gains
                    ),
                    "mean_fixed_k_selection_jaccard": stability,
                }
            )
    return output


def recompute_global_summaries(
    target_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    target_order: list[str],
) -> list[dict[str, Any]]:
    lookup = {
        (row["target_id"], row["candidate_id"]): row for row in target_rows
    }
    output: list[dict[str, Any]] = []
    for order, candidate in enumerate(candidates):
        rows = [
            lookup[(target_id, candidate["candidate_id"])]
            for target_id in target_order
        ]
        baseline_gains = [float(row["mean_gain_over_baseline_v1"]) for row in rows]
        pair_off_gains = [float(row["mean_gain_over_pair_off"]) for row in rows]
        output.append(
            {
                "candidate_order": order,
                "candidate_id": candidate["candidate_id"],
                "mode": candidate["mode"],
                "eligible_for_freeze": bool(candidate["eligible_for_freeze"]),
                "lambda_mad": candidate["lambda_mad"],
                "sign_support_threshold": candidate["sign_support_threshold"],
                "pair_scale": candidate["pair_scale"],
                "mean_target_gain_over_baseline_v1": statistics.fmean(baseline_gains),
                "worst_target_gain_over_baseline_v1": min(baseline_gains),
                "nonnegative_target_count": sum(
                    value >= -TOLERANCE for value in baseline_gains
                ),
                "positive_target_count": sum(
                    value > TOLERANCE for value in baseline_gains
                ),
                "mean_target_gain_over_pair_off": statistics.fmean(pair_off_gains),
                "worst_target_gain_over_pair_off": min(pair_off_gains),
                "nonnegative_target_count_over_pair_off": sum(
                    value >= -TOLERANCE for value in pair_off_gains
                ),
                "positive_target_count_over_pair_off": sum(
                    value > TOLERANCE for value in pair_off_gains
                ),
                "mean_target_selection_jaccard": statistics.fmean(
                    float(row["mean_fixed_k_selection_jaccard"]) for row in rows
                ),
            }
        )
    return output


def recompute_loto(
    target_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    target_order: list[str],
) -> list[dict[str, Any]]:
    lookup = {
        (row["target_id"], row["candidate_id"]): row for row in target_rows
    }
    output: list[dict[str, Any]] = []
    for held_target in target_order:
        development_targets = [value for value in target_order if value != held_target]
        choices: list[dict[str, Any]] = []
        for order, candidate in enumerate(candidates):
            if not bool(candidate["eligible_for_freeze"]):
                continue
            rows = [
                lookup[(target_id, candidate["candidate_id"])]
                for target_id in development_targets
            ]
            choices.append(
                {
                    "candidate_order": order,
                    "candidate_id": candidate["candidate_id"],
                    "development_mean_gain_over_pair_off": statistics.fmean(
                        float(row["mean_gain_over_pair_off"]) for row in rows
                    ),
                    "development_mean_fixed_k_bedroc": statistics.fmean(
                        float(row["mean_fixed_k_holdout_robust_bedroc"])
                        for row in rows
                    ),
                }
            )
        selected = min(
            choices,
            key=lambda row: (
                -float(row["development_mean_gain_over_pair_off"]),
                -float(row["development_mean_fixed_k_bedroc"]),
                int(row["candidate_order"]),
            ),
        )
        held = lookup[(held_target, selected["candidate_id"])]
        baseline = lookup[(held_target, "baseline_v1")]
        pair_off = lookup[(held_target, "pair_off")]
        held_value = float(held["mean_fixed_k_holdout_robust_bedroc"])
        output.append(
            {
                "held_target_id": held_target,
                "selected_candidate_id": selected["candidate_id"],
                "development_target_ids": "+".join(development_targets),
                "development_mean_gain_over_pair_off": selected[
                    "development_mean_gain_over_pair_off"
                ],
                "development_mean_fixed_k_bedroc": selected[
                    "development_mean_fixed_k_bedroc"
                ],
                "held_target_mean_fixed_k_bedroc": held_value,
                "held_target_baseline_mean_fixed_k_bedroc": baseline[
                    "mean_fixed_k_holdout_robust_bedroc"
                ],
                "held_target_gain_over_baseline_v1": held_value
                - float(baseline["mean_fixed_k_holdout_robust_bedroc"]),
                "held_target_pair_off_mean_fixed_k_bedroc": pair_off[
                    "mean_fixed_k_holdout_robust_bedroc"
                ],
                "held_target_gain_over_pair_off": held_value
                - float(pair_off["mean_fixed_k_holdout_robust_bedroc"]),
            }
        )
    return output


def run(
    config_path: Path,
    result_path: Path,
    root: Path,
    output_path: Path,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    result_path = result_path.resolve()
    config = read_json(config_path)
    result = read_json(result_path)
    if result.get("status") != "stage64_cross_target_uncertainty_shrunk_qubo_complete":
        raise ValueError("Stage64 source analysis did not complete")
    if checked_hash(root, result["config"]).resolve() != config_path:
        raise ValueError("Stage64 result config differs")
    auditor = config["implementation"]["independent_auditor"]
    if checked_hash(root, auditor).resolve() != Path(__file__).resolve():
        raise ValueError("Stage64 auditor identity differs")
    for value in config["inputs"].values():
        checked_hash(root, value)
    for target in config["targets"].values():
        for value in target["inputs"].values():
            checked_hash(root, value)
    outputs = {key: checked_hash(root, value) for key, value in result["outputs"].items()}

    metrics = read_csv(outputs["fixed_k_metrics_csv"])
    pair_diagnostics = read_csv(outputs["pair_diagnostics_csv"])
    target_summary = read_csv(outputs["target_summary_csv"])
    global_summary = read_csv(outputs["global_summary_csv"])
    loto_summary = read_csv(outputs["loto_summary_csv"])
    model_record = read_json(outputs["model_record_json"])
    candidates = [dict(value) for value in config["candidate_grid"]]
    target_order = [str(value) for value in config["development"]["target_order"]]
    expected_metric_count = len(target_order) * 4 * len(candidates) * len(K_VALUES)
    expected_pair_count = len(target_order) * 4 * len(candidates)
    if len(metrics) != expected_metric_count or len(pair_diagnostics) != expected_pair_count:
        raise ValueError("Stage64 primary output dimensions differ")
    if len(target_summary) != len(target_order) * len(candidates):
        raise ValueError("Stage64 target summary dimension differs")
    if len(global_summary) != len(candidates) or len(loto_summary) != len(target_order):
        raise ValueError("Stage64 global or LOTO dimension differs")
    metric_keys = {
        (
            row["target_id"],
            int(row["outer_fold"]),
            row["candidate_id"],
            int(row["subset_size"]),
        )
        for row in metrics
    }
    expected_keys = {
        (target_id, fold, candidate["candidate_id"], subset_size)
        for target_id in target_order
        for fold in range(4)
        for candidate in candidates
        for subset_size in K_VALUES
    }
    if metric_keys != expected_keys or len(metric_keys) != len(metrics):
        raise ValueError("Stage64 fixed-k grid differs")

    stage63 = read_csv(checked_hash(root, config["inputs"]["stage63_fixed_k_landscape"]))
    baseline = {
        (
            row["target_id"],
            int(row["outer_fold"]),
            int(row["subset_size"]),
        ): row
        for row in metrics
        if row["candidate_id"] == "baseline_v1"
    }
    source = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in stage63
    }
    if set(baseline) != set(source):
        raise ValueError("Stage64 baseline reconstruction grid differs")
    for key, row in baseline.items():
        if row["selected_subset"] != source[key]["selected_subset"]:
            raise ValueError("Stage64 baseline reconstruction subset differs")
        if not close(row["train_qubo_objective"], source[key]["train_qubo_objective"]):
            raise ValueError("Stage64 baseline reconstruction objective differs")
        if not close(row["holdout_robust_bedroc"], source[key]["holdout_robust_bedroc"]):
            raise ValueError("Stage64 baseline reconstruction BEDROC differs")

    recomputed_targets = recompute_target_summaries(metrics, candidates, target_order)
    observed_targets = {
        (row["target_id"], row["candidate_id"]): row for row in target_summary
    }
    target_float_fields = {
        "mean_fixed_k_holdout_robust_bedroc",
        "mean_gain_over_baseline_v1",
        "minimum_fold_k_gain_over_baseline_v1",
        "mean_gain_over_pair_off",
        "minimum_fold_k_gain_over_pair_off",
        "mean_fixed_k_selection_jaccard",
    }
    for row in recomputed_targets:
        compare_fields(
            observed_targets[(row["target_id"], row["candidate_id"])],
            row,
            target_float_fields,
        )

    recomputed_global = recompute_global_summaries(
        recomputed_targets, candidates, target_order
    )
    observed_global = {row["candidate_id"]: row for row in global_summary}
    global_float_fields = {
        "lambda_mad",
        "sign_support_threshold",
        "pair_scale",
        "mean_target_gain_over_baseline_v1",
        "worst_target_gain_over_baseline_v1",
        "mean_target_gain_over_pair_off",
        "worst_target_gain_over_pair_off",
        "mean_target_selection_jaccard",
    }
    for row in recomputed_global:
        observed = observed_global[row["candidate_id"]]
        expected = dict(row)
        expected["eligible_for_freeze"] = str(row["eligible_for_freeze"])
        compare_fields(observed, expected, global_float_fields)
    eligible = [row for row in recomputed_global if row["eligible_for_freeze"]]
    selected = min(
        eligible,
        key=lambda row: (
            -float(row["worst_target_gain_over_pair_off"]),
            -float(row["mean_target_gain_over_pair_off"]),
            -float(row["worst_target_gain_over_baseline_v1"]),
            -float(row["mean_target_gain_over_baseline_v1"]),
            int(row["candidate_order"]),
        ),
    )
    compare_fields(result["selected_candidate"], selected, global_float_fields)

    recomputed_loto = recompute_loto(recomputed_targets, candidates, target_order)
    observed_loto = {row["held_target_id"]: row for row in loto_summary}
    loto_float_fields = {
        "development_mean_gain_over_pair_off",
        "development_mean_fixed_k_bedroc",
        "held_target_mean_fixed_k_bedroc",
        "held_target_baseline_mean_fixed_k_bedroc",
        "held_target_gain_over_baseline_v1",
        "held_target_pair_off_mean_fixed_k_bedroc",
        "held_target_gain_over_pair_off",
    }
    for row in recomputed_loto:
        compare_fields(observed_loto[row["held_target_id"]], row, loto_float_fields)

    thresholds = config["freeze_gate"]
    loto_baseline_gains = [float(row["held_target_gain_over_baseline_v1"]) for row in recomputed_loto]
    loto_pair_off_gains = [float(row["held_target_gain_over_pair_off"]) for row in recomputed_loto]
    checks = {
        "nonbaseline_candidate_selected": selected["candidate_id"] != "baseline_v1",
        "minimum_mean_target_gain": float(selected["mean_target_gain_over_baseline_v1"])
        >= float(thresholds["minimum_mean_target_gain"]) - TOLERANCE,
        "minimum_worst_target_gain": float(selected["worst_target_gain_over_baseline_v1"])
        >= float(thresholds["minimum_worst_target_gain"]) - TOLERANCE,
        "minimum_nonnegative_target_count": int(selected["nonnegative_target_count"])
        >= int(thresholds["minimum_nonnegative_target_count"]),
        "minimum_mean_target_gain_over_pair_off": float(selected["mean_target_gain_over_pair_off"])
        >= float(thresholds["minimum_mean_target_gain_over_pair_off"]) - TOLERANCE,
        "minimum_worst_target_gain_over_pair_off": float(selected["worst_target_gain_over_pair_off"])
        >= float(thresholds["minimum_worst_target_gain_over_pair_off"]) - TOLERANCE,
        "minimum_nonnegative_target_count_over_pair_off": int(
            selected["nonnegative_target_count_over_pair_off"]
        )
        >= int(thresholds["minimum_nonnegative_target_count_over_pair_off"]),
        "nonnegative_loto_mean_gain": statistics.fmean(loto_baseline_gains)
        >= float(thresholds["minimum_loto_mean_gain"]) - TOLERANCE,
        "minimum_positive_loto_target_count": sum(
            value > TOLERANCE for value in loto_baseline_gains
        )
        >= int(thresholds["minimum_positive_loto_target_count"]),
        "nonnegative_loto_mean_gain_over_pair_off": statistics.fmean(loto_pair_off_gains)
        >= float(thresholds["minimum_loto_mean_gain_over_pair_off"]) - TOLERANCE,
        "minimum_positive_loto_target_count_over_pair_off": sum(
            value > TOLERANCE for value in loto_pair_off_gains
        )
        >= int(thresholds["minimum_positive_loto_target_count_over_pair_off"]),
    }
    if result["freeze_gate"]["checks"] != checks:
        raise ValueError("Stage64 freeze checks differ")
    frozen = all(checks.values())
    if bool(result["freeze_gate"]["objective_v2_frozen"]) != frozen:
        raise ValueError("Stage64 freeze decision differs")
    if model_record["selected_candidate"]["candidate_id"] != selected["candidate_id"]:
        raise ValueError("Stage64 model record selection differs")
    expected_model_status = (
        "stage64_uncertainty_shrunk_rank_pair_qubo_v2_frozen"
        if frozen
        else "stage64_uncertainty_shrunk_rank_pair_qubo_v2_not_frozen"
    )
    if model_record["status"] != expected_model_status:
        raise ValueError("Stage64 model record status differs")
    boundary = result["data_boundary"]
    if any(
        int(boundary[key]) != 0
        for key in (
            "fresh_validation_rows_read",
            "locked_test_rows_read",
            "new_docking_jobs",
            "quantum_hardware_jobs",
        )
    ):
        raise ValueError("Stage64 crossed a protected data or execution boundary")
    if result["decision"]["same_target_retuning_authorized"] is not False:
        raise ValueError("Stage64 improperly authorizes same-target retuning")
    if result["decision"]["fresh_validation_authorized"] is not False:
        raise ValueError("Stage64 improperly authorizes fresh validation")
    if result["decision"]["quantum_hardware_authorized"] is not False:
        raise ValueError("Stage64 improperly authorizes hardware")

    audit = {
        "schema_version": "1.0",
        "status": "stage64_cross_target_uncertainty_shrunk_qubo_independent_audit_ok",
        "source_result": {
            "path": result_path.relative_to(root).as_posix(),
            "sha256": sha256(result_path),
            "size_bytes": result_path.stat().st_size,
        },
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": sha256(config_path),
            "size_bytes": config_path.stat().st_size,
        },
        "row_counts": {
            "fixed_k_metrics": len(metrics),
            "pair_diagnostics": len(pair_diagnostics),
            "target_summary": len(target_summary),
            "global_summary": len(global_summary),
            "loto_summary": len(loto_summary),
        },
        "baseline_reproduction_cells_independently_verified": len(baseline),
        "candidate_summaries_independently_recomputed": True,
        "loto_selection_independently_recomputed": True,
        "freeze_gate_independently_recomputed": True,
        "all_output_hashes_exact": True,
        "data_boundary_exact": True,
        "selected_candidate_id": selected["candidate_id"],
        "objective_v2_frozen": frozen,
        "interpretation_boundary": result["interpretation_boundary"],
    }
    output_path = output_path if output_path.is_absolute() else root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
