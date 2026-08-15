"""Independently audit Stage65 pair-sign mechanism outputs."""

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


TOLERANCE = 1e-12


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


def checked(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage65 identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage65 size differs: {path}")
    return path


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value) == "True":
        return True
    if str(value) == "False":
        return False
    raise ValueError(f"invalid Stage65 Boolean: {value}")


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    output = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[start][1]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for index in range(start, stop):
            output[ordered[index][0]] = rank
        start = stop
    return output


def spearman(left: list[float], right: list[float]) -> float:
    x = average_ranks(left)
    y = average_ranks(right)
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    scale_x = math.sqrt(sum((value - mean_x) ** 2 for value in x))
    scale_y = math.sqrt(sum((value - mean_y) ** 2 for value in y))
    if scale_x <= TOLERANCE or scale_y <= TOLERANCE:
        return 0.0
    return numerator / (scale_x * scale_y)


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def pairwise_jaccard(values: list[str]) -> float:
    sets = [{item for item in value.split("+") if item} for value in values]
    pairs = list(itertools.combinations(sets, 2))
    return (
        statistics.fmean(len(left & right) / len(left | right) for left, right in pairs)
        if pairs
        else 1.0
    )


def compare(
    observed: dict[str, Any], expected: dict[str, Any], float_fields: set[str]
) -> None:
    for key, value in expected.items():
        if key in float_fields:
            if abs(float(observed[key]) - float(value)) > TOLERANCE:
                raise ValueError(f"Stage65 numeric summary differs: {key}")
        elif str(observed[key]) != str(value):
            raise ValueError(f"Stage65 summary differs: {key}")


def edge_summary(
    rows: list[dict[str, str]], target_id: str, outer_fold: int | None
) -> dict[str, Any]:
    selected = [row for row in rows if row["target_id"] == target_id]
    if outer_fold is not None:
        selected = [row for row in selected if int(row["outer_fold"]) == outer_fold]
    train_positive = [row for row in selected if as_bool(row["train_positive"])]
    train_negative = [row for row in selected if as_bool(row["train_negative"])]
    stable_positive = [row for row in selected if as_bool(row["stable_positive"])]
    stable_negative = [row for row in selected if as_bool(row["stable_negative"])]
    lcb_positive = [row for row in selected if as_bool(row["lcb_positive"])]
    lcb_negative = [row for row in selected if as_bool(row["lcb_negative"])]
    return {
        "target_id": target_id,
        "outer_fold": "all" if outer_fold is None else outer_fold,
        "pair_count": len(selected),
        "train_holdout_pair_residual_spearman": spearman(
            [float(row["train_pair_residual"]) for row in selected],
            [float(row["holdout_pair_residual"]) for row in selected],
        ),
        "all_edge_holdout_positive_rate": rate(
            sum(as_bool(row["holdout_positive"]) for row in selected), len(selected)
        ),
        "train_positive_edge_count": len(train_positive),
        "train_positive_holdout_positive_rate": rate(
            sum(as_bool(row["holdout_positive"]) for row in train_positive),
            len(train_positive),
        ),
        "train_negative_edge_count": len(train_negative),
        "train_negative_holdout_negative_rate": rate(
            sum(as_bool(row["holdout_negative"]) for row in train_negative),
            len(train_negative),
        ),
        "stable_positive_edge_count": len(stable_positive),
        "stable_positive_holdout_positive_rate": rate(
            sum(as_bool(row["holdout_positive"]) for row in stable_positive),
            len(stable_positive),
        ),
        "stable_negative_edge_count": len(stable_negative),
        "stable_negative_holdout_negative_rate": rate(
            sum(as_bool(row["holdout_negative"]) for row in stable_negative),
            len(stable_negative),
        ),
        "lcb_positive_edge_count": len(lcb_positive),
        "lcb_positive_holdout_positive_rate": rate(
            sum(as_bool(row["holdout_positive"]) for row in lcb_positive),
            len(lcb_positive),
        ),
        "lcb_negative_edge_count": len(lcb_negative),
        "lcb_negative_holdout_negative_rate": rate(
            sum(as_bool(row["holdout_negative"]) for row in lcb_negative),
            len(lcb_negative),
        ),
    }


def candidate_summaries(
    metrics: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    target_order: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_off = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): float(
            row["holdout_robust_bedroc"]
        )
        for row in metrics
        if row["candidate_id"] == "pair_off"
    }
    target_rows: list[dict[str, Any]] = []
    for target_id in target_order:
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            rows = [
                row
                for row in metrics
                if row["target_id"] == target_id
                and row["candidate_id"] == candidate_id
                and int(row["subset_size"]) >= 2
            ]
            values = [float(row["holdout_robust_bedroc"]) for row in rows]
            gains = [
                value
                - pair_off[
                    (
                        target_id,
                        int(row["outer_fold"]),
                        int(row["subset_size"]),
                    )
                ]
                for row, value in zip(rows, values)
            ]
            target_rows.append(
                {
                    "target_id": target_id,
                    "candidate_id": candidate_id,
                    "fixed_k_cell_count": len(rows),
                    "mean_fixed_k_holdout_robust_bedroc": statistics.fmean(values),
                    "mean_gain_over_pair_off": statistics.fmean(gains),
                    "minimum_fold_k_gain_over_pair_off": min(gains),
                    "nonnegative_fold_k_gain_over_pair_off_count": sum(
                        value >= -TOLERANCE for value in gains
                    ),
                    "positive_fold_k_gain_over_pair_off_count": sum(
                        value > TOLERANCE for value in gains
                    ),
                    "mean_fixed_k_selection_jaccard": statistics.fmean(
                        pairwise_jaccard(
                            [
                                row["selected_subset"]
                                for row in rows
                                if int(row["subset_size"]) == subset_size
                            ]
                        )
                        for subset_size in range(2, 7)
                    ),
                }
            )
    lookup = {
        (row["target_id"], row["candidate_id"]): row for row in target_rows
    }
    global_rows: list[dict[str, Any]] = []
    for order, candidate in enumerate(candidates):
        rows = [
            lookup[(target_id, candidate["candidate_id"])]
            for target_id in target_order
        ]
        gains = [float(row["mean_gain_over_pair_off"]) for row in rows]
        global_rows.append(
            {
                "candidate_order": order,
                "candidate_id": candidate["candidate_id"],
                "mode": candidate["mode"],
                "pair_scale": candidate["pair_scale"],
                "sign_support_threshold": candidate["sign_support_threshold"],
                "lambda_mad": candidate["lambda_mad"],
                "mean_target_gain_over_pair_off": statistics.fmean(gains),
                "worst_target_gain_over_pair_off": min(gains),
                "nonnegative_target_count_over_pair_off": sum(
                    value >= -TOLERANCE for value in gains
                ),
                "positive_target_count_over_pair_off": sum(
                    value > TOLERANCE for value in gains
                ),
                "mean_target_selection_jaccard": statistics.fmean(
                    float(row["mean_fixed_k_selection_jaccard"]) for row in rows
                ),
            }
        )
    return target_rows, global_rows


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
    if result.get("status") != "stage65_cross_target_pair_sign_mechanism_complete":
        raise ValueError("Stage65 source result did not complete")
    if checked(root, result["config"]).resolve() != config_path:
        raise ValueError("Stage65 result config differs")
    auditor = config["implementation"]["independent_auditor"]
    if checked(root, auditor).resolve() != Path(__file__).resolve():
        raise ValueError("Stage65 auditor identity differs")
    for value in config["implementation"].values():
        checked(root, value)
    inputs = {key: checked(root, value) for key, value in config["inputs"].items()}
    outputs = {key: checked(root, value) for key, value in result["outputs"].items()}
    stage64_result = read_json(inputs["stage64_result"])

    edges = read_csv(outputs["edge_transfer_csv"])
    edge_folds = read_csv(outputs["edge_fold_summary_csv"])
    edge_targets = read_csv(outputs["edge_target_summary_csv"])
    metrics = read_csv(outputs["fixed_k_metrics_csv"])
    target_summary = read_csv(outputs["target_summary_csv"])
    global_summary = read_csv(outputs["global_summary_csv"])
    candidates = [dict(value) for value in config["candidate_grid"]]
    target_order = [str(value) for value in config["diagnosis"]["target_order"]]
    expected_edges = sum(
        4
        * int(result["target_input_audits"][target_id]["receptor_count"])
        * (int(result["target_input_audits"][target_id]["receptor_count"]) - 1)
        // 2
        for target_id in target_order
    )
    expected_metrics = len(target_order) * 4 * len(candidates) * 6
    if len(edges) != expected_edges or len(metrics) != expected_metrics:
        raise ValueError("Stage65 primary dimensions differ")
    if len(edge_folds) != 16 or len(edge_targets) != 4:
        raise ValueError("Stage65 edge summary dimensions differ")
    if len(target_summary) != len(target_order) * len(candidates):
        raise ValueError("Stage65 target summary dimension differs")
    if len(global_summary) != len(candidates):
        raise ValueError("Stage65 global summary dimension differs")

    stage64_metrics = read_csv(
        checked(root, stage64_result["outputs"]["fixed_k_metrics_csv"])
    )
    observed_pair_off = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in metrics
        if row["candidate_id"] == "pair_off"
    }
    expected_pair_off = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in stage64_metrics
        if row["candidate_id"] == "pair_off"
    }
    if set(observed_pair_off) != set(expected_pair_off):
        raise ValueError("Stage65 pair-off reproduction grid differs")
    for key, row in observed_pair_off.items():
        source = expected_pair_off[key]
        if row["selected_subset"] != source["selected_subset"]:
            raise ValueError("Stage65 pair-off subset differs")
        for field in ("train_qubo_objective", "holdout_robust_bedroc"):
            if abs(float(row[field]) - float(source[field])) > TOLERANCE:
                raise ValueError(f"Stage65 pair-off {field} differs")

    fold_lookup = {
        (row["target_id"], int(row["outer_fold"])): row for row in edge_folds
    }
    target_lookup = {row["target_id"]: row for row in edge_targets}
    edge_float_fields = {
        "train_holdout_pair_residual_spearman",
        "all_edge_holdout_positive_rate",
        "train_positive_holdout_positive_rate",
        "train_negative_holdout_negative_rate",
        "stable_positive_holdout_positive_rate",
        "stable_negative_holdout_negative_rate",
        "lcb_positive_holdout_positive_rate",
        "lcb_negative_holdout_negative_rate",
    }
    recomputed_folds = []
    for target_id in target_order:
        for fold in range(4):
            row = edge_summary(edges, target_id, fold)
            recomputed_folds.append(row)
            compare(fold_lookup[(target_id, fold)], row, edge_float_fields)
        row = edge_summary(edges, target_id, None)
        compare(target_lookup[target_id], row, edge_float_fields)

    recomputed_targets, recomputed_global = candidate_summaries(
        metrics, candidates, target_order
    )
    observed_targets = {
        (row["target_id"], row["candidate_id"]): row for row in target_summary
    }
    target_float_fields = {
        "mean_fixed_k_holdout_robust_bedroc",
        "mean_gain_over_pair_off",
        "minimum_fold_k_gain_over_pair_off",
        "mean_fixed_k_selection_jaccard",
    }
    for row in recomputed_targets:
        compare(
            observed_targets[(row["target_id"], row["candidate_id"])],
            row,
            target_float_fields,
        )
    observed_global = {row["candidate_id"]: row for row in global_summary}
    global_float_fields = {
        "pair_scale",
        "sign_support_threshold",
        "lambda_mad",
        "mean_target_gain_over_pair_off",
        "worst_target_gain_over_pair_off",
        "mean_target_selection_jaccard",
    }
    for row in recomputed_global:
        compare(observed_global[row["candidate_id"]], row, global_float_fields)

    primary_id = str(config["decision_gate"]["primary_candidate_id"])
    primary = next(row for row in recomputed_global if row["candidate_id"] == primary_id)
    compare(result["primary_candidate"], primary, global_float_fields)
    lcb_edges = [row for row in edges if as_bool(row["lcb_positive"])]
    all_rate = rate(
        sum(as_bool(row["holdout_positive"]) for row in edges), len(edges)
    )
    lcb_rate = rate(
        sum(as_bool(row["holdout_positive"]) for row in lcb_edges),
        len(lcb_edges),
    )
    edge_transfer = {
        "pair_edge_observation_count": len(edges),
        "mean_fold_spearman": statistics.fmean(
            float(row["train_holdout_pair_residual_spearman"])
            for row in recomputed_folds
        ),
        "negative_fold_spearman_count": sum(
            float(row["train_holdout_pair_residual_spearman"]) < 0.0
            for row in recomputed_folds
        ),
        "all_edge_holdout_positive_rate": all_rate,
        "lcb_positive_edge_count": len(lcb_edges),
        "lcb_positive_holdout_positive_rate": lcb_rate,
        "lcb_positive_precision_advantage": lcb_rate - all_rate,
    }
    for key, value in edge_transfer.items():
        observed = result["edge_transfer"][key]
        if isinstance(value, float):
            if abs(float(observed) - value) > TOLERANCE:
                raise ValueError(f"Stage65 edge transfer differs: {key}")
        elif int(observed) != value:
            raise ValueError(f"Stage65 edge transfer count differs: {key}")

    thresholds = config["decision_gate"]
    checks = {
        "minimum_mean_target_gain_over_pair_off": float(
            primary["mean_target_gain_over_pair_off"]
        )
        >= float(thresholds["minimum_mean_target_gain_over_pair_off"]) - TOLERANCE,
        "minimum_worst_target_gain_over_pair_off": float(
            primary["worst_target_gain_over_pair_off"]
        )
        >= float(thresholds["minimum_worst_target_gain_over_pair_off"]) - TOLERANCE,
        "minimum_nonnegative_target_count_over_pair_off": int(
            primary["nonnegative_target_count_over_pair_off"]
        )
        >= int(thresholds["minimum_nonnegative_target_count_over_pair_off"]),
        "minimum_mean_fold_pair_residual_spearman": float(
            edge_transfer["mean_fold_spearman"]
        )
        >= float(thresholds["minimum_mean_fold_pair_residual_spearman"])
        - TOLERANCE,
        "minimum_lcb_positive_edge_count": int(
            edge_transfer["lcb_positive_edge_count"]
        )
        >= int(thresholds["minimum_lcb_positive_edge_count"]),
        "minimum_lcb_positive_precision_advantage": float(
            edge_transfer["lcb_positive_precision_advantage"]
        )
        >= float(thresholds["minimum_lcb_positive_precision_advantage"])
        - TOLERANCE,
    }
    if result["decision_gate"]["checks"] != checks:
        raise ValueError("Stage65 decision checks differ")
    supported = all(checks.values())
    if bool(result["decision_gate"]["pair_residual_route_supported"]) != supported:
        raise ValueError("Stage65 decision differs")
    boundary = result["data_boundary"]
    for key in (
        "fresh_validation_rows_read",
        "locked_test_rows_read",
        "new_docking_jobs",
        "quantum_hardware_jobs",
    ):
        if int(boundary[key]) != 0:
            raise ValueError("Stage65 crossed a protected boundary")
    if result["decision"]["same_target_retuning_authorized"] is not False:
        raise ValueError("Stage65 improperly authorizes same-target retuning")
    if result["decision"]["fresh_validation_authorized"] is not False:
        raise ValueError("Stage65 improperly authorizes fresh validation")
    if result["decision"]["quantum_hardware_authorized"] is not False:
        raise ValueError("Stage65 improperly authorizes hardware")

    audit = {
        "schema_version": "1.0",
        "status": "stage65_cross_target_pair_sign_mechanism_independent_audit_ok",
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
            "edge_transfer": len(edges),
            "edge_fold_summary": len(edge_folds),
            "edge_target_summary": len(edge_targets),
            "fixed_k_metrics": len(metrics),
            "target_summary": len(target_summary),
            "global_summary": len(global_summary),
        },
        "pair_off_reproduction_cells_independently_verified": len(
            observed_pair_off
        ),
        "edge_summaries_independently_recomputed": True,
        "candidate_summaries_independently_recomputed": True,
        "decision_gate_independently_recomputed": True,
        "all_output_hashes_exact": True,
        "data_boundary_exact": True,
        "pair_residual_route_supported": supported,
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
