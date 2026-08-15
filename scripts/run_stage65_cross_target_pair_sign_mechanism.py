"""Adjudicate positive and negative pair evidence across historical targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage42d_bace1_large_pool_qubo_screen import bedroc_metrics, rank_cube
from scripts.run_stage42f_bace1_rank_sensitive_pair_qubo import (
    pair_coefficients,
    qubo_value,
)
from scripts.run_stage64_cross_target_uncertainty_shrunk_qubo import (
    K_VALUES,
    ROBUST_SIGMA_SCALE,
    TOLERANCE,
    classical_by_size_cached,
    load_target,
    pairwise_jaccard,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verified(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage65 frozen identity differs: {path}")
    return path


def subset_name(subset: tuple[int, ...], receptor_ids: list[str]) -> str:
    return "+".join(receptor_ids[index] for index in subset)


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


def jackknife_statistics(
    train_scores: np.ndarray,
    labels: np.ndarray,
    ligand_rows: list[dict[str, str]],
    alpha: float,
    block_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    full_ranks = rank_cube(
        train_scores, np.ones(train_scores.shape[1], dtype=bool)
    )
    full_singleton, full_pair = pair_coefficients(full_ranks, labels, alpha)
    assignments = make_frozen_group_folds(ligand_rows, block_count, seed)
    block_ids = np.asarray([assignments[row["ligand_id"]] for row in ligand_rows])
    pair_estimates: list[np.ndarray] = []
    for block in range(block_count):
        keep = block_ids != block
        ranks = rank_cube(train_scores, keep)
        _, pair = pair_coefficients(ranks[:, keep, :], labels[keep], alpha)
        pair_estimates.append(pair)
    stack = np.stack(pair_estimates)
    median = np.median(stack, axis=0)
    spread = ROBUST_SIGMA_SCALE * np.median(np.abs(stack - median), axis=0)
    return {
        "full_singleton": full_singleton,
        "full_pair": full_pair,
        "median_pair": median,
        "pair_spread": spread,
        "positive_support": np.mean(stack > TOLERANCE, axis=0),
        "negative_support": np.mean(stack < -TOLERANCE, axis=0),
    }


def candidate_pair(
    statistics_: dict[str, np.ndarray], candidate: dict[str, Any]
) -> np.ndarray:
    full = statistics_["full_pair"]
    mode = str(candidate["mode"])
    scale = float(candidate["pair_scale"])
    support = float(candidate["sign_support_threshold"])
    lambda_mad = float(candidate["lambda_mad"])
    if mode == "pair_off":
        output = np.zeros_like(full)
    elif mode == "signed":
        output = full.copy()
    elif mode == "positive":
        output = np.maximum(full, 0.0)
    elif mode == "negative":
        output = np.minimum(full, 0.0)
    elif mode == "stable_positive":
        output = np.where(
            statistics_["positive_support"] + TOLERANCE >= support,
            np.maximum(full, 0.0),
            0.0,
        )
    elif mode == "stable_negative":
        output = np.where(
            statistics_["negative_support"] + TOLERANCE >= support,
            np.minimum(full, 0.0),
            0.0,
        )
    elif mode == "lcb_positive":
        output = np.maximum(
            statistics_["median_pair"]
            - lambda_mad * statistics_["pair_spread"],
            0.0,
        )
    elif mode == "lcb_negative":
        output = np.minimum(
            statistics_["median_pair"]
            + lambda_mad * statistics_["pair_spread"],
            0.0,
        )
    else:
        raise ValueError(f"unknown Stage65 candidate mode: {mode}")
    output *= scale
    np.fill_diagonal(output, 0.0)
    return output


def edge_rows_for_fold(
    target_id: str,
    outer_fold: int,
    receptor_ids: list[str],
    statistics_: dict[str, np.ndarray],
    holdout_pair: np.ndarray,
    support_threshold: float,
    lambda_mad: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(range(len(receptor_ids)), 2):
        train_value = float(statistics_["full_pair"][left, right])
        holdout_value = float(holdout_pair[left, right])
        median = float(statistics_["median_pair"][left, right])
        spread = float(statistics_["pair_spread"][left, right])
        positive_support = float(statistics_["positive_support"][left, right])
        negative_support = float(statistics_["negative_support"][left, right])
        rows.append(
            {
                "target_id": target_id,
                "outer_fold": outer_fold,
                "left_receptor_id": receptor_ids[left],
                "right_receptor_id": receptor_ids[right],
                "train_pair_residual": train_value,
                "holdout_pair_residual": holdout_value,
                "jackknife_median_pair_residual": median,
                "jackknife_pair_spread": spread,
                "positive_sign_support": positive_support,
                "negative_sign_support": negative_support,
                "train_positive": train_value > TOLERANCE,
                "train_negative": train_value < -TOLERANCE,
                "holdout_positive": holdout_value > TOLERANCE,
                "holdout_negative": holdout_value < -TOLERANCE,
                "stable_positive": train_value > TOLERANCE
                and positive_support + TOLERANCE >= support_threshold,
                "stable_negative": train_value < -TOLERANCE
                and negative_support + TOLERANCE >= support_threshold,
                "lcb_positive": median - lambda_mad * spread > TOLERANCE,
                "lcb_negative": median + lambda_mad * spread < -TOLERANCE,
            }
        )
    return rows


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_edges(
    rows: list[dict[str, Any]], target_id: str, outer_fold: int | None
) -> dict[str, Any]:
    selected = [row for row in rows if row["target_id"] == target_id]
    if outer_fold is not None:
        selected = [row for row in selected if int(row["outer_fold"]) == outer_fold]
    train_positive = [row for row in selected if row["train_positive"]]
    train_negative = [row for row in selected if row["train_negative"]]
    stable_positive = [row for row in selected if row["stable_positive"]]
    stable_negative = [row for row in selected if row["stable_negative"]]
    lcb_positive = [row for row in selected if row["lcb_positive"]]
    lcb_negative = [row for row in selected if row["lcb_negative"]]
    return {
        "target_id": target_id,
        "outer_fold": "all" if outer_fold is None else outer_fold,
        "pair_count": len(selected),
        "train_holdout_pair_residual_spearman": spearman(
            [float(row["train_pair_residual"]) for row in selected],
            [float(row["holdout_pair_residual"]) for row in selected],
        ),
        "all_edge_holdout_positive_rate": rate(
            sum(row["holdout_positive"] for row in selected), len(selected)
        ),
        "train_positive_edge_count": len(train_positive),
        "train_positive_holdout_positive_rate": rate(
            sum(row["holdout_positive"] for row in train_positive),
            len(train_positive),
        ),
        "train_negative_edge_count": len(train_negative),
        "train_negative_holdout_negative_rate": rate(
            sum(row["holdout_negative"] for row in train_negative),
            len(train_negative),
        ),
        "stable_positive_edge_count": len(stable_positive),
        "stable_positive_holdout_positive_rate": rate(
            sum(row["holdout_positive"] for row in stable_positive),
            len(stable_positive),
        ),
        "stable_negative_edge_count": len(stable_negative),
        "stable_negative_holdout_negative_rate": rate(
            sum(row["holdout_negative"] for row in stable_negative),
            len(stable_negative),
        ),
        "lcb_positive_edge_count": len(lcb_positive),
        "lcb_positive_holdout_positive_rate": rate(
            sum(row["holdout_positive"] for row in lcb_positive),
            len(lcb_positive),
        ),
        "lcb_negative_edge_count": len(lcb_negative),
        "lcb_negative_holdout_negative_rate": rate(
            sum(row["holdout_negative"] for row in lcb_negative),
            len(lcb_negative),
        ),
    }


def summarize_candidates(
    metrics: list[dict[str, Any]],
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
            rows = [
                row
                for row in metrics
                if row["target_id"] == target_id
                and row["candidate_id"] == candidate["candidate_id"]
                and int(row["subset_size"]) >= 2
            ]
            values = [float(row["holdout_robust_bedroc"]) for row in rows]
            gains = [
                float(row["holdout_robust_bedroc"])
                - pair_off[
                    (
                        target_id,
                        int(row["outer_fold"]),
                        int(row["subset_size"]),
                    )
                ]
                for row in rows
            ]
            stability = statistics.fmean(
                pairwise_jaccard(
                    [
                        str(row["selected_subset"])
                        for row in rows
                        if int(row["subset_size"]) == subset_size
                    ]
                )
                for subset_size in range(2, 7)
            )
            target_rows.append(
                {
                    "target_id": target_id,
                    "candidate_id": candidate["candidate_id"],
                    "fixed_k_cell_count": len(rows),
                    "mean_fixed_k_holdout_robust_bedroc": statistics.fmean(values),
                    "mean_gain_over_pair_off": statistics.fmean(gains),
                    "minimum_fold_k_gain_over_pair_off": min(gains),
                    "nonnegative_fold_k_gain_over_pair_off_count": sum(
                        gain >= -TOLERANCE for gain in gains
                    ),
                    "positive_fold_k_gain_over_pair_off_count": sum(
                        gain > TOLERANCE for gain in gains
                    ),
                    "mean_fixed_k_selection_jaccard": stability,
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
                    gain >= -TOLERANCE for gain in gains
                ),
                "positive_target_count_over_pair_off": sum(
                    gain > TOLERANCE for gain in gains
                ),
                "mean_target_selection_jaccard": statistics.fmean(
                    float(row["mean_fixed_k_selection_jaccard"]) for row in rows
                ),
            }
        )
    return target_rows, global_rows


def verify_pair_off_reproduction(
    metrics: list[dict[str, Any]], stage64_metrics: list[dict[str, str]]
) -> int:
    observed = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in metrics
        if row["candidate_id"] == "pair_off"
    }
    expected = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in stage64_metrics
        if row["candidate_id"] == "pair_off"
    }
    if set(observed) != set(expected):
        raise ValueError("Stage65 pair-off reproduction grid differs")
    for key, row in observed.items():
        source = expected[key]
        if row["selected_subset"] != source["selected_subset"]:
            raise ValueError(f"Stage65 pair-off subset differs: {key}")
        for field in ("train_qubo_objective", "holdout_robust_bedroc"):
            if abs(float(row[field]) - float(source[field])) > TOLERANCE:
                raise ValueError(f"Stage65 pair-off {field} differs: {key}")
    return len(observed)


def report_text(result: dict[str, Any]) -> str:
    primary = result["primary_candidate"]
    edge = result["edge_transfer"]
    gate = result["decision_gate"]
    return f"""# Stage65 cross-target pair-sign mechanism adjudication

## Scope

Stage65 uses only consumed historical development matrices from BACE1, PPARG,
PPARA, and PPARD. It performs no docking, protected-data read, or hardware job.

## Edge transfer

- Mean fold train-versus-holdout pair-residual Spearman: {edge['mean_fold_spearman']:+.6f}
- LCB-positive edge count: {edge['lcb_positive_edge_count']}
- LCB-positive holdout-positive rate: {edge['lcb_positive_holdout_positive_rate']:.6f}
- All-edge holdout-positive rate: {edge['all_edge_holdout_positive_rate']:.6f}

## Preregistered positive-LCB candidate

Candidate `{primary['candidate_id']}` has mean target gain
{primary['mean_target_gain_over_pair_off']:+.6f} and worst-target gain
{primary['worst_target_gain_over_pair_off']:+.6f} over pair-off. It is
nonnegative on {primary['nonnegative_target_count_over_pair_off']}/4 targets.

## Decision

Continue pair-residual QUBO development: **{'PASS' if gate['pair_residual_route_supported'] else 'NO-GO'}**.

If NO-GO, the next objective must use explicit ligand/region coverage auxiliary
variables rather than another rescaling of the same pair residual.

## Boundary

This is post-hoc mechanism adjudication. It does not establish independent
efficacy, solver advantage, quantum execution, speedup, or quantum advantage.
"""


def compute_analysis(config: dict[str, Any], root: Path) -> dict[str, Any]:
    stage64_config_path = verified(root, config["inputs"]["stage64_config"])
    stage64_result_path = verified(root, config["inputs"]["stage64_result"])
    stage64_audit_path = verified(root, config["inputs"]["stage64_audit"])
    stage64_config = read_json(stage64_config_path)
    stage64_result = read_json(stage64_result_path)
    stage64_audit = read_json(stage64_audit_path)
    if stage64_result.get("status") != (
        "stage64_cross_target_uncertainty_shrunk_qubo_complete"
    ):
        raise ValueError("Stage64 source result did not complete")
    if stage64_audit.get("status") != (
        "stage64_cross_target_uncertainty_shrunk_qubo_independent_audit_ok"
    ):
        raise ValueError("Stage64 source audit did not pass")
    if stage64_result["freeze_gate"]["objective_v2_frozen"] is not False:
        raise ValueError("Stage65 requires the Stage64 no-go result")

    target_order = [str(value) for value in config["diagnosis"]["target_order"]]
    candidates = [dict(value) for value in config["candidate_grid"]]
    if candidates[0]["candidate_id"] != "pair_off":
        raise ValueError("Stage65 candidate grid must begin with pair_off")
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("Stage65 candidate IDs are not unique")
    primary_id = str(config["decision_gate"]["primary_candidate_id"])
    if primary_id not in {row["candidate_id"] for row in candidates}:
        raise ValueError("Stage65 primary candidate is absent")
    targets = {
        target_id: load_target(
            root, target_id, stage64_config["targets"][target_id]
        )
        for target_id in target_order
    }

    edge_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for target_index, target_id in enumerate(target_order):
        target = targets[target_id]
        ligand_ids = target["ligand_ids"]
        labels = target["labels"]
        receptor_ids = target["receptor_ids"]
        for outer_fold in range(4):
            train_mask = np.asarray(
                [target["outer"][ligand_id] != outer_fold for ligand_id in ligand_ids]
            )
            holdout_mask = ~train_mask
            ranks = rank_cube(target["scores"], train_mask)
            train_rows = [
                row for row, keep in zip(target["ligands"], train_mask) if keep
            ]
            statistics_ = jackknife_statistics(
                target["scores"][:, train_mask, :],
                labels[train_mask],
                train_rows,
                float(config["diagnosis"]["bedroc_alpha"]),
                int(config["diagnosis"]["jackknife_block_count"]),
                int(config["diagnosis"]["jackknife_seed_base"])
                + target_index * 100
                + outer_fold,
            )
            _, holdout_pair = pair_coefficients(
                ranks[:, holdout_mask, :],
                labels[holdout_mask],
                float(config["diagnosis"]["bedroc_alpha"]),
            )
            edge_rows.extend(
                edge_rows_for_fold(
                    target_id,
                    outer_fold,
                    receptor_ids,
                    statistics_,
                    holdout_pair,
                    float(config["diagnosis"]["stable_sign_support_threshold"]),
                    float(config["diagnosis"]["lcb_lambda_mad"]),
                )
            )
            for candidate in candidates:
                complement = candidate_pair(statistics_, candidate)
                selected, records = classical_by_size_cached(
                    len(receptor_ids),
                    6,
                    statistics_["full_singleton"],
                    complement,
                    int(config["diagnosis"]["classical_beam_width"]),
                )
                for subset_size in K_VALUES:
                    subset = selected[subset_size]
                    holdout = bedroc_metrics(
                        ranks[:, holdout_mask, :],
                        labels[holdout_mask],
                        subset,
                        float(config["diagnosis"]["bedroc_alpha"]),
                    )
                    metric_rows.append(
                        {
                            "target_id": target_id,
                            "outer_fold": outer_fold,
                            "candidate_id": candidate["candidate_id"],
                            "subset_size": subset_size,
                            "selected_subset": subset_name(subset, receptor_ids),
                            "train_qubo_objective": qubo_value(
                                subset,
                                statistics_["full_singleton"],
                                complement,
                            ),
                            "holdout_primary_bedroc": holdout["primary_bedroc"],
                            "holdout_mean_seed_bedroc": holdout["mean_seed_bedroc"],
                            "holdout_worst_seed_bedroc": holdout["worst_seed_bedroc"],
                            "holdout_robust_bedroc": holdout[
                                "robust_bedroc_composite"
                            ],
                            "search_start_state_count": records[subset_size][
                                "start_state_count"
                            ],
                            "search_local_endpoint_count": records[subset_size][
                                "local_endpoint_count"
                            ],
                        }
                    )
            print(
                json.dumps(
                    {
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "candidate_count": len(candidates),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    expected_edges = sum(
        4 * len(targets[target_id]["receptor_ids"])
        * (len(targets[target_id]["receptor_ids"]) - 1)
        // 2
        for target_id in target_order
    )
    expected_metrics = len(target_order) * 4 * len(candidates) * len(K_VALUES)
    if len(edge_rows) != expected_edges or len(metric_rows) != expected_metrics:
        raise ValueError("Stage65 output dimensions differ")
    stage64_metrics_path = verified(
        root, stage64_result["outputs"]["fixed_k_metrics_csv"]
    )
    pair_off_reproduction_count = verify_pair_off_reproduction(
        metric_rows, read_csv(stage64_metrics_path)
    )
    edge_fold_rows = [
        summarize_edges(edge_rows, target_id, fold)
        for target_id in target_order
        for fold in range(4)
    ]
    edge_target_rows = [
        summarize_edges(edge_rows, target_id, None) for target_id in target_order
    ]
    target_rows, global_rows = summarize_candidates(
        metric_rows, candidates, target_order
    )
    global_lookup = {row["candidate_id"]: row for row in global_rows}
    primary = global_lookup[primary_id]
    lcb_edges = [row for row in edge_rows if row["lcb_positive"]]
    all_positive_rate = rate(
        sum(row["holdout_positive"] for row in edge_rows), len(edge_rows)
    )
    lcb_positive_rate = rate(
        sum(row["holdout_positive"] for row in lcb_edges), len(lcb_edges)
    )
    edge_transfer = {
        "pair_edge_observation_count": len(edge_rows),
        "mean_fold_spearman": statistics.fmean(
            float(row["train_holdout_pair_residual_spearman"])
            for row in edge_fold_rows
        ),
        "negative_fold_spearman_count": sum(
            float(row["train_holdout_pair_residual_spearman"]) < 0.0
            for row in edge_fold_rows
        ),
        "all_edge_holdout_positive_rate": all_positive_rate,
        "lcb_positive_edge_count": len(lcb_edges),
        "lcb_positive_holdout_positive_rate": lcb_positive_rate,
        "lcb_positive_precision_advantage": lcb_positive_rate - all_positive_rate,
    }
    thresholds = config["decision_gate"]
    checks = {
        "minimum_mean_target_gain_over_pair_off": float(
            primary["mean_target_gain_over_pair_off"]
        )
        >= float(thresholds["minimum_mean_target_gain_over_pair_off"])
        - TOLERANCE,
        "minimum_worst_target_gain_over_pair_off": float(
            primary["worst_target_gain_over_pair_off"]
        )
        >= float(thresholds["minimum_worst_target_gain_over_pair_off"])
        - TOLERANCE,
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
    supported = all(checks.values())
    return {
        "edge_rows": edge_rows,
        "edge_fold_rows": edge_fold_rows,
        "edge_target_rows": edge_target_rows,
        "metric_rows": metric_rows,
        "target_rows": target_rows,
        "global_rows": global_rows,
        "primary_candidate": primary,
        "edge_transfer": edge_transfer,
        "decision_gate": {
            "checks": checks,
            "pair_residual_route_supported": supported,
        },
        "decision": {
            "positive_pair_objective_freeze_authorized": supported,
            "auxiliary_coverage_qubo_design_authorized": not supported,
            "new_target_docking_authorized": False,
            "fresh_validation_authorized": False,
            "quantum_hardware_authorized": False,
            "same_target_retuning_authorized": False,
            "next_action": (
                "freeze the positive-LCB pair rule for future preregistration"
                if supported
                else "stop pair-residual rescaling and design an auxiliary-variable coverage QUBO"
            ),
        },
        "pair_off_reproduction_cell_count": pair_off_reproduction_count,
        "target_input_audits": {
            target_id: {
                "ligand_count": len(targets[target_id]["ligand_ids"]),
                "receptor_count": len(targets[target_id]["receptor_ids"]),
                "score_row_count": int(np.prod(targets[target_id]["scores"].shape)),
                "input_descriptors": targets[target_id]["input_descriptors"],
            }
            for target_id in target_order
        },
    }


def run(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    runner = root / str(config["implementation"]["runner"]["path"])
    if runner.resolve() != Path(__file__).resolve():
        raise ValueError("Stage65 runner path differs")
    for key, value in config["implementation"].items():
        verified(root, value)
    analysis = compute_analysis(config, root)
    outputs = {key: root / str(value) for key, value in config["outputs"].items()}
    write_csv(outputs["edge_transfer_csv"], analysis["edge_rows"])
    write_csv(outputs["edge_fold_summary_csv"], analysis["edge_fold_rows"])
    write_csv(outputs["edge_target_summary_csv"], analysis["edge_target_rows"])
    write_csv(outputs["fixed_k_metrics_csv"], analysis["metric_rows"])
    write_csv(outputs["target_summary_csv"], analysis["target_rows"])
    write_csv(outputs["global_summary_csv"], analysis["global_rows"])
    fingerprint = canonical_sha256(
        {
            key: value
            for key, value in analysis.items()
            if key != "target_input_audits"
        }
    )
    result = {
        "schema_version": "1.0",
        "status": "stage65_cross_target_pair_sign_mechanism_complete",
        "experiment_class": "post-hoc cross-target pair-evidence mechanism adjudication",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, runner),
        "analysis_payload_sha256": fingerprint,
        "candidate_count": len(config["candidate_grid"]),
        "edge_transfer_row_count": len(analysis["edge_rows"]),
        "fixed_k_metric_count": len(analysis["metric_rows"]),
        "pair_off_reproduction_cell_count": analysis[
            "pair_off_reproduction_cell_count"
        ],
        "primary_candidate": analysis["primary_candidate"],
        "edge_transfer": analysis["edge_transfer"],
        "decision_gate": analysis["decision_gate"],
        "decision": analysis["decision"],
        "target_input_audits": analysis["target_input_audits"],
        "data_boundary": {
            "historical_development_targets_read": 4,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text(report_text(result), encoding="ascii")
    result["outputs"] = {
        key: descriptor(root, path)
        for key, path in outputs.items()
        if key not in {"result_json", "audit_json"}
    }
    write_json(outputs["result_json"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
