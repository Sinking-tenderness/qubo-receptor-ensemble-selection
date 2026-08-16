"""Develop an uncertainty-shrunk rank-pair QUBO across historical targets."""

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
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage42d_bace1_large_pool_qubo_screen import (
    bedroc_metrics,
    rank_cube,
)
from scripts.run_stage42f_bace1_rank_sensitive_pair_qubo import (
    pair_coefficients,
    qubo_value,
)


SEED_IDS = ("seed0", "seed1", "seed2")
K_VALUES = tuple(range(1, 7))
TOLERANCE = 1e-12
ROBUST_SIGMA_SCALE = 1.4826


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
        raise ValueError(f"Stage64 frozen input differs: {path}")
    return path


def subset_name(subset: tuple[int, ...], receptor_ids: list[str]) -> str:
    return "+".join(receptor_ids[index] for index in subset)


def subset_set(value: str) -> set[str]:
    result = {item for item in value.split("+") if item}
    if not result:
        raise ValueError("selected subset is empty")
    return result


def pairwise_jaccard(values: list[str]) -> float:
    sets = [subset_set(value) for value in values]
    pairs = list(itertools.combinations(sets, 2))
    return (
        statistics.fmean(len(left & right) / len(left | right) for left, right in pairs)
        if pairs
        else 1.0
    )


def build_score_cube(
    score_rows: list[dict[str, str]],
    ligand_ids: list[str],
    receptor_ids: list[str],
) -> np.ndarray:
    ligand_index = {value: index for index, value in enumerate(ligand_ids)}
    receptor_index = {value: index for index, value in enumerate(receptor_ids)}
    seed_index = {value: index for index, value in enumerate(SEED_IDS)}
    cube = np.full((3, len(ligand_ids), len(receptor_ids)), np.nan, dtype=float)
    seen: set[tuple[str, str, str]] = set()
    for row in score_rows:
        key = (row["seed_id"], row["ligand_id"], row["receptor_id"])
        if key in seen:
            raise ValueError(f"duplicate Stage64 score key: {key}")
        try:
            cube[
                seed_index[row["seed_id"]],
                ligand_index[row["ligand_id"]],
                receptor_index[row["receptor_id"]],
            ] = float(row["gpu_score"])
        except KeyError as error:
            raise ValueError(f"unknown Stage64 score identity: {key}") from error
        seen.add(key)
    expected = 3 * len(ligand_ids) * len(receptor_ids)
    if len(seen) != expected or not np.isfinite(cube).all():
        raise ValueError("Stage64 score cube is incomplete")
    return cube


def load_target(
    root: Path, target_id: str, spec: dict[str, Any]
) -> dict[str, Any]:
    paths = {key: verified(root, value) for key, value in spec["inputs"].items()}
    ligands = read_csv(paths["ligand_manifest"])
    receptors = read_csv(paths["receptor_manifest"])
    scores = read_csv(paths["scores"])
    assignments = read_csv(paths["outer_assignments"])
    expected = spec["expected"]
    if len(ligands) != int(expected["ligand_count"]):
        raise ValueError(f"{target_id} ligand count differs")
    if len(receptors) != int(expected["receptor_count"]):
        raise ValueError(f"{target_id} receptor count differs")
    if Counter(row["label"] for row in ligands) != Counter(expected["label_counts"]):
        raise ValueError(f"{target_id} label counts differ")
    if len({row["ligand_id"] for row in ligands}) != len(ligands):
        raise ValueError(f"{target_id} ligand IDs are not unique")
    if len({row["conformer_id"] for row in receptors}) != len(receptors):
        raise ValueError(f"{target_id} receptor IDs are not unique")
    allowed_receptor_statuses = set(
        expected.get("allowed_receptor_statuses", ["ok"])
    )
    if any(
        row.get("status", "ok") not in allowed_receptor_statuses
        for row in receptors
    ):
        raise ValueError(f"{target_id} receptor manifest contains a failed row")
    if any(
        row.get("status", "ok") != "ok"
        or row.get("pose_integrity_status", "ok") != "ok"
        for row in scores
    ):
        raise ValueError(f"{target_id} score matrix contains a failed row")
    observed_score_target_ids = {
        row["target_id"] for row in scores if row.get("target_id")
    }
    if observed_score_target_ids != set(expected["score_target_ids"]):
        raise ValueError(f"{target_id} score target metadata differs")
    if {row.get("split", "train") for row in ligands} != {"train"}:
        raise ValueError(f"{target_id} crossed a train boundary")
    ligand_by_id = {row["ligand_id"]: row for row in ligands}
    if len(assignments) != len(ligands) or {
        row["ligand_id"] for row in assignments
    } != set(ligand_by_id):
        raise ValueError(f"{target_id} outer assignments differ")
    outer: dict[str, int] = {}
    for row in assignments:
        ligand = ligand_by_id[row["ligand_id"]]
        if row["label"] != ligand["label"]:
            raise ValueError(f"{target_id} assignment label differs")
        if row["split_group_id"] != ligand["split_group_id"]:
            raise ValueError(f"{target_id} assignment group differs")
        outer[row["ligand_id"]] = int(row["outer_fold"])
    if set(outer.values()) != {0, 1, 2, 3}:
        raise ValueError(f"{target_id} outer fold IDs differ")
    group_folds: dict[str, set[int]] = defaultdict(set)
    for row in ligands:
        group_folds[row["split_group_id"]].add(outer[row["ligand_id"]])
    if any(len(values) != 1 for values in group_folds.values()):
        raise ValueError(f"{target_id} split group crosses outer folds")
    ligand_ids = [row["ligand_id"] for row in ligands]
    receptor_ids = [row["conformer_id"] for row in receptors]
    labels = np.asarray([int(row["label"] == "active") for row in ligands], dtype=int)
    cube = build_score_cube(scores, ligand_ids, receptor_ids)
    return {
        "target_id": target_id,
        "ligands": ligands,
        "ligand_ids": ligand_ids,
        "receptor_ids": receptor_ids,
        "labels": labels,
        "scores": cube,
        "outer": outer,
        "input_descriptors": {
            key: descriptor(root, path) for key, path in paths.items()
        },
    }


def jackknife_pair_statistics(
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
    full_singleton, full_complement = pair_coefficients(full_ranks, labels, alpha)
    assignments = make_frozen_group_folds(ligand_rows, block_count, seed)
    fold_ids = np.asarray([assignments[row["ligand_id"]] for row in ligand_rows])
    jackknife_singletons: list[np.ndarray] = []
    jackknife_complements: list[np.ndarray] = []
    for block in range(block_count):
        keep = fold_ids != block
        if int(np.sum(labels[keep] == 1)) == 0 or int(np.sum(labels[keep] == 0)) == 0:
            raise ValueError("Stage64 jackknife block removed a complete label class")
        jackknife_ranks = rank_cube(train_scores, keep)
        singleton, complement = pair_coefficients(
            jackknife_ranks[:, keep, :], labels[keep], alpha
        )
        jackknife_singletons.append(singleton)
        jackknife_complements.append(complement)
    singleton_stack = np.stack(jackknife_singletons)
    complement_stack = np.stack(jackknife_complements)
    singleton_median = np.median(singleton_stack, axis=0)
    complement_median = np.median(complement_stack, axis=0)
    singleton_spread = ROBUST_SIGMA_SCALE * np.median(
        np.abs(singleton_stack - singleton_median), axis=0
    )
    complement_spread = ROBUST_SIGMA_SCALE * np.median(
        np.abs(complement_stack - complement_median), axis=0
    )
    return {
        "full_singleton": full_singleton,
        "full_complement": full_complement,
        "singleton_spread": singleton_spread,
        "complement_spread": complement_spread,
        "positive_support": np.mean(complement_stack > TOLERANCE, axis=0),
        "negative_support": np.mean(complement_stack < -TOLERANCE, axis=0),
    }


def candidate_coefficients(
    statistics_: dict[str, np.ndarray], candidate: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    singleton = statistics_["full_singleton"].copy()
    full = statistics_["full_complement"]
    if candidate["mode"] == "baseline":
        return singleton, full.copy()
    if candidate["mode"] == "pair_off":
        return singleton, np.zeros_like(full)
    if candidate["mode"] != "soft_threshold":
        raise ValueError(f"unknown Stage64 candidate mode: {candidate['mode']}")
    shrinkage = float(candidate["lambda_mad"]) * statistics_["complement_spread"]
    complement = np.sign(full) * np.maximum(np.abs(full) - shrinkage, 0.0)
    positive = full > TOLERANCE
    negative = full < -TOLERANCE
    threshold = float(candidate["sign_support_threshold"])
    supported = (
        (positive & (statistics_["positive_support"] + TOLERANCE >= threshold))
        | (negative & (statistics_["negative_support"] + TOLERANCE >= threshold))
        | (~positive & ~negative)
    )
    complement = np.where(supported, complement, 0.0)
    complement *= float(candidate["pair_scale"])
    np.fill_diagonal(complement, 0.0)
    return singleton, complement


def coefficient_record(
    target_id: str,
    outer_fold: int,
    candidate: dict[str, Any],
    complement: np.ndarray,
    statistics_: dict[str, np.ndarray],
) -> dict[str, Any]:
    triangle = np.triu_indices(complement.shape[0], 1)
    values = complement[triangle]
    spread = statistics_["complement_spread"][triangle]
    return {
        "target_id": target_id,
        "outer_fold": outer_fold,
        "candidate_id": candidate["candidate_id"],
        "pair_count": len(values),
        "nonzero_pair_count": int(np.sum(np.abs(values) > TOLERANCE)),
        "positive_pair_count": int(np.sum(values > TOLERANCE)),
        "negative_pair_count": int(np.sum(values < -TOLERANCE)),
        "mean_absolute_pair_coefficient": float(np.mean(np.abs(values))),
        "maximum_absolute_pair_coefficient": float(np.max(np.abs(values))),
        "mean_jackknife_pair_spread": float(np.mean(spread)),
        "positive_support_75_pair_count": int(
            np.sum(statistics_["positive_support"][triangle] >= 0.75)
        ),
        "negative_support_75_pair_count": int(
            np.sum(statistics_["negative_support"][triangle] >= 0.75)
        ),
    }


def classical_by_size_cached(
    receptor_count: int,
    maximum_size: int,
    singleton: np.ndarray,
    complement: np.ndarray,
    beam_width: int,
) -> tuple[dict[int, tuple[int, ...]], dict[int, dict[str, int]]]:
    cache: dict[tuple[int, ...], float] = {}

    def cache_values(subsets: set[tuple[int, ...]] | list[tuple[int, ...]]) -> None:
        missing_by_size: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        for subset in subsets:
            if subset not in cache:
                missing_by_size[len(subset)].append(subset)
        for size, missing in missing_by_size.items():
            ordered = sorted(missing)
            indices = np.asarray(ordered, dtype=int)
            values = np.mean(singleton[indices], axis=1)
            if size > 1:
                positions = list(itertools.combinations(range(size), 2))
                left = np.asarray([value_[0] for value_ in positions], dtype=int)
                right = np.asarray([value_[1] for value_ in positions], dtype=int)
                values = values + np.mean(
                    complement[indices[:, left], indices[:, right]], axis=1
                )
            cache.update(
                (subset, float(score)) for subset, score in zip(ordered, values)
            )

    def value(subset: tuple[int, ...]) -> float:
        if subset not in cache:
            cache_values([subset])
        return cache[subset]

    def key(subset: tuple[int, ...]) -> tuple[Any, ...]:
        return (-value(subset), subset)

    def local_swap(subset: tuple[int, ...]) -> tuple[int, ...]:
        current = subset
        while True:
            selected = set(current)
            current_value = value(current)
            neighbors = {
                tuple(sorted((selected - {removed}) | {added}))
                for removed in current
                for added in range(receptor_count)
                if added not in selected
            }
            cache_values(neighbors)
            improving = [
                neighbor
                for neighbor in neighbors
                if value(neighbor) > current_value + TOLERANCE
            ]
            if not improving:
                return current
            current = min(improving, key=key)

    starts_by_size: dict[int, set[tuple[int, ...]]] = {
        1: {(index,) for index in range(receptor_count)}
    }
    beam = sorted(starts_by_size[1], key=key)[:beam_width]
    for size in range(2, maximum_size + 1):
        expanded = {
            tuple(sorted((*subset, added)))
            for subset in beam
            for added in range(receptor_count)
            if added not in subset
        }
        cache_values(expanded)
        beam = sorted(expanded, key=key)[:beam_width]
        starts_by_size[size] = set(beam)

    for initial in range(receptor_count):
        current = (initial,)
        for size in range(2, maximum_size + 1):
            selected = set(current)
            candidates = [
                tuple(sorted((*current, added)))
                for added in range(receptor_count)
                if added not in selected
            ]
            cache_values(candidates)
            current = min(candidates, key=key)
            starts_by_size[size].add(current)

    selected: dict[int, tuple[int, ...]] = {}
    records: dict[int, dict[str, int]] = {}
    for size in range(1, maximum_size + 1):
        endpoints = {local_swap(value_) for value_ in starts_by_size[size]}
        selected[size] = min(endpoints, key=key)
        records[size] = {
            "start_state_count": len(starts_by_size[size]),
            "local_endpoint_count": len(endpoints),
        }
    return selected, records


def target_screen(
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
    alpha: float,
    beam_width: int,
    block_count: int,
    jackknife_seed_base: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
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
        statistics_ = jackknife_pair_statistics(
            target["scores"][:, train_mask, :],
            labels[train_mask],
            train_rows,
            alpha,
            block_count,
            jackknife_seed_base + outer_fold,
        )
        for candidate in candidates:
            singleton, complement = candidate_coefficients(statistics_, candidate)
            selected, search_records = classical_by_size_cached(
                len(receptor_ids), 6, singleton, complement, beam_width
            )
            coefficient_rows.append(
                coefficient_record(
                    target["target_id"],
                    outer_fold,
                    candidate,
                    complement,
                    statistics_,
                )
            )
            for subset_size in K_VALUES:
                subset = selected[subset_size]
                holdout = bedroc_metrics(
                    ranks[:, holdout_mask, :],
                    labels[holdout_mask],
                    subset,
                    alpha,
                )
                metric_rows.append(
                    {
                        "target_id": target["target_id"],
                        "outer_fold": outer_fold,
                        "candidate_id": candidate["candidate_id"],
                        "subset_size": subset_size,
                        "selected_subset": subset_name(subset, receptor_ids),
                        "train_qubo_objective": qubo_value(
                            subset, singleton, complement
                        ),
                        "holdout_primary_bedroc": holdout["primary_bedroc"],
                        "holdout_mean_seed_bedroc": holdout["mean_seed_bedroc"],
                        "holdout_worst_seed_bedroc": holdout["worst_seed_bedroc"],
                        "holdout_robust_bedroc": holdout[
                            "robust_bedroc_composite"
                        ],
                        "search_start_state_count": search_records[subset_size][
                            "start_state_count"
                        ],
                        "search_local_endpoint_count": search_records[subset_size][
                            "local_endpoint_count"
                        ],
                    }
                )
        print(
            json.dumps(
                {
                    "target_id": target["target_id"],
                    "outer_fold": outer_fold,
                    "candidate_count": len(candidates),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return metric_rows, coefficient_rows


def verify_baseline_reproduction(
    metric_rows: list[dict[str, Any]], stage63_rows: list[dict[str, str]]
) -> int:
    expected = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in stage63_rows
    }
    observed = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in metric_rows
        if row["candidate_id"] == "baseline_v1"
    }
    if set(expected) != set(observed):
        raise ValueError("Stage64 baseline reproduction grid differs")
    for key, source in expected.items():
        row = observed[key]
        if row["selected_subset"] != source["selected_subset"]:
            raise ValueError(f"Stage64 baseline subset differs: {key}")
        if abs(
            float(row["holdout_robust_bedroc"])
            - float(source["holdout_robust_bedroc"])
        ) > TOLERANCE:
            raise ValueError(f"Stage64 baseline BEDROC differs: {key}")
        if abs(
            float(row["train_qubo_objective"])
            - float(source["train_qubo_objective"])
        ) > TOLERANCE:
            raise ValueError(f"Stage64 baseline objective differs: {key}")
    return len(expected)


def summarize_candidates(
    metric_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    target_order: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    baseline = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): float(
            row["holdout_robust_bedroc"]
        )
        for row in metric_rows
        if row["candidate_id"] == "baseline_v1"
    }
    pair_off = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): float(
            row["holdout_robust_bedroc"]
        )
        for row in metric_rows
        if row["candidate_id"] == "pair_off"
    }
    if set(pair_off) != set(baseline):
        raise ValueError("Stage64 pair-off comparison grid differs")
    target_rows: list[dict[str, Any]] = []
    for target_id in target_order:
        for candidate in candidates:
            rows = [
                row
                for row in metric_rows
                if row["target_id"] == target_id
                and row["candidate_id"] == candidate["candidate_id"]
                and int(row["subset_size"]) >= 2
            ]
            values = [float(row["holdout_robust_bedroc"]) for row in rows]
            gains = [
                float(row["holdout_robust_bedroc"])
                - baseline[
                    (
                        target_id,
                        int(row["outer_fold"]),
                        int(row["subset_size"]),
                    )
                ]
                for row in rows
            ]
            pair_off_gains = [
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
                    "mean_gain_over_baseline_v1": statistics.fmean(gains),
                    "minimum_fold_k_gain_over_baseline_v1": min(gains),
                    "positive_fold_k_gain_count": sum(
                        gain > TOLERANCE for gain in gains
                    ),
                    "mean_gain_over_pair_off": statistics.fmean(pair_off_gains),
                    "minimum_fold_k_gain_over_pair_off": min(pair_off_gains),
                    "positive_fold_k_gain_over_pair_off_count": sum(
                        gain > TOLERANCE for gain in pair_off_gains
                    ),
                    "mean_fixed_k_selection_jaccard": stability,
                }
            )

    target_lookup = {
        (row["target_id"], row["candidate_id"]): row for row in target_rows
    }
    global_rows: list[dict[str, Any]] = []
    for order, candidate in enumerate(candidates):
        rows = [
            target_lookup[(target_id, candidate["candidate_id"])]
            for target_id in target_order
        ]
        gains = [float(row["mean_gain_over_baseline_v1"]) for row in rows]
        pair_off_gains = [float(row["mean_gain_over_pair_off"]) for row in rows]
        global_rows.append(
            {
                "candidate_order": order,
                "candidate_id": candidate["candidate_id"],
                "mode": candidate["mode"],
                "eligible_for_freeze": bool(candidate["eligible_for_freeze"]),
                "lambda_mad": candidate["lambda_mad"],
                "sign_support_threshold": candidate["sign_support_threshold"],
                "pair_scale": candidate["pair_scale"],
                "mean_target_gain_over_baseline_v1": statistics.fmean(gains),
                "worst_target_gain_over_baseline_v1": min(gains),
                "nonnegative_target_count": sum(gain >= -TOLERANCE for gain in gains),
                "positive_target_count": sum(gain > TOLERANCE for gain in gains),
                "mean_target_gain_over_pair_off": statistics.fmean(pair_off_gains),
                "worst_target_gain_over_pair_off": min(pair_off_gains),
                "nonnegative_target_count_over_pair_off": sum(
                    gain >= -TOLERANCE for gain in pair_off_gains
                ),
                "positive_target_count_over_pair_off": sum(
                    gain > TOLERANCE for gain in pair_off_gains
                ),
                "mean_target_selection_jaccard": statistics.fmean(
                    float(row["mean_fixed_k_selection_jaccard"]) for row in rows
                ),
            }
        )
    eligible_global_rows = [
        row for row in global_rows if row["eligible_for_freeze"]
    ]
    if not eligible_global_rows:
        raise ValueError("Stage64 has no candidate eligible for freezing")
    selected_global = min(
        eligible_global_rows,
        key=lambda row: (
            -float(row["worst_target_gain_over_pair_off"]),
            -float(row["mean_target_gain_over_pair_off"]),
            -float(row["worst_target_gain_over_baseline_v1"]),
            -float(row["mean_target_gain_over_baseline_v1"]),
            int(row["candidate_order"]),
        ),
    )

    loto_rows: list[dict[str, Any]] = []
    for held_target in target_order:
        development_targets = [
            target_id for target_id in target_order if target_id != held_target
        ]
        candidate_scores = []
        for order, candidate in enumerate(candidates):
            if not bool(candidate["eligible_for_freeze"]):
                continue
            development = [
                target_lookup[(target_id, candidate["candidate_id"])]
                for target_id in development_targets
            ]
            candidate_scores.append(
                {
                    "candidate_order": order,
                    "candidate_id": candidate["candidate_id"],
                    "development_mean_gain_over_pair_off": statistics.fmean(
                        float(row["mean_gain_over_pair_off"])
                        for row in development
                    ),
                    "development_mean_fixed_k_bedroc": statistics.fmean(
                        float(row["mean_fixed_k_holdout_robust_bedroc"])
                        for row in development
                    ),
                }
            )
        selected = min(
            candidate_scores,
            key=lambda row: (
                -float(row["development_mean_gain_over_pair_off"]),
                -float(row["development_mean_fixed_k_bedroc"]),
                int(row["candidate_order"]),
            ),
        )
        held = target_lookup[(held_target, selected["candidate_id"])]
        held_baseline = target_lookup[(held_target, "baseline_v1")]
        held_pair_off = target_lookup[(held_target, "pair_off")]
        loto_rows.append(
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
                "held_target_mean_fixed_k_bedroc": held[
                    "mean_fixed_k_holdout_robust_bedroc"
                ],
                "held_target_baseline_mean_fixed_k_bedroc": held_baseline[
                    "mean_fixed_k_holdout_robust_bedroc"
                ],
                "held_target_gain_over_baseline_v1": float(
                    held["mean_fixed_k_holdout_robust_bedroc"]
                )
                - float(held_baseline["mean_fixed_k_holdout_robust_bedroc"]),
                "held_target_pair_off_mean_fixed_k_bedroc": held_pair_off[
                    "mean_fixed_k_holdout_robust_bedroc"
                ],
                "held_target_gain_over_pair_off": float(
                    held["mean_fixed_k_holdout_robust_bedroc"]
                )
                - float(held_pair_off["mean_fixed_k_holdout_robust_bedroc"]),
            }
        )
    return target_rows, global_rows, loto_rows, selected_global


def report_text(result: dict[str, Any]) -> str:
    selected = result["selected_candidate"]
    gate = result["freeze_gate"]
    return f"""# Stage64 cross-target uncertainty-shrunk rank-pair QUBO

## Scope

Stage64 uses only consumed development results from BACE1, PPARG, PPARA, and
PPARD. It performs no docking, reads no fresh-validation or locked-test row, and
runs no quantum hardware job.

## Baseline reconstruction

The original Stage42f rank-pair QUBO was reconstructed exactly in
{result['baseline_reproduction_cell_count']} fixed-k outer-fold cells before any
candidate comparison.

## Candidate selection

The maximin candidate is `{selected['candidate_id']}` with pair scale
{selected['pair_scale']}, MAD shrinkage {selected['lambda_mad']}, and sign-support
threshold {selected['sign_support_threshold']}.

- Mean target gain over baseline: {selected['mean_target_gain_over_baseline_v1']:+.6f}
- Worst target gain over baseline: {selected['worst_target_gain_over_baseline_v1']:+.6f}
- Mean target gain over pair-off: {selected['mean_target_gain_over_pair_off']:+.6f}
- Worst target gain over pair-off: {selected['worst_target_gain_over_pair_off']:+.6f}
- Nonnegative targets: {selected['nonnegative_target_count']}/4
- Positive targets: {selected['positive_target_count']}/4

## Freeze decision

Uncertainty-shrunk objective frozen: **{'PASS' if gate['objective_v2_frozen'] else 'NO-GO'}**.

Checks:

- Non-baseline candidate selected: {gate['checks']['nonbaseline_candidate_selected']}
- Minimum mean target gain: {gate['checks']['minimum_mean_target_gain']}
- Minimum worst-target gain: {gate['checks']['minimum_worst_target_gain']}
- Minimum nonnegative target count: {gate['checks']['minimum_nonnegative_target_count']}
- Minimum mean gain over pair-off: {gate['checks']['minimum_mean_target_gain_over_pair_off']}
- Minimum worst-target gain over pair-off: {gate['checks']['minimum_worst_target_gain_over_pair_off']}
- Minimum nonnegative targets over pair-off: {gate['checks']['minimum_nonnegative_target_count_over_pair_off']}
- Nonnegative leave-one-target-out mean gain: {gate['checks']['nonnegative_loto_mean_gain']}
- Minimum positive leave-one-target-out targets: {gate['checks']['minimum_positive_loto_target_count']}
- Nonnegative leave-one-target-out mean gain over pair-off: {gate['checks']['nonnegative_loto_mean_gain_over_pair_off']}
- Minimum positive leave-one-target-out targets over pair-off: {gate['checks']['minimum_positive_loto_target_count_over_pair_off']}

## Boundary

This is post-hoc cross-target objective development. A pass freezes only the pair
coefficient rule for a future nested-k evaluation and genuinely new target. It
does not establish independent efficacy, solver advantage, quantum execution,
speedup, or quantum advantage.
"""


def compute_analysis(config: dict[str, Any], root: Path) -> dict[str, Any]:
    target_order = [str(value) for value in config["development"]["target_order"]]
    candidates = [dict(value) for value in config["candidate_grid"]]
    if [candidate["candidate_id"] for candidate in candidates][0] != "baseline_v1":
        raise ValueError("Stage64 candidate grid must begin with baseline_v1")
    if len({candidate["candidate_id"] for candidate in candidates}) != len(candidates):
        raise ValueError("Stage64 candidate IDs are not unique")
    required_candidate_ids = {"baseline_v1", "pair_off"}
    if not required_candidate_ids.issubset(
        {candidate["candidate_id"] for candidate in candidates}
    ):
        raise ValueError("Stage64 baseline or pair-off diagnostic is missing")
    if any(
        candidate["candidate_id"] in required_candidate_ids
        and bool(candidate["eligible_for_freeze"])
        for candidate in candidates
    ):
        raise ValueError("Stage64 comparators cannot be eligible for freezing")
    targets = {
        target_id: load_target(root, target_id, config["targets"][target_id])
        for target_id in target_order
    }
    stage63_path = verified(root, config["inputs"]["stage63_fixed_k_landscape"])
    stage63_audit_path = verified(root, config["inputs"]["stage63_audit"])
    if read_json(stage63_audit_path).get("status") != (
        "stage63_cross_target_rank_pair_failure_diagnosis_independent_audit_ok"
    ):
        raise ValueError("Stage63 independent audit did not pass")

    metrics: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    for target_index, target_id in enumerate(target_order):
        target_metrics, target_coefficients = target_screen(
            targets[target_id],
            candidates,
            float(config["development"]["bedroc_alpha"]),
            int(config["development"]["classical_beam_width"]),
            int(config["development"]["jackknife_block_count"]),
            int(config["development"]["jackknife_seed_base"])
            + target_index * 100,
        )
        metrics.extend(target_metrics)
        coefficients.extend(target_coefficients)
    expected_metric_count = len(target_order) * 4 * len(candidates) * 6
    if len(metrics) != expected_metric_count:
        raise ValueError("Stage64 candidate metric count differs")
    reproduced = verify_baseline_reproduction(metrics, read_csv(stage63_path))
    target_rows, global_rows, loto_rows, selected = summarize_candidates(
        metrics, candidates, target_order
    )
    loto_gains = [float(row["held_target_gain_over_baseline_v1"]) for row in loto_rows]
    loto_pair_off_gains = [
        float(row["held_target_gain_over_pair_off"]) for row in loto_rows
    ]
    thresholds = config["freeze_gate"]
    checks = {
        "nonbaseline_candidate_selected": selected["candidate_id"] != "baseline_v1",
        "minimum_mean_target_gain": float(
            selected["mean_target_gain_over_baseline_v1"]
        )
        >= float(thresholds["minimum_mean_target_gain"]) - TOLERANCE,
        "minimum_worst_target_gain": float(
            selected["worst_target_gain_over_baseline_v1"]
        )
        >= float(thresholds["minimum_worst_target_gain"]) - TOLERANCE,
        "minimum_nonnegative_target_count": int(selected["nonnegative_target_count"])
        >= int(thresholds["minimum_nonnegative_target_count"]),
        "minimum_mean_target_gain_over_pair_off": float(
            selected["mean_target_gain_over_pair_off"]
        )
        >= float(thresholds["minimum_mean_target_gain_over_pair_off"]) - TOLERANCE,
        "minimum_worst_target_gain_over_pair_off": float(
            selected["worst_target_gain_over_pair_off"]
        )
        >= float(thresholds["minimum_worst_target_gain_over_pair_off"]) - TOLERANCE,
        "minimum_nonnegative_target_count_over_pair_off": int(
            selected["nonnegative_target_count_over_pair_off"]
        )
        >= int(thresholds["minimum_nonnegative_target_count_over_pair_off"]),
        "nonnegative_loto_mean_gain": statistics.fmean(loto_gains)
        >= float(thresholds["minimum_loto_mean_gain"]) - TOLERANCE,
        "minimum_positive_loto_target_count": sum(
            gain > TOLERANCE for gain in loto_gains
        )
        >= int(thresholds["minimum_positive_loto_target_count"]),
        "nonnegative_loto_mean_gain_over_pair_off": statistics.fmean(
            loto_pair_off_gains
        )
        >= float(thresholds["minimum_loto_mean_gain_over_pair_off"]) - TOLERANCE,
        "minimum_positive_loto_target_count_over_pair_off": sum(
            gain > TOLERANCE for gain in loto_pair_off_gains
        )
        >= int(thresholds["minimum_positive_loto_target_count_over_pair_off"]),
    }
    frozen = all(checks.values())
    selected_definition = next(
        candidate
        for candidate in candidates
        if candidate["candidate_id"] == selected["candidate_id"]
    )
    model_record = {
        "schema_version": "1.0",
        "status": (
            "stage64_uncertainty_shrunk_rank_pair_qubo_v2_frozen"
            if frozen
            else "stage64_uncertainty_shrunk_rank_pair_qubo_v2_not_frozen"
        ),
        "objective_id": "bedroc20_uncertainty_shrunk_rank_pair_v2",
        "selected_candidate": selected_definition,
        "formula": (
            "For fixed k, maximize mean_i(q_i) plus mean_ij(delta_tilde_ij), "
            "where delta_tilde is sign-preserving soft-thresholding of the full "
            "pair residual by lambda_mad*1.4826*MAD(delete-block estimates), then "
            "requires the selected sign in at least the support threshold fraction "
            "of jackknife estimates and applies pair_scale."
        ),
        "singleton_term_changed_from_v1": False,
        "cardinality_rule_frozen": False,
        "future_use": (
            "eligible for Stage65 nested-k development and one genuinely new "
            "target only" if frozen else "retain all candidates as negative diagnostics"
        ),
    }
    return {
        "target_input_audits": {
            target_id: {
                "ligand_count": len(targets[target_id]["ligand_ids"]),
                "receptor_count": len(targets[target_id]["receptor_ids"]),
                "score_row_count": int(np.prod(targets[target_id]["scores"].shape)),
                "input_descriptors": targets[target_id]["input_descriptors"],
            }
            for target_id in target_order
        },
        "metric_rows": metrics,
        "coefficient_rows": coefficients,
        "target_rows": target_rows,
        "global_rows": global_rows,
        "loto_rows": loto_rows,
        "baseline_reproduction_cell_count": reproduced,
        "selected_candidate": selected,
        "freeze_gate": {
            "checks": checks,
            "objective_v2_frozen": frozen,
            "loto_mean_gain_over_baseline_v1": statistics.fmean(loto_gains),
            "loto_positive_target_count": sum(gain > TOLERANCE for gain in loto_gains),
            "loto_mean_gain_over_pair_off": statistics.fmean(loto_pair_off_gains),
            "loto_positive_target_count_over_pair_off": sum(
                gain > TOLERANCE for gain in loto_pair_off_gains
            ),
        },
        "model_record": model_record,
        "decision": {
            "stage65_nested_k_authorized": frozen,
            "new_target_docking_authorized": False,
            "fresh_validation_authorized": False,
            "quantum_hardware_authorized": False,
            "same_target_retuning_authorized": False,
            "next_action": (
                "run Stage65 nested-k cross-target evaluation"
                if frozen
                else "do not freeze this objective family; redesign pair evidence"
            ),
        },
    }


def run(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    runner = root / str(config["implementation"]["runner"]["path"])
    if runner.resolve() != Path(__file__).resolve():
        raise ValueError("Stage64 runner path differs")
    for key, value in config["implementation"].items():
        path = root / str(value["path"])
        if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
            raise ValueError(f"Stage64 implementation differs: {key}")
    analysis = compute_analysis(config, root)
    outputs = {key: root / str(value) for key, value in config["outputs"].items()}
    write_csv(outputs["fixed_k_metrics_csv"], analysis["metric_rows"])
    write_csv(outputs["pair_diagnostics_csv"], analysis["coefficient_rows"])
    write_csv(outputs["target_summary_csv"], analysis["target_rows"])
    write_csv(outputs["global_summary_csv"], analysis["global_rows"])
    write_csv(outputs["loto_summary_csv"], analysis["loto_rows"])
    write_json(outputs["model_record_json"], analysis["model_record"])

    fingerprint = canonical_sha256(
        {
            key: value
            for key, value in analysis.items()
            if key != "target_input_audits"
        }
    )
    result = {
        "schema_version": "1.0",
        "status": "stage64_cross_target_uncertainty_shrunk_qubo_complete",
        "experiment_class": "post-hoc cross-target historical objective development",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, runner),
        "analysis_payload_sha256": fingerprint,
        "target_input_audits": analysis["target_input_audits"],
        "candidate_count": len(config["candidate_grid"]),
        "fixed_k_metric_count": len(analysis["metric_rows"]),
        "baseline_reproduction_cell_count": analysis[
            "baseline_reproduction_cell_count"
        ],
        "selected_candidate": analysis["selected_candidate"],
        "freeze_gate": analysis["freeze_gate"],
        "decision": analysis["decision"],
        "data_boundary": {
            "historical_development_targets_read": 4,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "interpretation_boundary": (
            "Stage64 is post-hoc objective development on consumed historical "
            "targets. A pass can authorize nested-k development and future "
            "preregistration, but cannot establish independent efficacy, solver "
            "advantage, quantum execution, speedup, or quantum advantage."
        ),
    }
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text(
        report_text({**result, "model_record": analysis["model_record"]}),
        encoding="ascii",
    )
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
