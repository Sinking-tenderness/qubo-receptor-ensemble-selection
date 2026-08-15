"""Diagnose greedy gaps and interaction order in cross-target set utilities."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage19e_cross_target_qubo_v2 import (
    load_target,
    output_descriptor,
    read_csv,
    read_json,
    rooted,
    safe_spearman,
    vectorized_bedroc,
    verified,
    write_csv,
    write_json,
)
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage05_mk14_uncertainty_qubo_gate import choose_greedy, make_context


MATRIX_IDS = ("primary", "sensitivity", "seed0", "seed1", "seed2")
SEED_IDS = ("seed0", "seed1", "seed2")
METRIC_IDS = (
    "primary",
    "sensitivity",
    "mean_seed",
    "worst_seed",
    "robust_composite",
)


def build_mask_catalog(receptor_count: int, maximum_size: int) -> dict[str, Any]:
    if not 1 <= maximum_size <= receptor_count:
        raise ValueError("invalid maximum subset size")
    masks_by_size: dict[int, np.ndarray] = {}
    ordered = [0]
    for size in range(1, maximum_size + 1):
        masks = np.asarray(
            [
                mask
                for mask in range(1, 1 << receptor_count)
                if mask.bit_count() == size
            ],
            dtype=np.int32,
        )
        masks_by_size[size] = masks
        ordered.extend(int(mask) for mask in masks)
    all_masks = np.asarray(ordered, dtype=np.int32)
    column_by_mask = np.full(1 << receptor_count, -1, dtype=np.int32)
    column_by_mask[all_masks] = np.arange(len(all_masks), dtype=np.int32)
    added_bit = np.full(1 << receptor_count, -1, dtype=np.int16)
    for mask in all_masks[1:]:
        bit = int(mask) & -int(mask)
        added_bit[int(mask)] = bit.bit_length() - 1
    return {
        "receptor_count": receptor_count,
        "maximum_size": maximum_size,
        "all_masks": all_masks,
        "nonempty_masks": all_masks[1:],
        "masks_by_size": masks_by_size,
        "column_by_mask": column_by_mask,
        "added_bit": added_bit,
    }


def mask_subset(mask: int, receptor_ids: list[str]) -> tuple[str, ...]:
    return tuple(
        receptor_id
        for index, receptor_id in enumerate(receptor_ids)
        if mask & (1 << index)
    )


def mask_string(mask: int, receptor_ids: list[str]) -> str:
    return "+".join(sorted(mask_subset(mask, receptor_ids)))


def subset_mask(subset: tuple[str, ...], receptor_ids: list[str]) -> int:
    receptor_index = {
        receptor_id: index for index, receptor_id in enumerate(receptor_ids)
    }
    return sum(1 << receptor_index[receptor_id] for receptor_id in subset)


def score_landscape(
    context: dict[str, Any],
    receptor_ids: list[str],
    split: str,
    catalog: dict[str, Any],
    alpha: float,
    batch_size: int,
) -> dict[str, np.ndarray]:
    nonempty_count = len(catalog["nonempty_masks"])
    values: dict[str, np.ndarray] = {}
    for matrix_id in MATRIX_IDS:
        rows = sorted(
            context["matrices"][matrix_id][split],
            key=lambda row: str(row["ligand_id"]),
        )
        if not rows:
            raise ValueError(f"empty scoring split: {matrix_id}/{split}")
        labels = np.asarray(
            [int(row["label"] == "active") for row in rows], dtype=np.int8
        )
        score_matrix = np.asarray(
            [
                [float(row[receptor_id]) for receptor_id in receptor_ids]
                for row in rows
            ],
            dtype=float,
        )
        aggregate = np.full(
            (len(rows), nonempty_count + 1), np.inf, dtype=float
        )
        for size in range(1, int(catalog["maximum_size"]) + 1):
            masks = catalog["masks_by_size"][size]
            columns = catalog["column_by_mask"][masks]
            parents = masks & (masks - 1)
            parent_columns = catalog["column_by_mask"][parents]
            bits = catalog["added_bit"][masks]
            aggregate[:, columns] = np.minimum(
                aggregate[:, parent_columns], score_matrix[:, bits]
            )
        metric = np.empty(nonempty_count, dtype=float)
        for start in range(1, nonempty_count + 1, batch_size):
            stop = min(nonempty_count + 1, start + batch_size)
            metric[start - 1 : stop - 1] = vectorized_bedroc(
                aggregate[:, start:stop], labels, alpha
            )
        values[matrix_id] = metric
    seeds = np.vstack([values[seed_id] for seed_id in SEED_IDS])
    values["mean_seed"] = np.mean(seeds, axis=0)
    values["worst_seed"] = np.min(seeds, axis=0)
    values["robust_composite"] = (
        values["primary"] + values["mean_seed"] + values["worst_seed"]
    ) / 3.0
    return values


def utility(values: dict[str, np.ndarray], catalog: dict[str, Any], mask: int) -> float:
    column = int(catalog["column_by_mask"][mask])
    if column <= 0:
        raise ValueError(f"mask is not in the nonempty catalog: {mask}")
    return float(values["robust_composite"][column - 1])


def metrics_for_mask(
    values: dict[str, np.ndarray], catalog: dict[str, Any], mask: int
) -> dict[str, float]:
    index = int(catalog["column_by_mask"][mask]) - 1
    return {metric_id: float(values[metric_id][index]) for metric_id in METRIC_IDS}


def prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def best_mask(
    values: dict[str, np.ndarray],
    catalog: dict[str, Any],
    receptor_ids: list[str],
    masks: np.ndarray,
) -> int:
    return min(
        (int(mask) for mask in masks),
        key=lambda mask: (
            -utility(values, catalog, mask),
            mask_subset(mask, receptor_ids),
        ),
    )


def greedy_path(
    values: dict[str, np.ndarray],
    catalog: dict[str, Any],
    receptor_ids: list[str],
) -> dict[int, int]:
    path: dict[int, int] = {}
    current = 0
    for size in range(1, int(catalog["maximum_size"]) + 1):
        candidates = np.asarray(
            [
                current | (1 << index)
                for index in range(len(receptor_ids))
                if not current & (1 << index)
            ],
            dtype=np.int32,
        )
        current = best_mask(values, catalog, receptor_ids, candidates)
        path[size] = current
    return path


def descending_rank_for_size(
    values: dict[str, np.ndarray],
    catalog: dict[str, Any],
    selected_mask: int,
) -> int:
    size = selected_mask.bit_count()
    selected = utility(values, catalog, selected_mask)
    return 1 + sum(
        utility(values, catalog, int(mask)) > selected
        for mask in catalog["masks_by_size"][size]
    )


def greedy_diagnostics(
    target_id: str,
    context_id: str,
    outer_fold: int | None,
    receptor_ids: list[str],
    catalog: dict[str, Any],
    train_values: dict[str, np.ndarray],
    holdout_values: dict[str, np.ndarray] | None,
    legacy_greedy_path: dict[int, int],
    gap_tolerance: float,
) -> list[dict[str, Any]]:
    path = greedy_path(train_values, catalog, receptor_ids)
    rows: list[dict[str, Any]] = []
    for size in range(1, int(catalog["maximum_size"]) + 1):
        exact = best_mask(
            train_values,
            catalog,
            receptor_ids,
            catalog["masks_by_size"][size],
        )
        greedy = path[size]
        legacy_greedy = legacy_greedy_path[size]
        gap = utility(train_values, catalog, exact) - utility(
            train_values, catalog, greedy
        )
        row: dict[str, Any] = {
            "target_id": target_id,
            "context_id": context_id,
            "outer_fold": outer_fold,
            "subset_size": size,
            "train_exact_subset": mask_string(exact, receptor_ids),
            "train_composite_greedy_subset": mask_string(greedy, receptor_ids),
            "train_legacy_lexicographic_greedy_subset": mask_string(
                legacy_greedy, receptor_ids
            ),
            "train_subset_differs": exact != greedy,
            "train_exact_minus_composite_greedy_robust_composite": gap,
            "train_exact_minus_legacy_greedy_robust_composite": utility(
                train_values, catalog, exact
            )
            - utility(train_values, catalog, legacy_greedy),
            "train_strict_greedy_gap": gap > gap_tolerance,
            **prefixed(
                "train_exact", metrics_for_mask(train_values, catalog, exact)
            ),
            **prefixed(
                "train_composite_greedy",
                metrics_for_mask(train_values, catalog, greedy),
            ),
            **prefixed(
                "train_legacy_greedy",
                metrics_for_mask(train_values, catalog, legacy_greedy),
            ),
        }
        if holdout_values is not None:
            oracle = best_mask(
                holdout_values,
                catalog,
                receptor_ids,
                catalog["masks_by_size"][size],
            )
            row.update(
                {
                    "holdout_oracle_subset": mask_string(oracle, receptor_ids),
                    "holdout_exact_rank": descending_rank_for_size(
                        holdout_values, catalog, exact
                    ),
                    "holdout_greedy_rank": descending_rank_for_size(
                        holdout_values, catalog, greedy
                    ),
                    "holdout_legacy_greedy_rank": descending_rank_for_size(
                        holdout_values, catalog, legacy_greedy
                    ),
                    "holdout_exact_minus_composite_greedy_robust_composite": utility(
                        holdout_values, catalog, exact
                    )
                    - utility(holdout_values, catalog, greedy),
                    "holdout_exact_minus_legacy_greedy_robust_composite": utility(
                        holdout_values, catalog, exact
                    )
                    - utility(holdout_values, catalog, legacy_greedy),
                    "holdout_oracle_minus_train_exact_robust_composite": utility(
                        holdout_values, catalog, oracle
                    )
                    - utility(holdout_values, catalog, exact),
                    **prefixed(
                        "holdout_exact",
                        metrics_for_mask(holdout_values, catalog, exact),
                    ),
                    **prefixed(
                        "holdout_composite_greedy",
                        metrics_for_mask(holdout_values, catalog, greedy),
                    ),
                    **prefixed(
                        "holdout_legacy_greedy",
                        metrics_for_mask(holdout_values, catalog, legacy_greedy),
                    ),
                    **prefixed(
                        "holdout_oracle",
                        metrics_for_mask(holdout_values, catalog, oracle),
                    ),
                }
            )
        rows.append(row)
    return rows


def submodularity_diagnostics(
    values: dict[str, np.ndarray],
    catalog: dict[str, Any],
    receptor_count: int,
    tolerance: float,
) -> dict[str, Any]:
    comparisons = 0
    violations: list[float] = []
    for size in range(1, min(4, int(catalog["maximum_size"]) - 2) + 1):
        for source in catalog["masks_by_size"][size]:
            source = int(source)
            remaining = [
                index
                for index in range(receptor_count)
                if not source & (1 << index)
            ]
            for first, second in itertools.combinations(remaining, 2):
                first_mask = source | (1 << first)
                second_mask = source | (1 << second)
                both_mask = first_mask | (1 << second)
                cross_gain = (
                    utility(values, catalog, both_mask)
                    - utility(values, catalog, second_mask)
                    - utility(values, catalog, first_mask)
                    + utility(values, catalog, source)
                )
                comparisons += 1
                if cross_gain > tolerance:
                    violations.append(cross_gain)

    marginal_count = 0
    negative_marginal_count = 0
    negative_values: list[float] = []
    for size in range(1, int(catalog["maximum_size"])):
        for source in catalog["masks_by_size"][size]:
            source = int(source)
            source_value = utility(values, catalog, source)
            for index in range(receptor_count):
                if source & (1 << index):
                    continue
                gain = utility(
                    values, catalog, source | (1 << index)
                ) - source_value
                marginal_count += 1
                if gain < -tolerance:
                    negative_marginal_count += 1
                    negative_values.append(gain)
    return {
        "submodularity_comparison_count": comparisons,
        "submodularity_violation_count": len(violations),
        "submodularity_violation_fraction": len(violations) / comparisons,
        "mean_positive_cross_gain": (
            statistics.fmean(violations) if violations else 0.0
        ),
        "maximum_positive_cross_gain": max(violations, default=0.0),
        "marginal_edge_count": marginal_count,
        "negative_marginal_count": negative_marginal_count,
        "negative_marginal_fraction": negative_marginal_count / marginal_count,
        "mean_negative_marginal": (
            statistics.fmean(negative_values) if negative_values else 0.0
        ),
        "minimum_marginal": min(negative_values, default=0.0),
    }


def top_fraction_masks(
    values: np.ndarray, masks: np.ndarray, fraction: float
) -> set[int]:
    count = max(1, int(math.ceil(len(masks) * fraction)))
    order = sorted(
        range(len(masks)),
        key=lambda index: (-float(values[index]), int(masks[index])),
    )
    return {int(masks[index]) for index in order[:count]}


def pairwise_closure(
    values: dict[str, np.ndarray],
    catalog: dict[str, Any],
    receptor_ids: list[str],
    top_fractions: list[float],
) -> dict[str, Any]:
    triples = catalog["masks_by_size"][3]
    observed = np.asarray(
        [utility(values, catalog, int(mask)) for mask in triples], dtype=float
    )
    base = np.empty(len(triples), dtype=float)
    for index, triple in enumerate(triples):
        selected = [
            bit
            for bit in range(len(receptor_ids))
            if int(triple) & (1 << bit)
        ]
        pair_sum = sum(
            utility(values, catalog, (1 << first) | (1 << second))
            for first, second in itertools.combinations(selected, 2)
        )
        singleton_sum = sum(
            utility(values, catalog, 1 << bit) for bit in selected
        )
        base[index] = pair_sum - singleton_sum
    intercept = float(np.mean(observed - base))
    predicted = base + intercept
    residual = observed - predicted
    total_sum = float(np.sum((observed - np.mean(observed)) ** 2))
    residual_sum = float(np.sum(residual**2))
    predicted_index = min(
        range(len(triples)),
        key=lambda index: (-float(predicted[index]), int(triples[index])),
    )
    exact_index = min(
        range(len(triples)),
        key=lambda index: (-float(observed[index]), int(triples[index])),
    )
    top_overlap = {}
    for fraction in top_fractions:
        observed_top = top_fraction_masks(observed, triples, fraction)
        predicted_top = top_fraction_masks(predicted, triples, fraction)
        top_overlap[f"top_{fraction:g}_overlap_fraction"] = (
            len(observed_top & predicted_top) / len(observed_top)
        )
    return {
        "intercept": intercept,
        "observed": observed,
        "predicted": predicted,
        "residual": residual,
        "rank_spearman": safe_spearman(predicted, observed),
        "r2": 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0,
        "rmse": math.sqrt(float(np.mean(residual**2))),
        "residual_standard_deviation": float(np.std(residual, ddof=0)),
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "predicted_best_mask": int(triples[predicted_index]),
        "exact_best_mask": int(triples[exact_index]),
        "predicted_best_regret": float(
            observed[exact_index] - observed[predicted_index]
        ),
        **top_overlap,
    }


def closure_scalar_fields(value: dict[str, Any]) -> dict[str, Any]:
    excluded = {"observed", "predicted", "residual"}
    return {key: item for key, item in value.items() if key not in excluded}


def mean_pairwise_spearman(values: list[np.ndarray]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.fmean(
        safe_spearman(first, second)
        for first, second in itertools.combinations(values, 2)
    )


def aggregate_size_rows(
    rows: list[dict[str, Any]], gap_tolerance: float
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["outer_fold"] is not None:
            grouped[(str(row["target_id"]), int(row["subset_size"]))].append(row)
    output: list[dict[str, Any]] = []
    for (target_id, size), selected in sorted(grouped.items()):
        gaps = [
            float(row["train_exact_minus_composite_greedy_robust_composite"])
            for row in selected
        ]
        output.append(
            {
                "target_id": target_id,
                "subset_size": size,
                "fold_count": len(selected),
                "mean_train_exact_minus_greedy": statistics.fmean(gaps),
                "maximum_train_exact_minus_greedy": max(gaps),
                "strict_train_gap_fold_count": sum(
                    value > gap_tolerance for value in gaps
                ),
                "subset_difference_fold_count": sum(
                    bool(row["train_subset_differs"]) for row in selected
                ),
                "mean_holdout_exact_minus_greedy": statistics.fmean(
                    float(
                        row[
                            "holdout_exact_minus_composite_greedy_robust_composite"
                        ]
                    )
                    for row in selected
                ),
                "holdout_exact_win_fold_count": sum(
                    float(
                        row[
                            "holdout_exact_minus_composite_greedy_robust_composite"
                        ]
                    )
                    > 0.0
                    for row in selected
                ),
                "mean_holdout_exact_rank": statistics.fmean(
                    int(row["holdout_exact_rank"]) for row in selected
                ),
                "mean_holdout_composite_greedy_rank": statistics.fmean(
                    int(row["holdout_greedy_rank"]) for row in selected
                ),
            }
        )
    return output


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Stage 19g cross-target set-function landscape",
        "",
        "## Scope",
        "",
        "Post-hoc diagnostic on MK14 and PPARG training matrices only. No new docking, BACE1 docking, fresh-validation, or test row was read.",
        "",
        "## Target decision",
        "",
        "| Target | k=3 gap folds | Qualifying sizes | Pair top-1% overlap | Pair regret | Residual stability | Route |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for target_id, value in result["target_diagnosis"].items():
        lines.append(
            f"| {target_id} | {value['current_k3_gap_fold_count']}/4 | "
            f"{','.join(str(item) for item in value['qualifying_subset_sizes']) or 'none'} | "
            f"{value['mean_pairwise_top_1pct_overlap']:.3f} | "
            f"{value['mean_pairwise_selected_regret']:.6f} | "
            f"{value['outer_train_residual_stability_spearman']:.3f} | "
            f"{value['recommended_route']} |"
        )
    lines.extend(
        [
            "",
            "## Set-function structure",
            "",
            "| Target | Submodularity violations | Negative marginals | Legacy k=3 train gap | Legacy k=3 holdout delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for target_id, value in result["target_diagnosis"].items():
        lines.append(
            f"| {target_id} | {value['mean_train_submodularity_violation_fraction']:.3f} | "
            f"{value['mean_train_negative_marginal_fraction']:.3f} | "
            f"{value['mean_k3_exact_minus_legacy_train']:.6f} | "
            f"{value['mean_k3_exact_minus_legacy_holdout']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Greedy gaps by size",
            "",
            "| Target | k | Mean train gap | Gap folds | Mean holdout delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["size_aggregate"]:
        lines.append(
            f"| {row['target_id']} | {row['subset_size']} | "
            f"{row['mean_train_exact_minus_greedy']:.6f} | "
            f"{row['strict_train_gap_fold_count']}/4 | "
            f"{row['mean_holdout_exact_minus_greedy']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Cross-target route: `{result['decision']['cross_target_route']}`",
            f"- BACE1 method amendment authorized: `{result['decision']['bace1_method_amendment_authorized']}`",
            f"- Next stage: `{result['decision']['next_stage']}`",
            "",
            "## Boundary",
            "",
            result["interpretation_boundary"],
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 19g implementation path differs")
    input_paths = {
        key: verified(root, descriptor)
        for key, descriptor in config["inputs"].items()
    }
    source_config = read_json(input_paths["stage19e_config"])
    source_result = read_json(input_paths["stage19e_result"])
    source_audit = read_json(input_paths["stage19e_audit"])
    stage19f_result = read_json(input_paths["stage19f_result"])
    stage19f_audit = read_json(input_paths["stage19f_audit"])
    if source_result["status"] != "stage19e_quadratic_v2_not_supported_do_not_amend_bace1":
        raise ValueError("Stage 19e source status differs")
    if source_audit["status"] != "stage19e_cross_target_qubo_v2_nested_diagnostic_audit_ok":
        raise ValueError("Stage 19e source audit differs")
    if stage19f_result["status"] != "stage19f_stable_pair_qubo_not_supported_do_not_amend_bace1":
        raise ValueError("Stage 19f source status differs")
    if stage19f_audit["status"] != "stage19f_cross_target_stable_pair_qubo_audit_ok":
        raise ValueError("Stage 19f source audit differs")

    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage 19g outputs exist; pass --overwrite")

    diagnostic = config["diagnostic"]
    maximum_size = int(diagnostic["maximum_subset_size"])
    bedroc_alpha = float(diagnostic["bedroc_alpha"])
    batch_size = int(diagnostic["bedroc_batch_size"])
    outer_count = int(diagnostic["outer_fold_count"])
    fold_seed = int(diagnostic["fold_seed"])
    gap_tolerance = float(diagnostic["greedy_gap_tolerance"])
    submod_tolerance = float(diagnostic["submodularity_tolerance"])
    top_fractions = [float(value) for value in diagnostic["top_fractions"]]
    receptor_count = int(diagnostic["receptor_count"])
    catalog = build_mask_catalog(receptor_count, maximum_size)

    source_methods = read_csv(input_paths["stage19e_outer_methods"])
    source_method_index = {
        (row["target_id"], int(row["outer_fold"]), row["method"]): row
        for row in source_methods
    }
    greedy_rows: list[dict[str, Any]] = []
    submod_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    triple_rows: list[dict[str, Any]] = []
    residuals_by_target: dict[str, list[np.ndarray]] = defaultdict(list)
    input_dimensions: dict[str, Any] = {}

    for target_id, target_spec in source_config["targets"].items():
        print(f"loading_target={target_id}", flush=True)
        ligands, receptor_ids, matrices = load_target(root, target_id, target_spec)
        if len(receptor_ids) != receptor_count:
            raise ValueError(f"{target_id} receptor count differs")
        assignments = make_frozen_group_folds(ligands, outer_count, fold_seed)
        all_ids = {row["ligand_id"] for row in ligands}
        model_spec = {
            "coverage_fraction": float(target_spec["v1_qubo"]["coverage_fraction"]),
            "utility_metric": "bedroc",
        }
        input_dimensions[target_id] = {
            "ligand_count": len(ligands),
            "receptor_count": len(receptor_ids),
            "nonempty_subset_count": len(catalog["nonempty_masks"]),
            "maximum_subset_size": maximum_size,
        }

        contexts: list[tuple[str, int | None, dict[str, Any], bool]] = []
        for outer_fold in range(outer_count):
            holdout = {
                ligand_id
                for ligand_id, fold in assignments.items()
                if fold == outer_fold
            }
            contexts.append(
                (
                    f"outer_{outer_fold}",
                    outer_fold,
                    make_context(
                        all_ids - holdout,
                        holdout,
                        matrices,
                        receptor_ids,
                        model_spec,
                    ),
                    True,
                )
            )
        contexts.append(
            (
                "full_train",
                None,
                make_context(all_ids, set(), matrices, receptor_ids, model_spec),
                False,
            )
        )

        for context_id, outer_fold, context, has_holdout in contexts:
            print(f"target={target_id} context={context_id}", flush=True)
            train_values = score_landscape(
                context,
                receptor_ids,
                "train",
                catalog,
                bedroc_alpha,
                batch_size,
            )
            holdout_values = (
                score_landscape(
                    context,
                    receptor_ids,
                    "validation",
                    catalog,
                    bedroc_alpha,
                    batch_size,
                )
                if has_holdout
                else None
            )
            legacy_greedy_path = {
                size: subset_mask(
                    tuple(
                        sorted(
                            choose_greedy(
                                context,
                                receptor_ids,
                                size,
                                str(target_spec["v1_qubo"]["aggregation"]),
                            )
                        )
                    ),
                    receptor_ids,
                )
                for size in range(1, maximum_size + 1)
            }
            context_greedy = greedy_diagnostics(
                target_id,
                context_id,
                outer_fold,
                receptor_ids,
                catalog,
                train_values,
                holdout_values,
                legacy_greedy_path,
                gap_tolerance,
            )
            greedy_rows.extend(context_greedy)

            if outer_fold is not None:
                k3 = next(row for row in context_greedy if row["subset_size"] == 3)
                recorded_greedy = source_method_index[
                    (target_id, outer_fold, "direct_greedy")
                ]["selected_subset"]
                recorded_exact = source_method_index[
                    (target_id, outer_fold, "composite_exact")
                ]["selected_subset"]
                if (
                    k3["train_legacy_lexicographic_greedy_subset"]
                    != recorded_greedy
                ):
                    raise ValueError(f"{target_id}/{outer_fold} greedy reproduction differs")
                if k3["train_exact_subset"] != recorded_exact:
                    raise ValueError(f"{target_id}/{outer_fold} exact reproduction differs")

            train_submod = submodularity_diagnostics(
                train_values, catalog, receptor_count, submod_tolerance
            )
            submod_row: dict[str, Any] = {
                "target_id": target_id,
                "context_id": context_id,
                "outer_fold": outer_fold,
                **prefixed("train", train_submod),
            }
            if holdout_values is not None:
                submod_row.update(
                    prefixed(
                        "holdout",
                        submodularity_diagnostics(
                            holdout_values,
                            catalog,
                            receptor_count,
                            submod_tolerance,
                        ),
                    )
                )
            submod_rows.append(submod_row)

            train_closure = pairwise_closure(
                train_values, catalog, receptor_ids, top_fractions
            )
            closure_row: dict[str, Any] = {
                "target_id": target_id,
                "context_id": context_id,
                "outer_fold": outer_fold,
                **prefixed("train", closure_scalar_fields(train_closure)),
                "train_predicted_best_subset": mask_string(
                    train_closure["predicted_best_mask"], receptor_ids
                ),
                "train_exact_best_subset": mask_string(
                    train_closure["exact_best_mask"], receptor_ids
                ),
            }
            holdout_closure = None
            if holdout_values is not None:
                holdout_closure = pairwise_closure(
                    holdout_values, catalog, receptor_ids, top_fractions
                )
                selected_mask = int(train_closure["predicted_best_mask"])
                oracle_mask = int(holdout_closure["exact_best_mask"])
                closure_row.update(
                    {
                        **prefixed(
                            "holdout", closure_scalar_fields(holdout_closure)
                        ),
                        "holdout_residual_train_correlation": safe_spearman(
                            train_closure["residual"], holdout_closure["residual"]
                        ),
                        "train_pairwise_selected_holdout_robust_composite": utility(
                            holdout_values, catalog, selected_mask
                        ),
                        "train_pairwise_selected_holdout_rank": descending_rank_for_size(
                            holdout_values, catalog, selected_mask
                        ),
                        "holdout_oracle_minus_train_pairwise_selected": utility(
                            holdout_values, catalog, oracle_mask
                        )
                        - utility(holdout_values, catalog, selected_mask),
                    }
                )
                residuals_by_target[target_id].append(train_closure["residual"])
            closure_rows.append(closure_row)

            triples = catalog["masks_by_size"][3]
            for index, mask in enumerate(triples):
                row: dict[str, Any] = {
                    "target_id": target_id,
                    "context_id": context_id,
                    "outer_fold": outer_fold,
                    "subset": mask_string(int(mask), receptor_ids),
                    "train_true_robust_composite": float(
                        train_closure["observed"][index]
                    ),
                    "train_pairwise_prediction": float(
                        train_closure["predicted"][index]
                    ),
                    "train_third_order_residual": float(
                        train_closure["residual"][index]
                    ),
                }
                if holdout_closure is not None:
                    row.update(
                        {
                            "holdout_true_robust_composite": float(
                                holdout_closure["observed"][index]
                            ),
                            "holdout_pairwise_prediction": float(
                                holdout_closure["predicted"][index]
                            ),
                            "holdout_third_order_residual": float(
                                holdout_closure["residual"][index]
                            ),
                        }
                    )
                triple_rows.append(row)

    size_aggregate = aggregate_size_rows(greedy_rows, gap_tolerance)
    gate = config["route_gate"]
    target_diagnosis: dict[str, Any] = {}
    for target_id in source_config["targets"]:
        size_rows = [row for row in size_aggregate if row["target_id"] == target_id]
        qualifying = [
            int(row["subset_size"])
            for row in size_rows
            if int(row["subset_size"]) >= int(gate["minimum_route_subset_size"])
            and float(row["mean_train_exact_minus_greedy"])
            >= float(gate["minimum_mean_train_greedy_gap"])
            and int(row["strict_train_gap_fold_count"])
            >= int(gate["minimum_gap_folds_per_target_size"])
        ]
        current = next(row for row in size_rows if int(row["subset_size"]) == 3)
        outer_closure = [
            row
            for row in closure_rows
            if row["target_id"] == target_id and row["outer_fold"] is not None
        ]
        top_key = "train_top_0.01_overlap_fraction"
        mean_top = statistics.fmean(float(row[top_key]) for row in outer_closure)
        mean_regret = statistics.fmean(
            float(row["train_predicted_best_regret"]) for row in outer_closure
        )
        residual_stability = mean_pairwise_spearman(
            residuals_by_target[target_id]
        )
        outer_submod = [
            row
            for row in submod_rows
            if row["target_id"] == target_id and row["outer_fold"] is not None
        ]
        outer_k3 = [
            row
            for row in greedy_rows
            if row["target_id"] == target_id
            and row["outer_fold"] is not None
            and int(row["subset_size"]) == 3
        ]
        pairwise_fidelity = (
            mean_top >= float(gate["minimum_mean_pairwise_top_1pct_overlap"])
            and mean_regret
            <= float(gate["maximum_mean_pairwise_selected_regret"])
        )
        enough_scale_gaps = len(qualifying) >= int(
            gate["minimum_qualifying_sizes_for_scale_route"]
        )
        current_gap = 3 in qualifying
        stable_higher_order = residual_stability >= float(
            gate["minimum_outer_residual_stability_spearman"]
        )
        if current_gap and pairwise_fidelity:
            route = "pairwise_closure_qubo_candidate"
        elif enough_scale_gaps and stable_higher_order and not pairwise_fidelity:
            route = "auxiliary_or_higher_order_qubo_candidate"
        elif enough_scale_gaps:
            route = "larger_k_greedy_gap_but_representation_unresolved"
        else:
            route = "no_stable_greedy_gap_for_efficacy_claim"
        target_diagnosis[target_id] = {
            "current_k3_mean_train_greedy_gap": current[
                "mean_train_exact_minus_greedy"
            ],
            "current_k3_gap_fold_count": current["strict_train_gap_fold_count"],
            "qualifying_subset_sizes": qualifying,
            "mean_pairwise_top_1pct_overlap": mean_top,
            "mean_pairwise_selected_regret": mean_regret,
            "mean_pairwise_rank_spearman": statistics.fmean(
                float(row["train_rank_spearman"]) for row in outer_closure
            ),
            "mean_train_holdout_residual_spearman": statistics.fmean(
                float(row["holdout_residual_train_correlation"])
                for row in outer_closure
            ),
            "outer_train_residual_stability_spearman": residual_stability,
            "mean_train_submodularity_violation_fraction": statistics.fmean(
                float(row["train_submodularity_violation_fraction"])
                for row in outer_submod
            ),
            "mean_train_negative_marginal_fraction": statistics.fmean(
                float(row["train_negative_marginal_fraction"])
                for row in outer_submod
            ),
            "mean_k3_exact_minus_legacy_train": statistics.fmean(
                float(row["train_exact_minus_legacy_greedy_robust_composite"])
                for row in outer_k3
            ),
            "mean_k3_exact_minus_legacy_holdout": statistics.fmean(
                float(row["holdout_exact_minus_legacy_greedy_robust_composite"])
                for row in outer_k3
            ),
            "pairwise_fidelity_gate": pairwise_fidelity,
            "larger_k_greedy_gap_gate": enough_scale_gaps,
            "stable_higher_order_gate": stable_higher_order,
            "recommended_route": route,
        }

    routes = {value["recommended_route"] for value in target_diagnosis.values()}
    if routes == {"pairwise_closure_qubo_candidate"}:
        cross_target_route = "develop_one_frozen_pairwise_closure_qubo"
        next_stage = "stage19h_pairwise_closure_qubo_train_only_development"
    elif routes == {"auxiliary_or_higher_order_qubo_candidate"}:
        cross_target_route = "develop_one_auxiliary_coverage_or_higher_order_qubo"
        next_stage = "stage19h_auxiliary_coverage_qubo_train_only_development"
    else:
        cross_target_route = "no_cross_target_efficacy_qubo_route_authorized"
        next_stage = "review_target_selection_or_quantum_application_only_claim"

    write_csv(outputs["greedy_path_csv"], greedy_rows)
    write_csv(outputs["submodularity_csv"], submod_rows)
    write_csv(outputs["pairwise_closure_csv"], closure_rows)
    write_csv(outputs["triple_landscape_csv"], triple_rows)
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "stage19g_cross_target_set_function_landscape_complete",
        "experiment_id": config["experiment_id"],
        "experiment_class": "posthoc_cross_target_train_only_diagnostic",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "data_boundary": {
            "train_rows_read_by_target": {
                target_id: int(spec["expected"]["ligand_count"])
                for target_id, spec in source_config["targets"].items()
            },
            "new_docking_jobs": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "bace1_docking_rows_read": 0,
        },
        "input_dimensions": input_dimensions,
        "size_aggregate": size_aggregate,
        "target_diagnosis": target_diagnosis,
        "decision": {
            "cross_target_route": cross_target_route,
            "bace1_method_amendment_authorized": False,
            "next_stage": next_stage,
        },
        "outputs": {
            key: output_descriptor(root, path)
            for key, path in outputs.items()
            if key not in ("result_json", "report_md")
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    write_report(outputs["report_md"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
