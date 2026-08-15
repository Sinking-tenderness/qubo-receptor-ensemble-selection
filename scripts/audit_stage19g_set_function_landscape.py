"""Independently audit the Stage 19g set-function landscape diagnostic."""

from __future__ import annotations

import argparse
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
    read_csv,
    read_json,
    rooted,
    safe_spearman,
    score_subsets,
)
from scripts.diagnose_stage19g_set_function_landscape import (
    build_mask_catalog,
    mask_string,
    subset_mask,
)
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage05_mk14_uncertainty_qubo_gate import choose_greedy, make_context


DEFAULT_CONFIG = Path("configs/stage19g_cross_target_set_function_landscape.json")
DEFAULT_OUTPUT = Path("data/stage19g_cross_target_set_function_landscape_audit.json")
METRIC_IDS = (
    "primary",
    "sensitivity",
    "mean_seed",
    "worst_seed",
    "robust_composite",
)


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def verify_descriptor(root: Path, descriptor: dict[str, Any]) -> Path:
    path = rooted(root, str(descriptor["path"]))
    if not path.is_file() or file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"descriptor differs: {path}")
    if "size_bytes" in descriptor and path.stat().st_size != int(
        descriptor["size_bytes"]
    ):
        raise ValueError(f"descriptor size differs: {path}")
    return path


def utility(values: dict[str, np.ndarray], index_by_mask: dict[int, int], mask: int) -> float:
    return float(values["robust_composite"][index_by_mask[mask]])


def metric_values(
    values: dict[str, np.ndarray], index_by_mask: dict[int, int], mask: int
) -> dict[str, float]:
    index = index_by_mask[mask]
    return {key: float(values[key][index]) for key in METRIC_IDS}


def score_independently(
    context: dict[str, Any],
    receptor_ids: list[str],
    split: str,
    masks: np.ndarray,
    alpha: float,
) -> tuple[dict[str, np.ndarray], dict[int, int]]:
    index_by_mask = {int(mask): index for index, mask in enumerate(masks)}
    values = {
        key: np.empty(len(masks), dtype=float)
        for key in (*METRIC_IDS, "seed0", "seed1", "seed2")
    }
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, mask in enumerate(masks):
        grouped[int(mask).bit_count()].append(index)
    for indices in grouped.values():
        subsets = [
            tuple(
                sorted(
                    receptor_id
                    for bit, receptor_id in enumerate(receptor_ids)
                    if int(masks[index]) & (1 << bit)
                )
            )
            for index in indices
        ]
        scored = score_subsets(context, subsets, receptor_ids, split, alpha)
        for key, array in scored.items():
            values[key][indices] = array
    return values, index_by_mask


def best_mask(
    values: dict[str, np.ndarray],
    index_by_mask: dict[int, int],
    masks: np.ndarray,
    receptor_ids: list[str],
) -> int:
    return min(
        (int(mask) for mask in masks),
        key=lambda mask: (
            -utility(values, index_by_mask, mask),
            tuple(
                sorted(
                    receptor_id
                    for index, receptor_id in enumerate(receptor_ids)
                    if mask & (1 << index)
                )
            ),
        ),
    )


def greedy_path(
    values: dict[str, np.ndarray],
    index_by_mask: dict[int, int],
    receptor_ids: list[str],
    maximum_size: int,
) -> dict[int, int]:
    selected = 0
    output: dict[int, int] = {}
    for size in range(1, maximum_size + 1):
        candidates = [
            selected | (1 << index)
            for index in range(len(receptor_ids))
            if not selected & (1 << index)
        ]
        selected = best_mask(
            values, index_by_mask, np.asarray(candidates, dtype=np.int32), receptor_ids
        )
        output[size] = selected
    return output


def rank_at_size(
    values: dict[str, np.ndarray],
    index_by_mask: dict[int, int],
    masks: np.ndarray,
    selected: int,
) -> int:
    selected_value = utility(values, index_by_mask, selected)
    return 1 + sum(
        utility(values, index_by_mask, int(mask)) > selected_value for mask in masks
    )


def submodularity(
    values: dict[str, np.ndarray],
    index_by_mask: dict[int, int],
    masks_by_size: dict[int, np.ndarray],
    receptor_count: int,
    maximum_size: int,
    tolerance: float,
) -> dict[str, Any]:
    comparisons = 0
    violations: list[float] = []
    for size in range(1, min(4, maximum_size - 2) + 1):
        for source_value in masks_by_size[size]:
            source = int(source_value)
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
                    utility(values, index_by_mask, both_mask)
                    - utility(values, index_by_mask, second_mask)
                    - utility(values, index_by_mask, first_mask)
                    + utility(values, index_by_mask, source)
                )
                comparisons += 1
                if cross_gain > tolerance:
                    violations.append(cross_gain)
    marginal_count = 0
    negative: list[float] = []
    for size in range(1, maximum_size):
        for source_value in masks_by_size[size]:
            source = int(source_value)
            source_utility = utility(values, index_by_mask, source)
            for index in range(receptor_count):
                if source & (1 << index):
                    continue
                gain = utility(
                    values, index_by_mask, source | (1 << index)
                ) - source_utility
                marginal_count += 1
                if gain < -tolerance:
                    negative.append(gain)
    return {
        "submodularity_comparison_count": comparisons,
        "submodularity_violation_count": len(violations),
        "submodularity_violation_fraction": len(violations) / comparisons,
        "mean_positive_cross_gain": statistics.fmean(violations) if violations else 0.0,
        "maximum_positive_cross_gain": max(violations, default=0.0),
        "marginal_edge_count": marginal_count,
        "negative_marginal_count": len(negative),
        "negative_marginal_fraction": len(negative) / marginal_count,
        "mean_negative_marginal": statistics.fmean(negative) if negative else 0.0,
        "minimum_marginal": min(negative, default=0.0),
    }


def pairwise_closure(
    values: dict[str, np.ndarray],
    index_by_mask: dict[int, int],
    triples: np.ndarray,
    receptor_count: int,
    top_fractions: list[float],
) -> dict[str, Any]:
    observed = np.asarray(
        [utility(values, index_by_mask, int(mask)) for mask in triples]
    )
    base = np.empty(len(triples), dtype=float)
    for row_index, triple_value in enumerate(triples):
        triple = int(triple_value)
        bits = [index for index in range(receptor_count) if triple & (1 << index)]
        base[row_index] = sum(
            utility(values, index_by_mask, (1 << first) | (1 << second))
            for first, second in itertools.combinations(bits, 2)
        ) - sum(utility(values, index_by_mask, 1 << bit) for bit in bits)
    intercept = float(np.mean(observed - base))
    predicted = base + intercept
    residual = observed - predicted
    total = float(np.sum((observed - np.mean(observed)) ** 2))
    residual_sum = float(np.sum(residual**2))

    def top_set(array: np.ndarray, fraction: float) -> set[int]:
        count = max(1, int(math.ceil(len(triples) * fraction)))
        order = sorted(
            range(len(triples)),
            key=lambda index: (-float(array[index]), int(triples[index])),
        )
        return {int(triples[index]) for index in order[:count]}

    predicted_best = min(
        range(len(triples)), key=lambda index: (-float(predicted[index]), int(triples[index]))
    )
    exact_best = min(
        range(len(triples)), key=lambda index: (-float(observed[index]), int(triples[index]))
    )
    result: dict[str, Any] = {
        "intercept": intercept,
        "rank_spearman": safe_spearman(predicted, observed),
        "r2": 1.0 - residual_sum / total if total else 1.0,
        "rmse": math.sqrt(float(np.mean(residual**2))),
        "residual_standard_deviation": float(np.std(residual, ddof=0)),
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "predicted_best_mask": int(triples[predicted_best]),
        "exact_best_mask": int(triples[exact_best]),
        "predicted_best_regret": float(
            observed[exact_best] - observed[predicted_best]
        ),
        "observed": observed,
        "predicted": predicted,
        "residual": residual,
    }
    for fraction in top_fractions:
        result[f"top_{fraction:g}_overlap_fraction"] = len(
            top_set(observed, fraction) & top_set(predicted, fraction)
        ) / max(1, int(math.ceil(len(triples) * fraction)))
    return result


def compare_metrics(
    row: dict[str, str],
    values: dict[str, np.ndarray],
    index_by_mask: dict[int, int],
    mask: int,
    prefix: str,
    label: str,
) -> None:
    metrics = metric_values(values, index_by_mask, mask)
    for key in METRIC_IDS:
        assert_close(float(row[f"{prefix}_{key}"]), metrics[key], f"{label}/{key}")


def compare_scalar_dict(
    row: dict[str, str], expected: dict[str, Any], prefix: str, label: str
) -> None:
    for key, value in expected.items():
        field = f"{prefix}_{key}"
        if field in row:
            if isinstance(value, int):
                if int(row[field]) != value:
                    raise ValueError(f"{label}/{field} differs")
            else:
                assert_close(float(row[field]), float(value), f"{label}/{field}")


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = rooted(root, config_path.as_posix())
    config = read_json(config_path)
    verify_descriptor(root, config["implementation"])
    input_paths = {
        key: verify_descriptor(root, descriptor)
        for key, descriptor in config["inputs"].items()
    }
    source_config = read_json(input_paths["stage19e_config"])
    for key in ("stage19e_result", "stage19e_audit", "stage19f_result", "stage19f_audit"):
        source = read_json(input_paths[key])
        if key.endswith("result") and "status" not in source:
            raise ValueError(f"missing source status: {key}")
    result_path = rooted(root, config["outputs"]["result_json"])
    result = read_json(result_path)
    if result["status"] != "stage19g_cross_target_set_function_landscape_complete":
        raise ValueError("unexpected Stage 19g status")
    if result["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("result identifies another config")
    output_paths = {
        key: verify_descriptor(root, descriptor)
        for key, descriptor in result["outputs"].items()
    }
    greedy_rows = read_csv(output_paths["greedy_path_csv"])
    submod_rows = read_csv(output_paths["submodularity_csv"])
    closure_rows = read_csv(output_paths["pairwise_closure_csv"])
    triple_rows = read_csv(output_paths["triple_landscape_csv"])
    expected_greedy = len(source_config["targets"]) * 5 * int(
        config["diagnostic"]["maximum_subset_size"]
    )
    expected_contexts = len(source_config["targets"]) * 5
    expected_triples = len(source_config["targets"]) * 5 * 560
    if len(greedy_rows) != expected_greedy:
        raise ValueError("greedy row count differs")
    if len(submod_rows) != expected_contexts or len(closure_rows) != expected_contexts:
        raise ValueError("context diagnostic row count differs")
    if len(triple_rows) != expected_triples:
        raise ValueError("triple landscape row count differs")

    diagnostic = config["diagnostic"]
    maximum_size = int(diagnostic["maximum_subset_size"])
    receptor_count = int(diagnostic["receptor_count"])
    catalog = build_mask_catalog(receptor_count, maximum_size)
    masks = catalog["nonempty_masks"]
    index_by_mask = {int(mask): index for index, mask in enumerate(masks)}
    masks_by_size = catalog["masks_by_size"]
    triples = masks_by_size[3]
    alpha = float(diagnostic["bedroc_alpha"])
    gap_tolerance = float(diagnostic["greedy_gap_tolerance"])
    submod_tolerance = float(diagnostic["submodularity_tolerance"])
    top_fractions = [float(value) for value in diagnostic["top_fractions"]]
    source_methods = read_csv(input_paths["stage19e_outer_methods"])
    source_method_index = {
        (row["target_id"], int(row["outer_fold"]), row["method"]): row
        for row in source_methods
    }

    raw_context_count = 0
    raw_triple_count = 0
    residuals: dict[str, list[np.ndarray]] = defaultdict(list)
    for target_id, target_spec in source_config["targets"].items():
        ligands, receptor_ids, matrices = load_target(root, target_id, target_spec)
        assignments = make_frozen_group_folds(ligands, 4, int(diagnostic["fold_seed"]))
        all_ids = {row["ligand_id"] for row in ligands}
        model_spec = {
            "coverage_fraction": float(target_spec["v1_qubo"]["coverage_fraction"]),
            "utility_metric": "bedroc",
        }
        contexts: list[tuple[str, int | None, dict[str, Any], bool]] = []
        for outer_fold in range(4):
            holdout = {
                ligand_id
                for ligand_id, fold in assignments.items()
                if fold == outer_fold
            }
            contexts.append(
                (
                    f"outer_{outer_fold}",
                    outer_fold,
                    make_context(all_ids - holdout, holdout, matrices, receptor_ids, model_spec),
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
            train_values, train_index = score_independently(
                context, receptor_ids, "train", masks, alpha
            )
            holdout_values = None
            if has_holdout:
                holdout_values, holdout_index = score_independently(
                    context, receptor_ids, "validation", masks, alpha
                )
            raw_context_count += 1
            selected_rows = {
                int(row["subset_size"]): row
                for row in greedy_rows
                if row["target_id"] == target_id
                and row["context_id"] == context_id
            }
            path = greedy_path(
                train_values, train_index, receptor_ids, maximum_size
            )
            for size in range(1, maximum_size + 1):
                exact = best_mask(
                    train_values,
                    train_index,
                    masks_by_size[size],
                    receptor_ids,
                )
                row = selected_rows[size]
                if row["train_exact_subset"] != mask_string(exact, receptor_ids):
                    raise ValueError(f"{target_id}/{context_id}/exact subset differs")
                composite = path[size]
                if row["train_composite_greedy_subset"] != mask_string(
                    composite, receptor_ids
                ):
                    raise ValueError(f"{target_id}/{context_id}/greedy subset differs")
                legacy = subset_mask(
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
                if row["train_legacy_lexicographic_greedy_subset"] != mask_string(
                    legacy, receptor_ids
                ):
                    raise ValueError(f"{target_id}/{context_id}/legacy greedy differs")
                assert_close(
                    float(row["train_exact_minus_composite_greedy_robust_composite"]),
                    utility(train_values, train_index, exact)
                    - utility(train_values, train_index, composite),
                    f"{target_id}/{context_id}/{size}/train gap",
                )
                compare_metrics(
                    row, train_values, train_index, exact, "train_exact", f"{target_id}/{context_id}/{size}/exact"
                )
                compare_metrics(
                    row, train_values, train_index, composite, "train_composite_greedy", f"{target_id}/{context_id}/{size}/greedy"
                )
                compare_metrics(
                    row, train_values, train_index, legacy, "train_legacy_greedy", f"{target_id}/{context_id}/{size}/legacy"
                )
                if holdout_values is not None:
                    if row["holdout_oracle_subset"] != mask_string(
                        best_mask(
                            holdout_values,
                            holdout_index,
                            masks_by_size[size],
                            receptor_ids,
                        ),
                        receptor_ids,
                    ):
                        raise ValueError(f"{target_id}/{context_id}/{size}/oracle differs")
                    for selected, prefix in (
                        (exact, "holdout_exact"),
                        (composite, "holdout_composite_greedy"),
                        (legacy, "holdout_legacy_greedy"),
                    ):
                        compare_metrics(
                            row,
                            holdout_values,
                            holdout_index,
                            selected,
                            prefix,
                            f"{target_id}/{context_id}/{size}/{prefix}",
                        )

            train_submod = submodularity(
                train_values,
                train_index,
                masks_by_size,
                receptor_count,
                maximum_size,
                submod_tolerance,
            )
            submod_row = next(
                row
                for row in submod_rows
                if row["target_id"] == target_id and row["context_id"] == context_id
            )
            compare_scalar_dict(submod_row, train_submod, "train", f"{target_id}/{context_id}/submod")
            if holdout_values is not None:
                compare_scalar_dict(
                    submod_row,
                    submodularity(
                        holdout_values,
                        holdout_index,
                        masks_by_size,
                        receptor_count,
                        maximum_size,
                        submod_tolerance,
                    ),
                    "holdout",
                    f"{target_id}/{context_id}/holdout submod",
                )

            train_closure = pairwise_closure(
                train_values, train_index, triples, receptor_count, top_fractions
            )
            closure_row = next(
                row
                for row in closure_rows
                if row["target_id"] == target_id and row["context_id"] == context_id
            )
            for key in (
                "intercept",
                "rank_spearman",
                "r2",
                "rmse",
                "residual_standard_deviation",
                "maximum_absolute_residual",
                "predicted_best_regret",
                *[f"top_{fraction:g}_overlap_fraction" for fraction in top_fractions],
            ):
                assert_close(
                    float(closure_row[f"train_{key}"]),
                    float(train_closure[key]),
                    f"{target_id}/{context_id}/train closure/{key}",
                )
            for key, expected in (
                ("train_predicted_best_subset", train_closure["predicted_best_mask"]),
                ("train_exact_best_subset", train_closure["exact_best_mask"]),
            ):
                expected_string = mask_string(int(expected), receptor_ids)
                if closure_row[key] != expected_string:
                    raise ValueError(f"{target_id}/{context_id}/{key} differs")

            holdout_closure = None
            if holdout_values is not None:
                holdout_closure = pairwise_closure(
                    holdout_values,
                    holdout_index,
                    triples,
                    receptor_count,
                    top_fractions,
                )
                for key in (
                    "intercept",
                    "rank_spearman",
                    "r2",
                    "rmse",
                    "residual_standard_deviation",
                    "maximum_absolute_residual",
                    "predicted_best_regret",
                    *[f"top_{fraction:g}_overlap_fraction" for fraction in top_fractions],
                ):
                    assert_close(
                        float(closure_row[f"holdout_{key}"]),
                        float(holdout_closure[key]),
                        f"{target_id}/{context_id}/holdout closure/{key}",
                    )
                assert_close(
                    float(closure_row["holdout_residual_train_correlation"]),
                    safe_spearman(
                        train_closure["residual"], holdout_closure["residual"]
                    ),
                    f"{target_id}/{context_id}/residual correlation",
                )
                residuals[target_id].append(train_closure["residual"])

            triple_index = {
                row["subset"]: row
                for row in triple_rows
                if row["target_id"] == target_id
                and row["context_id"] == context_id
            }
            if len(triple_index) != len(triples):
                raise ValueError(f"{target_id}/{context_id}/triple rows differ")
            for index, mask in enumerate(triples):
                row = triple_index[mask_string(int(mask), receptor_ids)]
                assert_close(
                    float(row["train_true_robust_composite"]),
                    float(train_closure["observed"][index]),
                    f"{target_id}/{context_id}/triple/{index}/train true",
                )
                assert_close(
                    float(row["train_pairwise_prediction"]),
                    float(train_closure["predicted"][index]),
                    f"{target_id}/{context_id}/triple/{index}/train prediction",
                )
                assert_close(
                    float(row["train_third_order_residual"]),
                    float(train_closure["residual"][index]),
                    f"{target_id}/{context_id}/triple/{index}/train residual",
                )
                if holdout_closure is not None:
                    assert_close(
                        float(row["holdout_true_robust_composite"]),
                        float(holdout_closure["observed"][index]),
                        f"{target_id}/{context_id}/triple/{index}/holdout true",
                    )
                raw_triple_count += 1

    # The result's gate is intentionally descriptive and can never authorize BACE1.
    if result["decision"]["bace1_method_amendment_authorized"] is not False:
        raise ValueError("Stage 19g authorized a protected-panel amendment")
    if result["data_boundary"] != {
        "train_rows_read_by_target": {"MK14": 696, "PPARG": 668},
        "new_docking_jobs": 0,
        "fresh_validation_rows_read": 0,
        "test_rows_read": 0,
        "bace1_docking_rows_read": 0,
    }:
        raise ValueError("Stage 19g data boundary differs")

    return {
        "schema_version": "1.0",
        "status": "stage19g_cross_target_set_function_landscape_audit_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "result": {
            "path": result_path.relative_to(root).as_posix(),
            "sha256": file_sha256(result_path),
        },
        "auditor": {
            "path": Path(__file__).resolve().relative_to(root).as_posix(),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "coverage": {
            "raw_contexts_recomputed": raw_context_count,
            "greedy_rows_recomputed": len(greedy_rows),
            "submodularity_rows_recomputed": len(submod_rows),
            "pairwise_closure_rows_recomputed": len(closure_rows),
            "triple_rows_recomputed": raw_triple_count,
        },
        "checks": {
            "all_input_and_output_hashes_verified": True,
            "all_set_utilities_recomputed_independently": True,
            "composite_greedy_and_exact_paths_recomputed": True,
            "legacy_greedy_reproduced": True,
            "submodularity_diagnostics_recomputed": True,
            "pairwise_closure_and_residuals_recomputed": True,
            "failed_cross_target_route_preserved": result["decision"][
                "cross_target_route"
            ]
            == "no_cross_target_efficacy_qubo_route_authorized",
            "bace1_method_amendment_authorized": False,
            "new_docking_jobs": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "bace1_docking_rows_read": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.config, args.root)
    output = rooted(args.root.resolve(), args.output.as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
