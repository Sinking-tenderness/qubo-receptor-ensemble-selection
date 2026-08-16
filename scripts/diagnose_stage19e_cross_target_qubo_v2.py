"""Diagnose a transferable quadratic QUBO v2 on MK14 and PPARG train rows."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json, write_json  # noqa: F401 (deduped)
import argparse
import csv
import itertools
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage12a_mk14_qubo_objective_adequacy import (
    build_explicit_qubo,
    design_matrix,
    exact_all_cardinalities,
    mean_jaccard,
)
from scripts.normalized_receptor_qubo import build_coefficients, coefficient_energy
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage05_mk14_uncertainty_qubo_gate import (
    choose_exhaustive,
    choose_greedy,
    make_context,
    pair_synergy_terms_for_aggregation,
)
from scripts.screen_stage10_mk14_expanded16_qubo_greedy import (
    build_matrices,
    fixed_cardinality_exact,
)


MATRIX_IDS = ("primary", "sensitivity", "seed0", "seed1", "seed2")
SEED_IDS = ("seed0", "seed1", "seed2")
SURROGATE_FAMILIES = ("additive", "quadratic")




def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows




def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rooted(root: Path, value: str) -> Path:
    path = (root / value.replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path leaves repository root: {value}") from error
    return path


def verified(root: Path, descriptor: dict[str, Any]) -> Path:
    path = rooted(root, str(descriptor["path"]))
    if not path.is_file() or file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"input identity differs: {path}")
    return path


def output_descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else 0.0


def choose_highest(values: np.ndarray, subsets: list[tuple[str, ...]]) -> int:
    return min(
        range(len(subsets)),
        key=lambda index: (-float(values[index]), subsets[index]),
    )


def descending_rank(values: np.ndarray, selected_index: int) -> int:
    selected = float(values[selected_index])
    return 1 + sum(float(value) > selected for value in values)


def vectorized_bedroc(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if scores.ndim != 2 or labels.ndim != 1 or scores.shape[0] != len(labels):
        raise ValueError("BEDROC array dimensions differ")
    total = len(labels)
    active_total = int(labels.sum())
    if total == 0 or active_total == 0 or active_total == total:
        raise ValueError("BEDROC requires both active and decoy rows")
    order = np.argsort(scores, axis=0, kind="stable")
    ranked_labels = labels[order]
    weights = np.exp(-alpha * np.arange(1, total + 1, dtype=float) / total)
    random_expected = active_total * float(weights.mean())
    observed_rie = np.sum(ranked_labels * weights[:, None], axis=0) / random_expected
    maximum_rie = float(weights[:active_total].sum()) / random_expected
    minimum_rie = float(weights[-active_total:].sum()) / random_expected
    return np.asarray(
        (observed_rie - minimum_rie) / (maximum_rie - minimum_rie),
        dtype=float,
    )


def score_subsets(
    context: dict[str, Any],
    subsets: list[tuple[str, ...]],
    receptor_ids: list[str],
    split: str,
    alpha: float,
) -> dict[str, np.ndarray]:
    receptor_index = {
        receptor_id: index for index, receptor_id in enumerate(receptor_ids)
    }
    subset_columns = np.asarray(
        [[receptor_index[value] for value in subset] for subset in subsets],
        dtype=int,
    )
    values: dict[str, np.ndarray] = {}
    for matrix_id in MATRIX_IDS:
        rows = sorted(
            context["matrices"][matrix_id][split],
            key=lambda row: str(row["ligand_id"]),
        )
        if not rows:
            raise ValueError(f"empty scoring split: {matrix_id}/{split}")
        labels = np.asarray(
            [int(row["label"] == "active") for row in rows], dtype=int
        )
        score_matrix = np.asarray(
            [
                [float(row[receptor_id]) for receptor_id in receptor_ids]
                for row in rows
            ],
            dtype=float,
        )
        aggregate = np.min(score_matrix[:, subset_columns], axis=2)
        values[matrix_id] = vectorized_bedroc(aggregate, labels, alpha)
    seed_values = np.vstack([values[seed_id] for seed_id in SEED_IDS])
    values["mean_seed"] = np.mean(seed_values, axis=0)
    values["worst_seed"] = np.min(seed_values, axis=0)
    values["robust_composite"] = (
        values["primary"] + values["mean_seed"] + values["worst_seed"]
    ) / 3.0
    return values


def subset_metrics(values: dict[str, np.ndarray], index: int) -> dict[str, float]:
    return {
        key: float(values[key][index])
        for key in (
            "primary",
            "sensitivity",
            "mean_seed",
            "worst_seed",
            "robust_composite",
        )
    }


def prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def fit_surrogate(
    features: np.ndarray, target: np.ndarray, alpha: float
) -> tuple[Ridge, np.ndarray]:
    model = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
    model.fit(features, target)
    return model, np.asarray(model.predict(features), dtype=float)


def alpha_summary_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["mean_validation_robust_composite"]),
        -float(row["worst_validation_robust_composite"]),
        -float(row["mean_validation_rank_spearman"]),
        float(row["alpha"]),
    )


def summarize_alpha_trials(
    rows: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields) + (float(row["alpha"]),)
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items(), key=lambda item: item[0]):
        prefix_values = dict(zip((*group_fields, "alpha"), key))
        subsets = [tuple(str(row["selected_subset"]).split("+")) for row in selected]
        output.append(
            {
                **prefix_values,
                "fold_count": len(selected),
                "mean_validation_robust_composite": statistics.fmean(
                    float(row["validation_robust_composite"]) for row in selected
                ),
                "worst_validation_robust_composite": min(
                    float(row["validation_robust_composite"]) for row in selected
                ),
                "mean_validation_primary": statistics.fmean(
                    float(row["validation_primary"]) for row in selected
                ),
                "mean_validation_worst_seed": statistics.fmean(
                    float(row["validation_worst_seed"]) for row in selected
                ),
                "mean_validation_rank_spearman": statistics.fmean(
                    float(row["validation_rank_spearman"]) for row in selected
                ),
                "mean_selected_subset_jaccard": mean_jaccard(subsets),
                "selected_subsets": ";".join(
                    str(row["selected_subset"]) for row in selected
                ),
            }
        )
    return output


def select_alpha(
    summary_rows: list[dict[str, Any]],
    target_id: str,
    family: str,
    outer_fold: int | None,
) -> dict[str, Any]:
    candidates = [
        row
        for row in summary_rows
        if row["target_id"] == target_id
        and row["model_family"] == family
        and (
            outer_fold is None
            or int(row.get("outer_fold", -1)) == int(outer_fold)
        )
    ]
    if not candidates:
        raise ValueError(f"no alpha candidates: {target_id}/{family}/{outer_fold}")
    return min(candidates, key=alpha_summary_key)


def select_v1(
    context: dict[str, Any],
    receptor_ids: list[str],
    target_spec: dict[str, Any],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    frozen = target_spec["v1_qubo"]
    target_size = int(frozen["target_size"])
    terms = pair_synergy_terms_for_aggregation(
        context["terms"], str(frozen["aggregation"])
    )
    coefficients = build_coefficients(
        terms,
        receptor_ids,
        target_size,
        {key: float(value) for key, value in frozen["weights"].items()},
        float(frozen["size_penalty"]),
    )
    subset, energy = fixed_cardinality_exact(
        coefficients, receptor_ids, target_size
    )
    return tuple(sorted(subset)), {"energy": energy, "coefficients": coefficients}


def load_target(
    root: Path, target_id: str, spec: dict[str, Any]
) -> tuple[list[dict[str, str]], list[str], dict[str, dict[str, dict[str, Any]]]]:
    paths = {
        key: verified(root, descriptor)
        for key, descriptor in spec["inputs"].items()
    }
    summary = read_json(paths["summary"])
    audit = read_json(paths["audit"])
    if summary.get("status") != spec["source_checks"]["summary_status"]:
        raise ValueError(f"{target_id} source summary did not pass")
    if audit.get("status") != spec["source_checks"]["audit_status"]:
        raise ValueError(f"{target_id} source audit did not pass")
    if "amendment" in paths:
        amendment = read_json(paths["amendment"])
        if amendment.get("status") != spec["source_checks"]["amendment_status"]:
            raise ValueError(f"{target_id} metadata amendment did not pass")

    ligands = read_csv(paths["ligand_manifest"])
    receptors = read_csv(paths["receptor_manifest"])
    scores = read_csv(paths["seed_scores"])
    expected = spec["expected"]
    receptor_ids = [row["conformer_id"] for row in receptors]
    if len(ligands) != int(expected["ligand_count"]):
        raise ValueError(f"{target_id} ligand count differs")
    if len(receptor_ids) != int(expected["receptor_count"]):
        raise ValueError(f"{target_id} receptor count differs")
    if Counter(row["label"] for row in ligands) != Counter(
        {key: int(value) for key, value in expected["label_counts"].items()}
    ):
        raise ValueError(f"{target_id} label counts differ")
    if {row["split"] for row in ligands} != {"train"}:
        raise ValueError(f"{target_id} observed a non-train ligand")
    if {row["target_id"] for row in ligands} != {target_id}:
        raise ValueError(f"{target_id} ligand metadata differs")
    if len(scores) != int(expected["pair_count"]):
        raise ValueError(f"{target_id} seed score count differs")
    if "target_id" in scores[0] and {row["target_id"] for row in scores} != {
        target_id
    }:
        raise ValueError(f"{target_id} score metadata differs")
    matrices = build_matrices(
        read_csv(paths["primary_matrix"]),
        read_csv(paths["sensitivity_matrix"]),
        scores,
        ligands,
        receptor_ids,
    )
    return ligands, receptor_ids, matrices


def method_row(
    target_id: str,
    outer_fold: int,
    method: str,
    subset: tuple[str, ...],
    subset_index: dict[tuple[str, ...], int],
    train_values: dict[str, np.ndarray],
    holdout_values: dict[str, np.ndarray],
    **extra: Any,
) -> dict[str, Any]:
    canonical = tuple(sorted(subset))
    index = subset_index[canonical]
    return {
        "target_id": target_id,
        "outer_fold": outer_fold,
        "method": method,
        "selected_subset": "+".join(canonical),
        "holdout_rank": descending_rank(
            holdout_values["robust_composite"], index
        ),
        **prefixed("train", subset_metrics(train_values, index)),
        **prefixed("holdout", subset_metrics(holdout_values, index)),
        **extra,
    }


def aggregate_outer_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target_id"]), str(row["method"]))].append(row)
    output: list[dict[str, Any]] = []
    for (target_id, method), selected in sorted(grouped.items()):
        output.append(
            {
                "target_id": target_id,
                "method": method,
                "fold_count": len(selected),
                "mean_holdout_robust_composite": statistics.fmean(
                    float(row["holdout_robust_composite"]) for row in selected
                ),
                "worst_holdout_robust_composite": min(
                    float(row["holdout_robust_composite"]) for row in selected
                ),
                "mean_holdout_primary": statistics.fmean(
                    float(row["holdout_primary"]) for row in selected
                ),
                "mean_holdout_worst_seed": statistics.fmean(
                    float(row["holdout_worst_seed"]) for row in selected
                ),
                "mean_holdout_rank": statistics.fmean(
                    int(row["holdout_rank"]) for row in selected
                ),
                "mean_subset_jaccard": mean_jaccard(
                    [tuple(str(row["selected_subset"]).split("+")) for row in selected]
                ),
                "selected_subsets": [row["selected_subset"] for row in selected],
            }
        )
    return output


def paired_comparison(
    rows: list[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    indexed = {
        (str(row["target_id"]), int(row["outer_fold"]), str(row["method"])): row
        for row in rows
    }
    target_ids = sorted({str(row["target_id"]) for row in rows})
    per_target: dict[str, dict[str, Any]] = {}
    all_deltas: list[float] = []
    for target_id in target_ids:
        folds = sorted(
            {
                int(row["outer_fold"])
                for row in rows
                if row["target_id"] == target_id
            }
        )
        deltas = [
            float(indexed[(target_id, fold, left)]["holdout_robust_composite"])
            - float(indexed[(target_id, fold, right)]["holdout_robust_composite"])
            for fold in folds
        ]
        all_deltas.extend(deltas)
        per_target[target_id] = {
            "fold_count": len(deltas),
            "mean_delta": statistics.fmean(deltas),
            "worst_delta": min(deltas),
            "positive_fold_count": sum(value > 0.0 for value in deltas),
            "nonnegative_fold_count": sum(value >= 0.0 for value in deltas),
            "fold_deltas": deltas,
        }
    return {
        "direction": f"{left} minus {right}",
        "fold_count": len(all_deltas),
        "mean_delta": statistics.fmean(all_deltas),
        "worst_delta": min(all_deltas),
        "positive_fold_count": sum(value > 0.0 for value in all_deltas),
        "per_target": per_target,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Stage 19e cross-target QUBO v2 diagnostic",
        "",
        "## Scope",
        "",
        "This is nested scaffold-CV development evidence on MK14 and PPARG train rows only.",
        "The inner folds choose ridge alpha; outer folds report performance.",
        "No fresh-validation or locked-test row was read.",
        "",
        "## Outer-fold results",
        "",
        "| Target | Method | Mean composite | Worst fold | Mean primary | Mean rank |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["outer_method_aggregate"]:
        lines.append(
            f"| {row['target_id']} | {row['method']} | "
            f"{float(row['mean_holdout_robust_composite']):.6f} | "
            f"{float(row['worst_holdout_robust_composite']):.6f} | "
            f"{float(row['mean_holdout_primary']):.6f} | "
            f"{float(row['mean_holdout_rank']):.2f} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for comparison in result["paired_comparisons"].values():
        lines.append(
            f"- {comparison['direction']}: mean "
            f"{float(comparison['mean_delta']):+.6f}, wins "
            f"{int(comparison['positive_fold_count'])}/{int(comparison['fold_count'])}."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Status: `{result['status']}`",
            f"- BACE1 amendment authorized: `{result['gate']['bace1_v2_amendment_authorized']}`",
            "",
            "## Boundary",
            "",
            str(result["interpretation_boundary"]),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 19e implementation path differs")
    for descriptor in config["prior_records"].values():
        verified(root, descriptor)

    outputs = {
        key: rooted(root, value)
        for key, value in config["outputs"].items()
    }
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage 19e outputs exist; pass --overwrite")

    diagnostic = config["diagnostic"]
    outer_count = int(diagnostic["outer_fold_count"])
    inner_count = int(diagnostic["inner_fold_count"])
    fold_seed = int(diagnostic["fold_seed"])
    inner_seed = int(diagnostic["inner_fold_seed"])
    target_size = int(diagnostic["target_size"])
    bedroc_alpha = float(diagnostic["bedroc_alpha"])
    ridge_alphas = [float(value) for value in diagnostic["ridge_alphas"]]

    inner_detail_rows: list[dict[str, Any]] = []
    outer_alpha_rows: list[dict[str, Any]] = []
    outer_method_rows: list[dict[str, Any]] = []
    fold_assignment_rows: list[dict[str, Any]] = []
    inner_assignment_rows: list[dict[str, Any]] = []
    target_full_evidence: dict[str, dict[str, Any]] = {}
    boundary_rows: dict[str, int] = {}

    for target_id, target_spec in config["targets"].items():
        print(f"loading_target={target_id}", flush=True)
        ligands, receptor_ids, matrices = load_target(root, target_id, target_spec)
        ligand_ids = sorted(row["ligand_id"] for row in ligands)
        all_ids = set(ligand_ids)
        assignments = make_frozen_group_folds(ligands, outer_count, fold_seed)
        fold_assignment_rows.extend(
            {
                "target_id": target_id,
                "ligand_id": row["ligand_id"],
                "label": row["label"],
                "split_group_id": row["split_group_id"],
                "scaffold_smiles": row["scaffold_smiles"],
                "outer_fold": assignments[row["ligand_id"]],
            }
            for row in ligands
        )
        subsets = sorted(
            tuple(sorted(subset))
            for subset in itertools.combinations(receptor_ids, target_size)
        )
        expected_subsets = math.comb(len(receptor_ids), target_size)
        if len(subsets) != expected_subsets:
            raise ValueError(f"{target_id} subset count differs")
        subset_index = {subset: index for index, subset in enumerate(subsets)}
        feature_sets = {
            family: design_matrix(
                subsets, receptor_ids, include_quadratic=(family == "quadratic")
            )
            for family in SURROGATE_FAMILIES
        }
        model_spec = {
            "coverage_fraction": float(target_spec["v1_qubo"]["coverage_fraction"]),
            "utility_metric": "bedroc",
        }
        target_outer_alpha_rows: list[dict[str, Any]] = []

        for outer_fold in range(outer_count):
            print(f"target={target_id} outer_fold={outer_fold}", flush=True)
            holdout_ids = {
                ligand_id
                for ligand_id, fold in assignments.items()
                if fold == outer_fold
            }
            train_ids = all_ids - holdout_ids
            outer_context = make_context(
                train_ids, holdout_ids, matrices, receptor_ids, model_spec
            )
            train_values = score_subsets(
                outer_context,
                subsets,
                receptor_ids,
                "train",
                bedroc_alpha,
            )
            holdout_values = score_subsets(
                outer_context,
                subsets,
                receptor_ids,
                "validation",
                bedroc_alpha,
            )

            outer_train_rows = [
                row for row in ligands if row["ligand_id"] in train_ids
            ]
            inner_assignments = make_frozen_group_folds(
                outer_train_rows,
                inner_count,
                inner_seed + outer_fold,
            )
            inner_assignment_rows.extend(
                {
                    "target_id": target_id,
                    "outer_fold": outer_fold,
                    "ligand_id": row["ligand_id"],
                    "label": row["label"],
                    "split_group_id": row["split_group_id"],
                    "scaffold_smiles": row["scaffold_smiles"],
                    "inner_fold": inner_assignments[row["ligand_id"]],
                }
                for row in outer_train_rows
            )
            inner_context_values: list[
                tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
            ] = []
            for inner_fold in range(inner_count):
                inner_validation = {
                    ligand_id
                    for ligand_id, fold in inner_assignments.items()
                    if fold == inner_fold
                }
                inner_train = train_ids - inner_validation
                context = make_context(
                    inner_train,
                    inner_validation,
                    matrices,
                    receptor_ids,
                    model_spec,
                )
                inner_context_values.append(
                    (
                        score_subsets(
                            context,
                            subsets,
                            receptor_ids,
                            "train",
                            bedroc_alpha,
                        ),
                        score_subsets(
                            context,
                            subsets,
                            receptor_ids,
                            "validation",
                            bedroc_alpha,
                        ),
                    )
                )

            fold_inner_rows: list[dict[str, Any]] = []
            for family in SURROGATE_FAMILIES:
                features, _ = feature_sets[family]
                for alpha in ridge_alphas:
                    for inner_fold, (
                        inner_train_values,
                        inner_validation_values,
                    ) in enumerate(inner_context_values):
                        model, predictions = fit_surrogate(
                            features,
                            inner_train_values["robust_composite"],
                            alpha,
                        )
                        selected_index = choose_highest(predictions, subsets)
                        row = {
                            "target_id": target_id,
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "model_family": family,
                            "alpha": alpha,
                            "selected_subset": "+".join(subsets[selected_index]),
                            "train_r2": float(
                                model.score(
                                    features,
                                    inner_train_values["robust_composite"],
                                )
                            ),
                            "train_rank_spearman": safe_spearman(
                                predictions,
                                inner_train_values["robust_composite"],
                            ),
                            "validation_rank_spearman": safe_spearman(
                                predictions,
                                inner_validation_values["robust_composite"],
                            ),
                            **prefixed(
                                "validation",
                                subset_metrics(
                                    inner_validation_values, selected_index
                                ),
                            ),
                        }
                        inner_detail_rows.append(row)
                        fold_inner_rows.append(row)

            inner_summary = summarize_alpha_trials(
                fold_inner_rows,
                ("target_id", "outer_fold", "model_family"),
            )
            for family in SURROGATE_FAMILIES:
                selected_alpha = float(
                    select_alpha(
                        inner_summary, target_id, family, outer_fold
                    )["alpha"]
                )
                features, _ = feature_sets[family]
                for alpha in ridge_alphas:
                    model, predictions = fit_surrogate(
                        features, train_values["robust_composite"], alpha
                    )
                    selected_index = choose_highest(predictions, subsets)
                    row = {
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "model_family": family,
                        "alpha": alpha,
                        "selected_by_inner_cv": alpha == selected_alpha,
                        "selected_subset": "+".join(subsets[selected_index]),
                        "train_r2": float(
                            model.score(features, train_values["robust_composite"])
                        ),
                        "train_rank_spearman": safe_spearman(
                            predictions, train_values["robust_composite"]
                        ),
                        "validation_rank_spearman": safe_spearman(
                            predictions, holdout_values["robust_composite"]
                        ),
                        **prefixed(
                            "validation",
                            subset_metrics(holdout_values, selected_index),
                        ),
                    }
                    outer_alpha_rows.append(row)
                    target_outer_alpha_rows.append(row)
                    if alpha == selected_alpha:
                        outer_method_rows.append(
                            method_row(
                                target_id,
                                outer_fold,
                                f"{family}_nested",
                                subsets[selected_index],
                                subset_index,
                                train_values,
                                holdout_values,
                                selected_alpha=selected_alpha,
                                train_fit_r2=float(
                                    model.score(
                                        features,
                                        train_values["robust_composite"],
                                    )
                                ),
                                holdout_rank_spearman=safe_spearman(
                                    predictions,
                                    holdout_values["robust_composite"],
                                ),
                            )
                        )

            v1_subset, v1_evidence = select_v1(
                outer_context, receptor_ids, target_spec
            )
            v1_energies = np.asarray(
                [
                    coefficient_energy(
                        subset_value, v1_evidence["coefficients"]
                    )
                    for subset_value in subsets
                ],
                dtype=float,
            )
            direct_greedy = tuple(
                sorted(
                    choose_greedy(
                        outer_context,
                        receptor_ids,
                        target_size,
                        str(target_spec["v1_qubo"]["aggregation"]),
                    )
                )
            )
            direct_exact = tuple(
                sorted(
                    choose_exhaustive(
                        outer_context,
                        receptor_ids,
                        target_size,
                        str(target_spec["v1_qubo"]["aggregation"]),
                    )
                )
            )
            composite_exact = subsets[
                choose_highest(train_values["robust_composite"], subsets)
            ]
            oracle = subsets[
                choose_highest(holdout_values["robust_composite"], subsets)
            ]
            for method, subset, extra in (
                (
                    "v1_qubo_exact",
                    v1_subset,
                    {
                        "v1_energy": float(v1_evidence["energy"]),
                        "train_holdout_rank_spearman": safe_spearman(
                            train_values["robust_composite"],
                            holdout_values["robust_composite"],
                        ),
                        "v1_train_rank_spearman": safe_spearman(
                            -v1_energies,
                            train_values["robust_composite"],
                        ),
                        "v1_holdout_rank_spearman": safe_spearman(
                            -v1_energies,
                            holdout_values["robust_composite"],
                        ),
                    },
                ),
                ("direct_greedy", direct_greedy, {}),
                ("direct_exact", direct_exact, {}),
                ("composite_exact", composite_exact, {}),
                ("holdout_oracle", oracle, {"diagnostic_only": True}),
            ):
                outer_method_rows.append(
                    method_row(
                        target_id,
                        outer_fold,
                        method,
                        subset,
                        subset_index,
                        train_values,
                        holdout_values,
                        **extra,
                    )
                )

        full_context = make_context(
            all_ids, set(), matrices, receptor_ids, model_spec
        )
        full_values = score_subsets(
            full_context, subsets, receptor_ids, "train", bedroc_alpha
        )
        outer_summary = summarize_alpha_trials(
            target_outer_alpha_rows,
            ("target_id", "model_family"),
        )
        target_models: dict[str, Any] = {}
        for family in SURROGATE_FAMILIES:
            selected = select_alpha(outer_summary, target_id, family, None)
            alpha = float(selected["alpha"])
            features, feature_names = feature_sets[family]
            model, predictions = fit_surrogate(
                features, full_values["robust_composite"], alpha
            )
            selected_index = choose_highest(predictions, subsets)
            evidence: dict[str, Any] = {
                "selected_alpha": alpha,
                "selected_subset": list(subsets[selected_index]),
                "full_train_metrics": subset_metrics(full_values, selected_index),
                "full_train_r2": float(
                    model.score(features, full_values["robust_composite"])
                ),
                "full_train_rank_spearman": safe_spearman(
                    predictions, full_values["robust_composite"]
                ),
                "alpha_selection": selected,
            }
            if family == "quadratic":
                qubo = build_explicit_qubo(
                    model,
                    feature_names,
                    receptor_ids,
                    target_size,
                    float(diagnostic["cardinality_penalty"]),
                )
                exact_subset, exact_energy = exact_all_cardinalities(
                    receptor_ids, qubo
                )
                if exact_subset != subsets[selected_index]:
                    raise ValueError(
                        f"{target_id} penalized QUBO changed the selected subset"
                    )
                pair_values = [
                    float(value) for value in qubo["surrogate"]["quadratic"].values()
                ]
                evidence.update(
                    {
                        "explicit_qubo": qubo,
                        "exact_all_cardinalities_subset": list(exact_subset),
                        "exact_all_cardinalities_energy": exact_energy,
                        "quadratic_term_count": len(pair_values),
                        "maximum_absolute_surrogate_pair_term": max(
                            abs(value) for value in pair_values
                        ),
                    }
                )
            target_models[family] = evidence
        target_full_evidence[target_id] = {
            "ligand_count": len(ligands),
            "receptor_count": len(receptor_ids),
            "subset_count": len(subsets),
            "models": target_models,
        }
        boundary_rows[target_id] = len(ligands)

    outer_aggregate = aggregate_outer_methods(outer_method_rows)
    comparisons = {
        "quadratic_vs_direct_greedy": paired_comparison(
            outer_method_rows, "quadratic_nested", "direct_greedy"
        ),
        "quadratic_vs_additive": paired_comparison(
            outer_method_rows, "quadratic_nested", "additive_nested"
        ),
        "quadratic_vs_v1": paired_comparison(
            outer_method_rows, "quadratic_nested", "v1_qubo_exact"
        ),
    }
    v1_rows_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outer_method_rows:
        if row["method"] == "v1_qubo_exact":
            v1_rows_by_target[str(row["target_id"])].append(row)
    objective_diagnosis = {
        target_id: {
            "fold_count": len(rows),
            "mean_train_to_holdout_subset_rank_spearman": statistics.fmean(
                float(row["train_holdout_rank_spearman"]) for row in rows
            ),
            "mean_v1_energy_vs_train_utility_rank_spearman": statistics.fmean(
                float(row["v1_train_rank_spearman"]) for row in rows
            ),
            "mean_v1_energy_vs_holdout_utility_rank_spearman": statistics.fmean(
                float(row["v1_holdout_rank_spearman"]) for row in rows
            ),
            "mean_v1_holdout_rank_of_560": statistics.fmean(
                int(row["holdout_rank"]) for row in rows
            ),
        }
        for target_id, rows in sorted(v1_rows_by_target.items())
    }
    gate_spec = config["development_support_gate"]
    direct = comparisons["quadratic_vs_direct_greedy"]
    additive = comparisons["quadratic_vs_additive"]
    v1 = comparisons["quadratic_vs_v1"]
    per_target_direct_positive = all(
        float(value["mean_delta"]) > float(gate_spec["minimum_target_mean_delta"])
        for value in direct["per_target"].values()
    )
    per_target_additive_positive = all(
        float(value["mean_delta"]) > float(gate_spec["minimum_target_mean_delta"])
        for value in additive["per_target"].values()
    )
    per_target_v1_positive = all(
        float(value["mean_delta"]) > float(gate_spec["minimum_target_mean_delta"])
        for value in v1["per_target"].values()
    )
    gate_passed = (
        per_target_direct_positive
        and per_target_additive_positive
        and per_target_v1_positive
        and int(direct["positive_fold_count"])
        >= int(gate_spec["minimum_positive_folds_of_eight"])
        and int(additive["positive_fold_count"])
        >= int(gate_spec["minimum_positive_folds_of_eight"])
        and int(v1["positive_fold_count"])
        >= int(gate_spec["minimum_positive_folds_of_eight"])
    )
    status = (
        "stage19e_quadratic_v2_supported_for_prospective_bace1_secondary"
        if gate_passed
        else "stage19e_quadratic_v2_not_supported_do_not_amend_bace1"
    )

    write_csv(outputs["fold_assignments_csv"], fold_assignment_rows)
    write_csv(outputs["inner_fold_assignments_csv"], inner_assignment_rows)
    write_csv(outputs["inner_trials_csv"], inner_detail_rows)
    write_csv(outputs["outer_alpha_trials_csv"], outer_alpha_rows)
    write_csv(outputs["outer_method_results_csv"], outer_method_rows)
    algorithm_record = {
        "schema_version": "1.0",
        "algorithm_id": "regularized-quadratic-subset-surrogate-v2-nested-cv",
        "status": (
            "development_gate_passed_prospective_secondary_only"
            if gate_passed
            else "development_gate_failed_not_authorized_for_bace1"
        ),
        "algorithm": {
            "target_size": target_size,
            "aggregation": "normalized min_score",
            "response": diagnostic["surrogate_response"],
            "feature_map": "one linear bit per receptor and one pair bit per receptor pair",
            "ridge_alpha_grid": ridge_alphas,
            "alpha_selection": diagnostic["nested_selection_rule"],
            "cardinality_penalty": float(diagnostic["cardinality_penalty"]),
        },
        "target_development_fits": target_full_evidence,
        "prospective_use": (
            "May be added as a frozen secondary method before BACE1 docking."
            if gate_passed
            else "Must not be added to BACE1; redesign requires a new development stage."
        ),
        "data_boundary": {
            "train_rows_read_by_target": boundary_rows,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "bace1_docking_rows_read": 0,
        },
    }
    write_json(outputs["algorithm_record_json"], algorithm_record)

    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "experiment_class": "posthoc_cross_target_train_only_development",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "input_dimensions": {
            target_id: {
                key: value
                for key, value in evidence.items()
                if key != "models"
            }
            for target_id, evidence in target_full_evidence.items()
        },
        "outer_method_aggregate": outer_aggregate,
        "objective_diagnosis": objective_diagnosis,
        "paired_comparisons": comparisons,
        "gate": {
            "rule": gate_spec,
            "per_target_quadratic_vs_direct_positive": per_target_direct_positive,
            "per_target_quadratic_vs_additive_positive": per_target_additive_positive,
            "per_target_quadratic_vs_v1_positive": per_target_v1_positive,
            "passed": gate_passed,
            "bace1_v2_amendment_authorized": gate_passed,
        },
        "target_full_train_models": {
            target_id: {
                family: {
                    key: value
                    for key, value in evidence["models"][family].items()
                    if key != "explicit_qubo"
                }
                for family in SURROGATE_FAMILIES
            }
            for target_id, evidence in target_full_evidence.items()
        },
        "data_boundary": {
            "train_rows_read_by_target": boundary_rows,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "bace1_docking_rows_read": 0,
        },
        "outputs": {
            key: output_descriptor(root, path)
            for key, path in outputs.items()
            if key not in {"result_json", "report_md"}
        },
        "next_gate": (
            "freeze a BACE1 preregistration amendment before any BACE1 docking"
            if gate_passed
            else "retain BACE1 v1 preregistration and redesign the objective without BACE1 outcomes"
        ),
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
