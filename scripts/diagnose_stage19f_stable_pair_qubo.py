"""Evaluate a scaffold-stable, risk-controlled pair-synergy QUBO."""

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

from scripts.diagnose_stage12a_mk14_qubo_objective_adequacy import (
    exact_all_cardinalities,
    mean_jaccard,
)
from scripts.diagnose_stage19e_cross_target_qubo_v2 import (
    descending_rank,
    load_target,
    output_descriptor,
    read_csv,
    read_json,
    rooted,
    safe_spearman,
    score_subsets,
    subset_metrics,
    verified,
    write_csv,
    write_json,
)
from scripts.normalized_receptor_qubo import coefficient_energy
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage05_mk14_uncertainty_qubo_gate import make_context
from scripts.screen_stage10_mk14_expanded16_qubo_greedy import (
    fixed_cardinality_exact,
)


BASELINE_METHODS = ("direct_greedy", "additive_nested", "v1_qubo_exact")
CANDIDATE_METHOD = "stable_pair_nested"
LINEAR_METHOD = "stable_singleton_linear"


def context_block_view(
    context: dict[str, Any], split: str, ligand_ids: set[str]
) -> dict[str, Any]:
    return {
        "matrices": {
            matrix_id: {
                "train": [
                    row
                    for row in context["matrices"][matrix_id][split]
                    if row["ligand_id"] in ligand_ids
                ]
            }
            for matrix_id in ("primary", "sensitivity", "seed0", "seed1", "seed2")
        }
    }


def minmax(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-15):
        return np.zeros_like(values, dtype=float)
    return np.asarray((values - minimum) / (maximum - minimum), dtype=float)


def stable_statistics(
    context: dict[str, Any],
    receptor_ids: list[str],
    block_by_ligand: dict[str, int],
    bedroc_alpha: float,
) -> dict[str, Any]:
    singletons = [(receptor_id,) for receptor_id in receptor_ids]
    pairs = list(itertools.combinations(receptor_ids, 2))
    full_single = score_subsets(
        context, singletons, receptor_ids, "train", bedroc_alpha
    )["robust_composite"]
    blocks = sorted(set(block_by_ligand.values()))
    if len(blocks) < 2:
        raise ValueError("stable-pair fitting requires at least two scaffold blocks")
    block_synergies: list[np.ndarray] = []
    block_singletons: list[np.ndarray] = []
    receptor_index = {
        receptor_id: index for index, receptor_id in enumerate(receptor_ids)
    }
    for block in blocks:
        ligand_ids = {
            ligand_id
            for ligand_id, assignment in block_by_ligand.items()
            if assignment == block
        }
        view = context_block_view(context, "train", ligand_ids)
        single_values = score_subsets(
            view, singletons, receptor_ids, "train", bedroc_alpha
        )["robust_composite"]
        pair_values = score_subsets(
            view, pairs, receptor_ids, "train", bedroc_alpha
        )["robust_composite"]
        baseline = np.asarray(
            [
                max(single_values[receptor_index[first]], single_values[receptor_index[second]])
                for first, second in pairs
            ],
            dtype=float,
        )
        block_singletons.append(single_values)
        block_synergies.append(pair_values - baseline)
    synergy_matrix = np.vstack(block_synergies)
    return {
        "receptor_ids": receptor_ids,
        "singletons": singletons,
        "pairs": pairs,
        "block_ids": blocks,
        "linear_raw": full_single,
        "linear_normalized": minmax(full_single),
        "block_singletons": np.vstack(block_singletons),
        "pair_synergy_mean": np.mean(synergy_matrix, axis=0),
        "pair_synergy_std": np.std(synergy_matrix, axis=0, ddof=0),
        "pair_positive_fraction": np.mean(synergy_matrix > 0.0, axis=0),
        "pair_synergy_by_block": synergy_matrix,
    }


def stable_pair_coefficients(
    stats: dict[str, Any],
    pair_weight: float,
    risk_kappa: float,
    target_size: int,
    cardinality_penalty: float,
    minimum_positive_block_fraction: float,
    minimum_pair_lcb: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    means = np.asarray(stats["pair_synergy_mean"], dtype=float)
    standard_deviations = np.asarray(stats["pair_synergy_std"], dtype=float)
    positive_fraction = np.asarray(stats["pair_positive_fraction"], dtype=float)
    lcb = means - risk_kappa * standard_deviations
    retained = (
        (positive_fraction >= minimum_positive_block_fraction)
        & (lcb >= minimum_pair_lcb)
    )
    stable_raw = np.where(retained, lcb, 0.0)
    maximum = float(np.max(stable_raw))
    stable_normalized = (
        stable_raw / maximum if maximum > 0.0 else np.zeros_like(stable_raw)
    )
    receptor_ids = list(stats["receptor_ids"])
    pairs = list(stats["pairs"])
    linear_utility = np.asarray(stats["linear_normalized"], dtype=float)
    coefficients = {
        "convention": (
            "Q(x)=constant+sum_i linear[i]*x_i+"
            "sum_i<j quadratic[i__j]*x_i*x_j"
        ),
        "constant": cardinality_penalty * target_size**2,
        "linear": {
            receptor_id: -float(linear_utility[index])
            + cardinality_penalty * (1 - 2 * target_size)
            for index, receptor_id in enumerate(receptor_ids)
        },
        "quadratic": {
            f"{first}__{second}": -pair_weight * float(stable_normalized[index])
            + 2.0 * cardinality_penalty
            for index, (first, second) in enumerate(pairs)
        },
        "target_size": target_size,
        "cardinality_penalty": cardinality_penalty,
        "utility": {
            "linear_raw": {
                receptor_id: float(stats["linear_raw"][index])
                for index, receptor_id in enumerate(receptor_ids)
            },
            "linear_normalized": {
                receptor_id: float(linear_utility[index])
                for index, receptor_id in enumerate(receptor_ids)
            },
            "pair_weight": pair_weight,
            "risk_kappa": risk_kappa,
            "minimum_positive_block_fraction": minimum_positive_block_fraction,
            "minimum_pair_lcb": minimum_pair_lcb,
            "pair_stable_raw": {
                f"{first}__{second}": float(stable_raw[index])
                for index, (first, second) in enumerate(pairs)
            },
            "pair_stable_normalized": {
                f"{first}__{second}": float(stable_normalized[index])
                for index, (first, second) in enumerate(pairs)
            },
        },
    }
    diagnostics = {
        "block_count": len(stats["block_ids"]),
        "retained_pair_count": int(retained.sum()),
        "maximum_pair_lcb": maximum,
        "mean_pair_synergy": float(np.mean(means)),
        "maximum_pair_synergy": float(np.max(means)),
        "pair_weight": pair_weight,
        "risk_kappa": risk_kappa,
    }
    return coefficients, diagnostics


def triple_grid(receptor_ids: list[str]) -> list[tuple[str, ...]]:
    return sorted(
        tuple(sorted(subset))
        for subset in itertools.combinations(receptor_ids, 3)
    )


def select_from_coefficients(
    coefficients: dict[str, Any], receptor_ids: list[str], target_size: int
) -> tuple[tuple[str, ...], float, np.ndarray, list[tuple[str, ...]]]:
    subset, energy = fixed_cardinality_exact(
        coefficients, receptor_ids, target_size
    )
    triples = triple_grid(receptor_ids)
    energies = np.asarray(
        [coefficient_energy(value, coefficients) for value in triples],
        dtype=float,
    )
    return tuple(sorted(subset)), float(energy), energies, triples


def selected_pair_count(subset: tuple[str, ...], coefficients: dict[str, Any]) -> int:
    values = coefficients["utility"]["pair_stable_normalized"]
    count = 0
    for first, second in itertools.combinations(subset, 2):
        key = f"{first}__{second}"
        if key not in values:
            key = f"{second}__{first}"
        count += float(values[key]) > 0.0
    return count


def candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["mean_validation_robust_composite"]),
        -float(row["worst_validation_robust_composite"]),
        -float(row["mean_validation_rank_spearman"]),
        float(row["pair_weight"]),
        -float(row["risk_kappa"]),
    )


def summarize_candidates(
    rows: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields) + (
            float(row["pair_weight"]),
            float(row["risk_kappa"]),
        )
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for key, selected in sorted(grouped.items(), key=lambda item: item[0]):
        fields = (*group_fields, "pair_weight", "risk_kappa")
        subsets = [tuple(row["selected_subset"].split("+")) for row in selected]
        output.append(
            {
                **dict(zip(fields, key)),
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
                "mean_retained_pair_count": statistics.fmean(
                    int(row["retained_pair_count"]) for row in selected
                ),
                "folds_with_selected_nonzero_pair": sum(
                    int(row["selected_subset_pair_count"]) > 0 for row in selected
                ),
                "mean_subset_jaccard": mean_jaccard(subsets),
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
    per_target: dict[str, dict[str, Any]] = {}
    all_deltas: list[float] = []
    for target_id in sorted({str(row["target_id"]) for row in rows}):
        folds = sorted(
            int(row["outer_fold"])
            for row in rows
            if row["target_id"] == target_id and row["method"] == left
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


def aggregate_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                    [tuple(row["selected_subset"].split("+")) for row in selected]
                ),
            }
        )
    return output


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Stage 19f scaffold-stable pair QUBO",
        "",
        "## Scope",
        "",
        "This is nested scaffold-CV development evidence on MK14 and PPARG train rows only.",
        "Pair terms survive only when their cross-block lower confidence bound is positive.",
        "No BACE1 docking, fresh-validation, or locked-test row was read.",
        "",
        "## Results",
        "",
        "| Target | Method | Mean composite | Worst fold | Mean primary |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result["method_aggregate"]:
        lines.append(
            f"| {row['target_id']} | {row['method']} | "
            f"{float(row['mean_holdout_robust_composite']):.6f} | "
            f"{float(row['worst_holdout_robust_composite']):.6f} | "
            f"{float(row['mean_holdout_primary']):.6f} |"
        )
    lines.extend(["", "## Comparisons", ""])
    for comparison in result["paired_comparisons"].values():
        lines.append(
            f"- {comparison['direction']}: mean "
            f"{float(comparison['mean_delta']):+.6f}, wins "
            f"{comparison['positive_fold_count']}/{comparison['fold_count']}."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Status: `{result['status']}`",
            f"- BACE1 amendment authorized: `{result['gate']['bace1_amendment_authorized']}`",
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
        raise ValueError("Stage 19f implementation path differs")
    source_config_path = verified(root, config["inputs"]["stage19e_config"])
    source_result_path = verified(root, config["inputs"]["stage19e_result"])
    source_audit_path = verified(root, config["inputs"]["stage19e_audit"])
    source_methods_path = verified(root, config["inputs"]["stage19e_outer_methods"])
    source_config = read_json(source_config_path)
    source_result = read_json(source_result_path)
    source_audit = read_json(source_audit_path)
    if source_result["status"] != "stage19e_quadratic_v2_not_supported_do_not_amend_bace1":
        raise ValueError("Stage 19e source decision differs")
    if source_audit["status"] != "stage19e_cross_target_qubo_v2_nested_diagnostic_audit_ok":
        raise ValueError("Stage 19e source audit differs")
    if source_result["outputs"]["outer_method_results_csv"]["sha256"] != file_sha256(
        source_methods_path
    ):
        raise ValueError("Stage 19e outer method identity differs")

    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage 19f outputs exist; pass --overwrite")

    diagnostic = config["diagnostic"]
    outer_count = int(diagnostic["outer_fold_count"])
    inner_count = int(diagnostic["inner_fold_count"])
    fold_seed = int(diagnostic["fold_seed"])
    inner_seed = int(diagnostic["inner_fold_seed"])
    bedroc_alpha = float(diagnostic["bedroc_alpha"])
    target_size = int(diagnostic["target_size"])
    pair_weights = [float(value) for value in diagnostic["pair_weights"]]
    risk_kappas = [float(value) for value in diagnostic["risk_kappas"]]
    candidates = list(itertools.product(pair_weights, risk_kappas))
    penalty = float(diagnostic["cardinality_penalty"])
    minimum_fraction = float(diagnostic["minimum_positive_block_fraction"])
    minimum_lcb = float(diagnostic["minimum_pair_lcb"])

    source_method_rows = read_csv(source_methods_path)
    baseline_rows = [
        row for row in source_method_rows if row["method"] in BASELINE_METHODS
    ]
    inner_rows: list[dict[str, Any]] = []
    outer_candidate_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    full_models: dict[str, Any] = {}

    for target_id, target_spec in source_config["targets"].items():
        print(f"loading_target={target_id}", flush=True)
        ligands, receptor_ids, matrices = load_target(root, target_id, target_spec)
        assignments = make_frozen_group_folds(ligands, outer_count, fold_seed)
        all_ids = {row["ligand_id"] for row in ligands}
        triples = triple_grid(receptor_ids)
        triple_index = {subset: index for index, subset in enumerate(triples)}
        model_spec = {
            "coverage_fraction": float(target_spec["v1_qubo"]["coverage_fraction"]),
            "utility_metric": "bedroc",
        }
        target_outer_candidates: list[dict[str, Any]] = []

        for outer_fold in range(outer_count):
            print(f"target={target_id} outer_fold={outer_fold}", flush=True)
            holdout_ids = {
                ligand_id
                for ligand_id, fold in assignments.items()
                if fold == outer_fold
            }
            train_ids = all_ids - holdout_ids
            outer_train_rows = [
                row for row in ligands if row["ligand_id"] in train_ids
            ]
            inner_assignments = make_frozen_group_folds(
                outer_train_rows, inner_count, inner_seed + outer_fold
            )
            outer_context = make_context(
                train_ids, holdout_ids, matrices, receptor_ids, model_spec
            )
            outer_train_values = score_subsets(
                outer_context, triples, receptor_ids, "train", bedroc_alpha
            )
            outer_holdout_values = score_subsets(
                outer_context, triples, receptor_ids, "validation", bedroc_alpha
            )

            fold_inner_rows: list[dict[str, Any]] = []
            for inner_fold in range(inner_count):
                inner_validation = {
                    ligand_id
                    for ligand_id, fold in inner_assignments.items()
                    if fold == inner_fold
                }
                inner_train = train_ids - inner_validation
                context = make_context(
                    inner_train, inner_validation, matrices, receptor_ids, model_spec
                )
                block_map = {
                    ligand_id: fold
                    for ligand_id, fold in inner_assignments.items()
                    if ligand_id in inner_train
                }
                stats = stable_statistics(
                    context, receptor_ids, block_map, bedroc_alpha
                )
                validation_values = score_subsets(
                    context, triples, receptor_ids, "validation", bedroc_alpha
                )
                for pair_weight, risk_kappa in candidates:
                    coefficients, evidence = stable_pair_coefficients(
                        stats,
                        pair_weight,
                        risk_kappa,
                        target_size,
                        penalty,
                        minimum_fraction,
                        minimum_lcb,
                    )
                    subset, _, energies, _ = select_from_coefficients(
                        coefficients, receptor_ids, target_size
                    )
                    index = triple_index[subset]
                    row = {
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "pair_weight": pair_weight,
                        "risk_kappa": risk_kappa,
                        "selected_subset": "+".join(subset),
                        "retained_pair_count": evidence["retained_pair_count"],
                        "selected_subset_pair_count": selected_pair_count(
                            subset, coefficients
                        ),
                        "validation_rank_spearman": safe_spearman(
                            -energies, validation_values["robust_composite"]
                        ),
                        **{
                            f"validation_{key}": value
                            for key, value in subset_metrics(
                                validation_values, index
                            ).items()
                        },
                    }
                    inner_rows.append(row)
                    fold_inner_rows.append(row)
            inner_summary = summarize_candidates(
                fold_inner_rows,
                ("target_id", "outer_fold"),
            )
            selected_candidate = min(inner_summary, key=candidate_key)
            selected_pair_weight = float(selected_candidate["pair_weight"])
            selected_risk_kappa = float(selected_candidate["risk_kappa"])

            stats = stable_statistics(
                outer_context, receptor_ids, inner_assignments, bedroc_alpha
            )
            linear_coefficients, linear_evidence = stable_pair_coefficients(
                stats,
                0.0,
                0.0,
                target_size,
                penalty,
                minimum_fraction,
                minimum_lcb,
            )
            linear_subset, _, linear_energies, _ = select_from_coefficients(
                linear_coefficients, receptor_ids, target_size
            )
            linear_index = triple_index[linear_subset]
            method_rows.append(
                {
                    "target_id": target_id,
                    "outer_fold": outer_fold,
                    "method": LINEAR_METHOD,
                    "selected_subset": "+".join(linear_subset),
                    "holdout_rank": descending_rank(
                        outer_holdout_values["robust_composite"], linear_index
                    ),
                    **{
                        f"train_{key}": value
                        for key, value in subset_metrics(
                            outer_train_values, linear_index
                        ).items()
                    },
                    **{
                        f"holdout_{key}": value
                        for key, value in subset_metrics(
                            outer_holdout_values, linear_index
                        ).items()
                    },
                    "retained_pair_count": linear_evidence["retained_pair_count"],
                    "selected_subset_pair_count": 0,
                    "pair_weight": 0.0,
                    "risk_kappa": 0.0,
                }
            )

            for pair_weight, risk_kappa in candidates:
                coefficients, evidence = stable_pair_coefficients(
                    stats,
                    pair_weight,
                    risk_kappa,
                    target_size,
                    penalty,
                    minimum_fraction,
                    minimum_lcb,
                )
                subset, _, energies, _ = select_from_coefficients(
                    coefficients, receptor_ids, target_size
                )
                index = triple_index[subset]
                row = {
                    "target_id": target_id,
                    "outer_fold": outer_fold,
                    "pair_weight": pair_weight,
                    "risk_kappa": risk_kappa,
                    "selected_by_inner_cv": (
                        pair_weight == selected_pair_weight
                        and risk_kappa == selected_risk_kappa
                    ),
                    "selected_subset": "+".join(subset),
                    "retained_pair_count": evidence["retained_pair_count"],
                    "selected_subset_pair_count": selected_pair_count(
                        subset, coefficients
                    ),
                    "validation_rank_spearman": safe_spearman(
                        -energies, outer_holdout_values["robust_composite"]
                    ),
                    **{
                        f"validation_{key}": value
                        for key, value in subset_metrics(
                            outer_holdout_values, index
                        ).items()
                    },
                }
                outer_candidate_rows.append(row)
                target_outer_candidates.append(row)
                if row["selected_by_inner_cv"]:
                    method_rows.append(
                        {
                            "target_id": target_id,
                            "outer_fold": outer_fold,
                            "method": CANDIDATE_METHOD,
                            "selected_subset": "+".join(subset),
                            "holdout_rank": descending_rank(
                                outer_holdout_values["robust_composite"], index
                            ),
                            **{
                                f"train_{key}": value
                                for key, value in subset_metrics(
                                    outer_train_values, index
                                ).items()
                            },
                            **{
                                f"holdout_{key}": value
                                for key, value in subset_metrics(
                                    outer_holdout_values, index
                                ).items()
                            },
                            "retained_pair_count": evidence["retained_pair_count"],
                            "selected_subset_pair_count": row[
                                "selected_subset_pair_count"
                            ],
                            "pair_weight": pair_weight,
                            "risk_kappa": risk_kappa,
                            "holdout_rank_spearman": row[
                                "validation_rank_spearman"
                            ],
                        }
                    )

        outer_summary = summarize_candidates(
            target_outer_candidates, ("target_id",)
        )
        full_candidate = min(outer_summary, key=candidate_key)
        full_context = make_context(all_ids, set(), matrices, receptor_ids, model_spec)
        stats = stable_statistics(full_context, receptor_ids, assignments, bedroc_alpha)
        coefficients, evidence = stable_pair_coefficients(
            stats,
            float(full_candidate["pair_weight"]),
            float(full_candidate["risk_kappa"]),
            target_size,
            penalty,
            minimum_fraction,
            minimum_lcb,
        )
        subset, energy, energies, _ = select_from_coefficients(
            coefficients, receptor_ids, target_size
        )
        exact_subset, exact_energy = exact_all_cardinalities(
            receptor_ids, coefficients
        )
        if exact_subset != subset:
            raise ValueError(f"{target_id} cardinality penalty changed the optimum")
        full_values = score_subsets(
            full_context, triples, receptor_ids, "train", bedroc_alpha
        )
        index = triple_index[subset]
        full_models[target_id] = {
            "selected_pair_weight": float(full_candidate["pair_weight"]),
            "selected_risk_kappa": float(full_candidate["risk_kappa"]),
            "selected_subset": list(subset),
            "selected_subset_pair_count": selected_pair_count(subset, coefficients),
            "retained_pair_count": evidence["retained_pair_count"],
            "full_train_metrics": subset_metrics(full_values, index),
            "full_train_rank": descending_rank(
                full_values["robust_composite"], index
            ),
            "objective_vs_train_rank_spearman": safe_spearman(
                -energies, full_values["robust_composite"]
            ),
            "fixed_cardinality_energy": energy,
            "exact_all_cardinalities_subset": list(exact_subset),
            "exact_all_cardinalities_energy": exact_energy,
            "candidate_selection": full_candidate,
            "coefficients": coefficients,
        }

    comparison_rows: list[dict[str, Any]] = [*method_rows]
    comparison_rows.extend(
        {
            key: (
                None
                if value == ""
                else
                int(value)
                if key == "outer_fold"
                else float(value)
                if key.startswith(("train_", "holdout_")) and key != "holdout_rank"
                else int(value)
                if key == "holdout_rank"
                else value
            )
            for key, value in row.items()
        }
        for row in baseline_rows
    )
    comparisons = {
        f"stable_pair_vs_{method}": paired_comparison(
            comparison_rows, CANDIDATE_METHOD, method
        )
        for method in (*BASELINE_METHODS, LINEAR_METHOD)
    }
    method_aggregate = aggregate_methods(comparison_rows)
    gate_spec = config["development_support_gate"]
    comparison_checks = {
        key: (
            all(
                float(value["mean_delta"])
                > float(gate_spec["minimum_target_mean_delta"])
                for value in comparison["per_target"].values()
            )
            and int(comparison["positive_fold_count"])
            >= int(gate_spec["minimum_positive_folds_of_eight"])
        )
        for key, comparison in comparisons.items()
    }
    folds_with_selected_pair = sum(
        int(row["selected_subset_pair_count"]) > 0
        for row in method_rows
        if row["method"] == CANDIDATE_METHOD
    )
    meaningful_pair_check = folds_with_selected_pair >= int(
        gate_spec["minimum_folds_with_selected_nonzero_pair"]
    )
    gate_passed = all(comparison_checks.values()) and meaningful_pair_check
    status = (
        "stage19f_stable_pair_qubo_supported_for_prospective_bace1_secondary"
        if gate_passed
        else "stage19f_stable_pair_qubo_not_supported_do_not_amend_bace1"
    )

    write_csv(outputs["inner_candidate_trials_csv"], inner_rows)
    write_csv(outputs["outer_candidate_trials_csv"], outer_candidate_rows)
    write_csv(outputs["comparison_method_results_csv"], comparison_rows)
    model_record = {
        "schema_version": "1.0",
        "algorithm_id": "scaffold-stable-risk-controlled-pair-qubo-v1",
        "status": (
            "development_gate_passed_prospective_secondary_only"
            if gate_passed
            else "development_gate_failed_not_authorized_for_bace1"
        ),
        "algorithm": {
            "linear_term": "min-max normalized full-train singleton robust composite",
            "raw_pair_synergy": "pair robust composite minus the better singleton robust composite within each scaffold block",
            "stable_pair_term": "max(mean synergy - risk_kappa * population_std, 0) after sign-consistency and minimum-LCB gates",
            "pair_weights": pair_weights,
            "risk_kappas": risk_kappas,
            "minimum_positive_block_fraction": minimum_fraction,
            "minimum_pair_lcb": minimum_lcb,
            "target_size": target_size,
            "cardinality_penalty": penalty,
        },
        "target_development_models": full_models,
        "prospective_use": (
            "May be added as a frozen BACE1 secondary method before docking."
            if gate_passed
            else "Must not be added to BACE1."
        ),
        "data_boundary": {
            "train_rows_read_by_target": {"MK14": 696, "PPARG": 668},
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "bace1_docking_rows_read": 0,
        },
    }
    write_json(outputs["model_record_json"], model_record)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "experiment_class": "posthoc_cross_target_train_only_development",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "method_aggregate": method_aggregate,
        "paired_comparisons": comparisons,
        "gate": {
            "rule": gate_spec,
            "comparison_checks": comparison_checks,
            "folds_with_selected_nonzero_pair": folds_with_selected_pair,
            "meaningful_pair_check": meaningful_pair_check,
            "passed": gate_passed,
            "bace1_amendment_authorized": gate_passed,
        },
        "full_train_models": {
            target_id: {
                key: value
                for key, value in evidence.items()
                if key != "coefficients"
            }
            for target_id, evidence in full_models.items()
        },
        "data_boundary": {
            "train_rows_read_by_target": {"MK14": 696, "PPARG": 668},
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
            "freeze BACE1 secondary-method amendment before any BACE1 docking"
            if gate_passed
            else "stop target-specific QUBO retuning and review the quantum-application claim"
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
