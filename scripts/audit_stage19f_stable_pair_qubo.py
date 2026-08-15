"""Independently audit the Stage 19f stable-pair QUBO diagnostic."""

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
    subset_bedroc,
)
from scripts.diagnose_stage19e_cross_target_qubo_v2 import (
    load_target,
    safe_spearman,
    score_subsets,
)
from scripts.normalized_receptor_qubo import coefficient_energy
from scripts.prepare_receptor import file_sha256
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage05_mk14_uncertainty_qubo_gate import make_context


DEFAULT_CONFIG = Path("configs/stage19f_cross_target_stable_pair_qubo.json")
DEFAULT_OUTPUT = Path("data/stage19f_cross_target_stable_pair_qubo_audit.json")
MATRIX_IDS = ("primary", "sensitivity", "seed0", "seed1", "seed2")
SEED_IDS = ("seed0", "seed1", "seed2")
CANDIDATE_METHOD = "stable_pair_nested"
LINEAR_METHOD = "stable_singleton_linear"
BASELINE_METHODS = ("direct_greedy", "additive_nested", "v1_qubo_exact")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def rooted(root: Path, value: str) -> Path:
    path = (root / value.replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path leaves repository root: {value}") from error
    return path


def verify_descriptor(root: Path, descriptor: dict[str, Any]) -> Path:
    path = rooted(root, str(descriptor["path"]))
    if not path.is_file() or file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"descriptor differs: {path}")
    if "size_bytes" in descriptor and path.stat().st_size != int(
        descriptor["size_bytes"]
    ):
        raise ValueError(f"descriptor size differs: {path}")
    return path


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def scalar_subset_metrics(
    context: dict[str, Any], subset: tuple[str, ...], split: str
) -> dict[str, float]:
    values = {
        matrix_id: subset_bedroc(
            list(context["matrices"][matrix_id][split]), subset
        )
        for matrix_id in MATRIX_IDS
    }
    seeds = [values[seed_id] for seed_id in SEED_IDS]
    values["mean_seed"] = statistics.fmean(seeds)
    values["worst_seed"] = min(seeds)
    values["robust_composite"] = statistics.fmean(
        (values["primary"], values["mean_seed"], values["worst_seed"])
    )
    return values


def block_view(
    context: dict[str, Any], ligand_ids: set[str]
) -> dict[str, Any]:
    return {
        "matrices": {
            matrix_id: {
                "train": [
                    row
                    for row in context["matrices"][matrix_id]["train"]
                    if row["ligand_id"] in ligand_ids
                ]
            }
            for matrix_id in MATRIX_IDS
        }
    }


def independent_statistics(
    context: dict[str, Any],
    receptor_ids: list[str],
    blocks_by_ligand: dict[str, int],
    bedroc_alpha: float,
) -> dict[str, Any]:
    singletons = [(receptor_id,) for receptor_id in receptor_ids]
    pairs = list(itertools.combinations(receptor_ids, 2))
    full_single = score_subsets(
        context, singletons, receptor_ids, "train", bedroc_alpha
    )["robust_composite"]
    minimum = float(np.min(full_single))
    maximum = float(np.max(full_single))
    linear = (
        np.zeros_like(full_single)
        if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-15)
        else (full_single - minimum) / (maximum - minimum)
    )
    receptor_index = {
        receptor_id: index for index, receptor_id in enumerate(receptor_ids)
    }
    synergies: list[np.ndarray] = []
    block_ids = sorted(set(blocks_by_ligand.values()))
    for block_id in block_ids:
        ligand_ids = {
            ligand_id
            for ligand_id, value in blocks_by_ligand.items()
            if value == block_id
        }
        view = block_view(context, ligand_ids)
        singleton_values = score_subsets(
            view, singletons, receptor_ids, "train", bedroc_alpha
        )["robust_composite"]
        pair_values = score_subsets(
            view, pairs, receptor_ids, "train", bedroc_alpha
        )["robust_composite"]
        pair_baseline = np.asarray(
            [
                max(
                    singleton_values[receptor_index[first]],
                    singleton_values[receptor_index[second]],
                )
                for first, second in pairs
            ]
        )
        synergies.append(pair_values - pair_baseline)
    matrix = np.vstack(synergies)
    return {
        "receptor_ids": receptor_ids,
        "pairs": pairs,
        "linear_raw": full_single,
        "linear_normalized": linear,
        "pair_mean": np.mean(matrix, axis=0),
        "pair_std": np.std(matrix, axis=0, ddof=0),
        "pair_positive_fraction": np.mean(matrix > 0.0, axis=0),
    }


def independent_coefficients(
    stats: dict[str, Any],
    pair_weight: float,
    risk_kappa: float,
    target_size: int,
    penalty: float,
    minimum_fraction: float,
    minimum_lcb: float,
) -> tuple[dict[str, Any], int]:
    lcb = stats["pair_mean"] - risk_kappa * stats["pair_std"]
    retained = (
        (stats["pair_positive_fraction"] >= minimum_fraction)
        & (lcb >= minimum_lcb)
    )
    raw = np.where(retained, lcb, 0.0)
    maximum = float(np.max(raw))
    normalized = raw / maximum if maximum > 0.0 else np.zeros_like(raw)
    receptor_ids = list(stats["receptor_ids"])
    pairs = list(stats["pairs"])
    coefficients = {
        "convention": (
            "Q(x)=constant+sum_i linear[i]*x_i+"
            "sum_i<j quadratic[i__j]*x_i*x_j"
        ),
        "constant": penalty * target_size**2,
        "linear": {
            receptor_id: -float(stats["linear_normalized"][index])
            + penalty * (1 - 2 * target_size)
            for index, receptor_id in enumerate(receptor_ids)
        },
        "quadratic": {
            f"{first}__{second}": -pair_weight * float(normalized[index])
            + 2.0 * penalty
            for index, (first, second) in enumerate(pairs)
        },
        "target_size": target_size,
        "cardinality_penalty": penalty,
        "utility": {
            "linear_raw": {
                receptor_id: float(stats["linear_raw"][index])
                for index, receptor_id in enumerate(receptor_ids)
            },
            "linear_normalized": {
                receptor_id: float(stats["linear_normalized"][index])
                for index, receptor_id in enumerate(receptor_ids)
            },
            "pair_weight": pair_weight,
            "risk_kappa": risk_kappa,
            "minimum_positive_block_fraction": minimum_fraction,
            "minimum_pair_lcb": minimum_lcb,
            "pair_stable_raw": {
                f"{first}__{second}": float(raw[index])
                for index, (first, second) in enumerate(pairs)
            },
            "pair_stable_normalized": {
                f"{first}__{second}": float(normalized[index])
                for index, (first, second) in enumerate(pairs)
            },
        },
    }
    return coefficients, int(retained.sum())


def fixed_cardinality_optimum(
    coefficients: dict[str, Any], receptor_ids: list[str], target_size: int
) -> tuple[tuple[str, ...], float, np.ndarray, list[tuple[str, ...]]]:
    subsets = sorted(
        tuple(sorted(value))
        for value in itertools.combinations(receptor_ids, target_size)
    )
    energies = np.asarray(
        [coefficient_energy(subset, coefficients) for subset in subsets]
    )
    index = int(np.argmin(energies))
    return subsets[index], float(energies[index]), energies, subsets


def pair_count(subset: tuple[str, ...], coefficients: dict[str, Any]) -> int:
    pairs = coefficients["utility"]["pair_stable_normalized"]
    count = 0
    for first, second in itertools.combinations(subset, 2):
        key = f"{first}__{second}"
        if key not in pairs:
            key = f"{second}__{first}"
        count += float(pairs[key]) > 0.0
    return count


def candidate_key(rows: list[dict[str, str]], pair_weight: float, risk_kappa: float) -> tuple[float, ...]:
    selected = [
        row
        for row in rows
        if float(row["pair_weight"]) == pair_weight
        and float(row["risk_kappa"]) == risk_kappa
    ]
    return (
        -statistics.fmean(
            float(row["validation_robust_composite"]) for row in selected
        ),
        -min(float(row["validation_robust_composite"]) for row in selected),
        -statistics.fmean(
            float(row["validation_rank_spearman"]) for row in selected
        ),
        pair_weight,
        -risk_kappa,
    )


def selected_candidate(rows: list[dict[str, str]]) -> tuple[float, float]:
    candidates = sorted(
        {
            (float(row["pair_weight"]), float(row["risk_kappa"]))
            for row in rows
        }
    )
    return min(candidates, key=lambda value: candidate_key(rows, *value))


def paired_deltas(
    rows: list[dict[str, str]], left: str, right: str
) -> dict[str, Any]:
    indexed = {
        (row["target_id"], int(row["outer_fold"]), row["method"]): row
        for row in rows
    }
    per_target: dict[str, dict[str, Any]] = {}
    all_values: list[float] = []
    for target_id in sorted({row["target_id"] for row in rows}):
        folds = sorted(
            int(row["outer_fold"])
            for row in rows
            if row["target_id"] == target_id and row["method"] == left
        )
        values = [
            float(indexed[(target_id, fold, left)]["holdout_robust_composite"])
            - float(indexed[(target_id, fold, right)]["holdout_robust_composite"])
            for fold in folds
        ]
        all_values.extend(values)
        per_target[target_id] = {
            "mean_delta": statistics.fmean(values),
            "positive_fold_count": sum(value > 0.0 for value in values),
        }
    return {
        "fold_count": len(all_values),
        "mean_delta": statistics.fmean(all_values),
        "positive_fold_count": sum(value > 0.0 for value in all_values),
        "per_target": per_target,
    }


def compare_metrics(
    row: dict[str, str], metrics: dict[str, float], prefix: str, label: str
) -> None:
    for field in (
        "primary",
        "sensitivity",
        "mean_seed",
        "worst_seed",
        "robust_composite",
    ):
        assert_close(
            float(row[f"{prefix}_{field}"]),
            float(metrics[field]),
            f"{label}/{prefix}/{field}",
        )


def compare_nested(actual: Any, expected: Any, label: str) -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{label} keys differ")
        for key in actual:
            compare_nested(actual[key], expected[key], f"{label}/{key}")
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        assert_close(float(actual), float(expected), label)
    elif actual != expected:
        raise ValueError(f"{label} differs: {actual!r} != {expected!r}")


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

    result_path = rooted(root, config["outputs"]["result_json"])
    result = read_json(result_path)
    if result["status"] != "stage19f_stable_pair_qubo_not_supported_do_not_amend_bace1":
        raise ValueError("unexpected Stage 19f result status")
    if result["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("result identifies another config")
    expected_boundary = {
        "train_rows_read_by_target": {"MK14": 696, "PPARG": 668},
        "fresh_validation_rows_read": 0,
        "test_rows_read": 0,
        "bace1_docking_rows_read": 0,
    }
    if result["data_boundary"] != expected_boundary:
        raise ValueError("Stage 19f data boundary differs")

    output_paths = {
        key: verify_descriptor(root, descriptor)
        for key, descriptor in result["outputs"].items()
    }
    inner_rows = read_csv(output_paths["inner_candidate_trials_csv"])
    outer_rows = read_csv(output_paths["outer_candidate_trials_csv"])
    method_rows = read_csv(output_paths["comparison_method_results_csv"])
    model = read_json(output_paths["model_record_json"])

    diagnostic = config["diagnostic"]
    outer_count = int(diagnostic["outer_fold_count"])
    inner_count = int(diagnostic["inner_fold_count"])
    candidates = list(
        itertools.product(
            [float(value) for value in diagnostic["pair_weights"]],
            [float(value) for value in diagnostic["risk_kappas"]],
        )
    )
    expected_inner = len(source_config["targets"]) * outer_count * inner_count * len(candidates)
    expected_outer = len(source_config["targets"]) * outer_count * len(candidates)
    expected_methods = len(source_config["targets"]) * outer_count * 5
    if (len(inner_rows), len(outer_rows), len(method_rows)) != (
        expected_inner,
        expected_outer,
        expected_methods,
    ):
        raise ValueError("Stage 19f output row count differs")
    if len({tuple(row.values()) for row in inner_rows}) != len(inner_rows):
        raise ValueError("duplicate inner candidate row")
    if len({tuple(row.values()) for row in outer_rows}) != len(outer_rows):
        raise ValueError("duplicate outer candidate row")

    target_size = int(diagnostic["target_size"])
    penalty = float(diagnostic["cardinality_penalty"])
    minimum_fraction = float(diagnostic["minimum_positive_block_fraction"])
    minimum_lcb = float(diagnostic["minimum_pair_lcb"])
    bedroc_alpha = float(diagnostic["bedroc_alpha"])
    metric_sets = 0
    outer_candidates_recomputed = 0
    group_outer_folds: dict[tuple[str, str], set[int]] = defaultdict(set)
    scaffold_outer_folds: dict[tuple[str, str], set[int]] = defaultdict(set)
    group_inner_folds: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    scaffold_inner_folds: dict[tuple[str, int, str], set[int]] = defaultdict(set)

    for target_id, spec in source_config["targets"].items():
        ligands, receptor_ids, matrices = load_target(root, target_id, spec)
        assignments = make_frozen_group_folds(
            ligands, outer_count, int(diagnostic["fold_seed"])
        )
        for row in ligands:
            group_outer_folds[(target_id, row["split_group_id"])].add(
                assignments[row["ligand_id"]]
            )
            scaffold_outer_folds[(target_id, row["scaffold_smiles"])].add(
                assignments[row["ligand_id"]]
            )
        all_ids = {row["ligand_id"] for row in ligands}
        triples = sorted(
            tuple(sorted(value))
            for value in itertools.combinations(receptor_ids, target_size)
        )
        triple_index = {value: index for index, value in enumerate(triples)}
        model_spec = {
            "coverage_fraction": float(spec["v1_qubo"]["coverage_fraction"]),
            "utility_metric": "bedroc",
        }

        for outer_fold in range(outer_count):
            holdout = {
                ligand_id
                for ligand_id, fold in assignments.items()
                if fold == outer_fold
            }
            train = all_ids - holdout
            train_rows = [row for row in ligands if row["ligand_id"] in train]
            inner = make_frozen_group_folds(
                train_rows,
                inner_count,
                int(diagnostic["inner_fold_seed"]) + outer_fold,
            )
            for row in train_rows:
                group_inner_folds[(target_id, outer_fold, row["split_group_id"])].add(
                    inner[row["ligand_id"]]
                )
                scaffold_inner_folds[(target_id, outer_fold, row["scaffold_smiles"])].add(
                    inner[row["ligand_id"]]
                )

            fold_inner = [
                row
                for row in inner_rows
                if row["target_id"] == target_id
                and int(row["outer_fold"]) == outer_fold
            ]
            chosen = selected_candidate(fold_inner)
            fold_outer = [
                row
                for row in outer_rows
                if row["target_id"] == target_id
                and int(row["outer_fold"]) == outer_fold
            ]
            selected_outer = [
                row for row in fold_outer if row["selected_by_inner_cv"] == "True"
            ]
            if len(selected_outer) != 1 or (
                float(selected_outer[0]["pair_weight"]),
                float(selected_outer[0]["risk_kappa"]),
            ) != chosen:
                raise ValueError(f"{target_id}/{outer_fold} nested choice differs")

            context = make_context(train, holdout, matrices, receptor_ids, model_spec)
            stats = independent_statistics(context, receptor_ids, inner, bedroc_alpha)
            holdout_values = score_subsets(
                context, triples, receptor_ids, "validation", bedroc_alpha
            )
            selected_methods = [
                row
                for row in method_rows
                if row["target_id"] == target_id
                and int(row["outer_fold"]) == outer_fold
                and row["method"] in (CANDIDATE_METHOD, LINEAR_METHOD)
            ]
            if len(selected_methods) != 2:
                raise ValueError(f"{target_id}/{outer_fold} method rows differ")

            for row in fold_outer:
                pair_weight = float(row["pair_weight"])
                risk_kappa = float(row["risk_kappa"])
                coefficients, retained_count = independent_coefficients(
                    stats,
                    pair_weight,
                    risk_kappa,
                    target_size,
                    penalty,
                    minimum_fraction,
                    minimum_lcb,
                )
                subset, _, energies, _ = fixed_cardinality_optimum(
                    coefficients, receptor_ids, target_size
                )
                if "+".join(subset) != row["selected_subset"]:
                    raise ValueError(f"{target_id}/{outer_fold} outer subset differs")
                if retained_count != int(row["retained_pair_count"]):
                    raise ValueError(f"{target_id}/{outer_fold} retained pair count differs")
                if pair_count(subset, coefficients) != int(
                    row["selected_subset_pair_count"]
                ):
                    raise ValueError(f"{target_id}/{outer_fold} selected pair count differs")
                index = triple_index[subset]
                compare_metrics(
                    row,
                    {
                        key: float(values[index])
                        for key, values in holdout_values.items()
                    },
                    "validation",
                    f"{target_id}/{outer_fold}/{pair_weight}/{risk_kappa}",
                )
                assert_close(
                    float(row["validation_rank_spearman"]),
                    safe_spearman(-energies, holdout_values["robust_composite"]),
                    f"{target_id}/{outer_fold}/{pair_weight}/{risk_kappa}/rank",
                )
                outer_candidates_recomputed += 1

            for row in selected_methods:
                if row["method"] == CANDIDATE_METHOD:
                    pair_weight, risk_kappa = chosen
                else:
                    pair_weight, risk_kappa = 0.0, 0.0
                coefficients, retained_count = independent_coefficients(
                    stats,
                    pair_weight,
                    risk_kappa,
                    target_size,
                    penalty,
                    minimum_fraction,
                    minimum_lcb,
                )
                subset, _, _, _ = fixed_cardinality_optimum(
                    coefficients, receptor_ids, target_size
                )
                if "+".join(subset) != row["selected_subset"]:
                    raise ValueError(f"{target_id}/{outer_fold}/{row['method']} differs")
                compare_metrics(
                    row,
                    scalar_subset_metrics(context, subset, "train"),
                    "train",
                    f"{target_id}/{outer_fold}/{row['method']}",
                )
                compare_metrics(
                    row,
                    scalar_subset_metrics(context, subset, "validation"),
                    "holdout",
                    f"{target_id}/{outer_fold}/{row['method']}",
                )
                if retained_count != int(row["retained_pair_count"]):
                    raise ValueError("method retained pair count differs")
                metric_sets += 2

        target_outer = [row for row in outer_rows if row["target_id"] == target_id]
        full_choice = selected_candidate(target_outer)
        full_context = make_context(all_ids, set(), matrices, receptor_ids, model_spec)
        full_stats = independent_statistics(
            full_context, receptor_ids, assignments, bedroc_alpha
        )
        coefficients, retained_count = independent_coefficients(
            full_stats,
            full_choice[0],
            full_choice[1],
            target_size,
            penalty,
            minimum_fraction,
            minimum_lcb,
        )
        subset, energy, _, _ = fixed_cardinality_optimum(
            coefficients, receptor_ids, target_size
        )
        recorded = model["target_development_models"][target_id]
        if full_choice != (
            float(recorded["selected_pair_weight"]),
            float(recorded["selected_risk_kappa"]),
        ):
            raise ValueError(f"{target_id} full-train candidate differs")
        if list(subset) != recorded["selected_subset"]:
            raise ValueError(f"{target_id} full-train subset differs")
        if retained_count != int(recorded["retained_pair_count"]):
            raise ValueError(f"{target_id} full-train retained count differs")
        assert_close(energy, float(recorded["fixed_cardinality_energy"]), f"{target_id}/energy")
        compare_nested(coefficients, recorded["coefficients"], f"{target_id}/coefficients")
        metrics = scalar_subset_metrics(full_context, subset, "train")
        metrics = {
            field: metrics[field]
            for field in (
                "primary",
                "sensitivity",
                "mean_seed",
                "worst_seed",
                "robust_composite",
            )
        }
        compare_nested(metrics, recorded["full_train_metrics"], f"{target_id}/metrics")
        exact_subset, exact_energy = exact_all_cardinalities(receptor_ids, coefficients)
        if list(exact_subset) != recorded["exact_all_cardinalities_subset"]:
            raise ValueError(f"{target_id} all-cardinality optimum differs")
        assert_close(
            exact_energy,
            float(recorded["exact_all_cardinalities_energy"]),
            f"{target_id}/all-cardinality energy",
        )

    if any(len(folds) != 1 for folds in group_outer_folds.values()):
        raise ValueError("an outer split group crosses folds")
    if any(len(folds) != 1 for folds in scaffold_outer_folds.values()):
        raise ValueError("an outer scaffold crosses folds")
    if any(len(folds) != 1 for folds in group_inner_folds.values()):
        raise ValueError("an inner split group crosses folds")
    if any(len(folds) != 1 for folds in scaffold_inner_folds.values()):
        raise ValueError("an inner scaffold crosses folds")

    comparisons = {
        f"stable_pair_vs_{method}": paired_deltas(
            method_rows, CANDIDATE_METHOD, method
        )
        for method in (*BASELINE_METHODS, LINEAR_METHOD)
    }
    for key, recomputed in comparisons.items():
        recorded = result["paired_comparisons"][key]
        assert_close(recomputed["mean_delta"], recorded["mean_delta"], f"{key}/mean")
        if recomputed["positive_fold_count"] != recorded["positive_fold_count"]:
            raise ValueError(f"{key} positive fold count differs")
        for target_id, values in recomputed["per_target"].items():
            assert_close(
                values["mean_delta"],
                recorded["per_target"][target_id]["mean_delta"],
                f"{key}/{target_id}/mean",
            )

    gate_spec = config["development_support_gate"]
    checks = {
        key: (
            all(
                values["mean_delta"] > float(gate_spec["minimum_target_mean_delta"])
                for values in comparison["per_target"].values()
            )
            and comparison["positive_fold_count"]
            >= int(gate_spec["minimum_positive_folds_of_eight"])
        )
        for key, comparison in comparisons.items()
    }
    pair_folds = sum(
        int(row["selected_subset_pair_count"]) > 0
        for row in method_rows
        if row["method"] == CANDIDATE_METHOD
    )
    gate_passed = (
        all(checks.values())
        and pair_folds >= int(gate_spec["minimum_folds_with_selected_nonzero_pair"])
    )
    if gate_passed or result["gate"]["passed"] or result["gate"]["bace1_amendment_authorized"]:
        raise ValueError("failed Stage 19f gate was not preserved")
    if model["status"] != "development_gate_failed_not_authorized_for_bace1":
        raise ValueError("Stage 19f model authorization differs")
    if model["data_boundary"] != expected_boundary:
        raise ValueError("Stage 19f model data boundary differs")

    forbidden = ("fresh_validation", "locked_test", "bace1_docking")
    paths = [
        str(descriptor["path"]).lower()
        for descriptor in config["inputs"].values()
    ]
    if any(marker in path for marker in forbidden for path in paths):
        raise ValueError("Stage 19f input path crosses its data boundary")

    return {
        "schema_version": "1.0",
        "status": "stage19f_cross_target_stable_pair_qubo_audit_ok",
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
            "target_count": len(source_config["targets"]),
            "inner_candidate_rows_reselected": len(inner_rows),
            "outer_candidate_rows_raw_recomputed": outer_candidates_recomputed,
            "selected_method_train_or_holdout_metric_sets_recomputed": metric_sets,
            "full_train_qubo_optima_reenumerated": len(source_config["targets"]),
        },
        "checks": {
            "all_input_and_output_hashes_verified": True,
            "outer_scaffolds_fold_disjoint": True,
            "inner_scaffolds_fold_disjoint": True,
            "all_nested_candidates_reselected": True,
            "all_outer_candidate_subsets_and_metrics_recomputed": True,
            "full_train_coefficients_recomputed": True,
            "all_paired_deltas_recomputed": True,
            "failed_gate_reproduced": True,
            "bace1_amendment_authorized": False,
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
