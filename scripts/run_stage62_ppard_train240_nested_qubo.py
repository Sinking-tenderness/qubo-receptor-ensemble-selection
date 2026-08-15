"""Run the frozen Stage60 nested QUBO analysis on PPARD Train-240."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_stage42d_bace1_large_pool_qubo_screen import bedroc_metrics, rank_cube
from scripts.run_stage42f_bace1_rank_sensitive_pair_qubo import (
    classical_by_size,
    direct_greedy_by_size,
    exact_by_size,
    pair_coefficients,
    qubo_value,
)
from scripts.run_stage53_ppara_large_pool_qubo_transfer import (
    linear_by_size,
    metric_greedy_by_size,
)


TOLERANCE = 1e-12
SEED_IDS = ("seed0", "seed1", "seed2")
METHOD_ORDER = (
    "rank_pair_qubo_exact",
    "rank_pair_strong_classical",
    "rank_pair_direct_greedy",
    "best_single_receptor",
    "bedroc_linear_topk",
    "bedroc_nested_greedy",
    "bedroc_random_search",
    "all_receptors",
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
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
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
        raise ValueError(f"Stage62 frozen input differs: {path}")
    return path


def subset_name(subset: tuple[int, ...], receptor_ids: list[str]) -> str:
    return "+".join(receptor_ids[index] for index in subset)


def build_score_cube(
    rows: list[dict[str, str]], ligand_ids: list[str], receptor_ids: list[str]
) -> np.ndarray:
    ligand_index = {value: index for index, value in enumerate(ligand_ids)}
    receptor_index = {value: index for index, value in enumerate(receptor_ids)}
    seed_index = {value: index for index, value in enumerate(SEED_IDS)}
    cube = np.full((3, len(ligand_ids), len(receptor_ids)), np.nan, dtype=float)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if row["target_id"] != "PPARD":
            raise ValueError("Stage62 requires corrected PPARD target IDs")
        key = (row["seed_id"], row["ligand_id"], row["receptor_id"])
        if key in seen:
            raise ValueError(f"duplicate Stage62 score key: {key}")
        if row["seed_id"] not in seed_index:
            raise ValueError(f"unknown Stage62 seed: {row['seed_id']}")
        seen.add(key)
        try:
            cube[
                seed_index[row["seed_id"]],
                ligand_index[row["ligand_id"]],
                receptor_index[row["receptor_id"]],
            ] = float(row["gpu_score"])
        except KeyError as error:
            raise ValueError(f"unknown Stage62 score identity: {key}") from error
    expected = 3 * len(ligand_ids) * len(receptor_ids)
    if len(seen) != expected or not np.isfinite(cube).all():
        raise ValueError("Stage62 score cube is incomplete")
    return cube


def validate_inputs(
    config: dict[str, Any], root: Path
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, int],
    dict[tuple[int, str], int],
    dict[str, Any],
]:
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    stage60 = read_json(inputs["stage60_result"])
    stage60_audit = read_json(inputs["stage60_audit"])
    model = read_json(inputs["stage60_model_record"])
    pilot_matrix_audit = read_json(inputs["pilot_matrix_audit"])
    remaining_matrix_audit = read_json(inputs["remaining_matrix_audit"])
    pilot_amendment_audit = read_json(inputs["pilot_target_id_audit"])
    remaining_amendment_audit = read_json(inputs["remaining_target_id_audit"])
    if stage60.get("status") != "stage60_ppard_transferred_qubo_and_k_rule_frozen":
        raise ValueError("Stage60 freeze did not pass")
    if stage60_audit.get("status") != "stage60_ppard_transferred_qubo_independent_audit_ok":
        raise ValueError("Stage60 independent audit did not pass")
    if model.get("status") != "stage60_ppard_transferred_qubo_frozen":
        raise ValueError("Stage60 model record did not pass")
    if stage60["transferred_objective"] != config["objective"] or model["objective"] != config["objective"]:
        raise ValueError("Stage62 objective differs from Stage60")
    if stage60["nested_cv"] != config["nested_cv"] or model["nested_cv"] != config["nested_cv"]:
        raise ValueError("Stage62 nested-CV rule differs from Stage60")
    if model["comparators"] != config["comparators"]:
        raise ValueError("Stage62 comparators differ from Stage60")
    if pilot_matrix_audit.get("status") != "independent_stage58b_ppard_pilot96_unidock_matrix_audit_ok":
        raise ValueError("Stage58b pilot matrix audit did not pass")
    if remaining_matrix_audit.get("status") != "independent_stage61b_ppard_remaining144_unidock_matrix_audit_ok":
        raise ValueError("Stage61b remaining matrix audit did not pass")
    if pilot_amendment_audit.get("status") != "stage58c_ppard_target_id_amendment_independent_audit_ok":
        raise ValueError("Stage58c target-id amendment audit did not pass")
    if remaining_amendment_audit.get("status") != "stage61c_ppard_target_id_amendment_independent_audit_ok":
        raise ValueError("Stage61c target-id amendment audit did not pass")

    ligands = read_csv(inputs["train_manifest"])
    receptors = read_csv(inputs["receptor_manifest"])
    pilot_scores = read_csv(inputs["pilot_scores"])
    remaining_scores = read_csv(inputs["remaining_scores"])
    if len(ligands) != 240 or Counter(row["label"] for row in ligands) != {
        "active": 120,
        "decoy": 120,
    }:
        raise ValueError("Stage62 Train-240 dimensions differ")
    if len(receptors) != 29 or any(row["status"] != "ok" for row in receptors):
        raise ValueError("Stage62 receptor manifest differs")
    if {row["split"] for row in ligands} != {"train"} or {
        row["selection_role"] for row in ligands
    } != {"development_train"}:
        raise ValueError("Stage62 crossed a train boundary")
    ligand_ids = [row["ligand_id"] for row in ligands]
    if len(set(ligand_ids)) != 240:
        raise ValueError("Stage62 ligand IDs are not unique")
    pilot_ids = {row["ligand_id"] for row in ligands if row["pilot_selected"] == "True"}
    remaining_ids = {
        row["ligand_id"] for row in ligands if row["pilot_selected"] == "False"
    }
    if len(pilot_ids) != 96 or len(remaining_ids) != 144 or pilot_ids & remaining_ids:
        raise ValueError("Stage62 pilot/remaining partition differs")
    if {row["ligand_id"] for row in pilot_scores} != pilot_ids:
        raise ValueError("Stage62 pilot score identities differ")
    if {row["ligand_id"] for row in remaining_scores} != remaining_ids:
        raise ValueError("Stage62 remaining score identities differ")
    score_rows = pilot_scores + remaining_scores
    if len(score_rows) != 20880 or any(
        row["status"] != "ok" or row["pose_integrity_status"] != "ok"
        for row in score_rows
    ):
        raise ValueError("Stage62 score technical gate differs")

    by_id = {row["ligand_id"]: row for row in ligands}
    outer_rows = read_csv(inputs["outer_assignments"])
    inner_rows = read_csv(inputs["inner_assignments"])
    if len(outer_rows) != 240 or {row["ligand_id"] for row in outer_rows} != set(ligand_ids):
        raise ValueError("Stage62 outer assignments differ")
    outer = {row["ligand_id"]: int(row["outer_fold"]) for row in outer_rows}
    if set(outer.values()) != {0, 1, 2, 3}:
        raise ValueError("Stage62 outer fold IDs differ")
    for row in outer_rows:
        source = by_id[row["ligand_id"]]
        for key in ("label", "split_group_id", "scaffold_smiles", "pilot_selected"):
            if row[key] != source[key]:
                raise ValueError(f"Stage62 outer metadata differs: {row['ligand_id']}/{key}")
    if len(inner_rows) != 720:
        raise ValueError("Stage62 inner assignment row count differs")
    inner: dict[tuple[int, str], int] = {}
    for row in inner_rows:
        outer_fold = int(row["outer_fold"])
        ligand_id = row["ligand_id"]
        if outer[ligand_id] == outer_fold:
            raise ValueError("Stage62 outer holdout entered an inner fold")
        source = by_id[ligand_id]
        for key in ("label", "split_group_id", "scaffold_smiles"):
            if row[key] != source[key]:
                raise ValueError(f"Stage62 inner metadata differs: {ligand_id}/{key}")
        key = (outer_fold, ligand_id)
        if key in inner:
            raise ValueError(f"duplicate Stage62 inner assignment: {key}")
        inner[key] = int(row["inner_fold"])
    for outer_fold in range(4):
        expected_ids = {value for value in ligand_ids if outer[value] != outer_fold}
        observed_ids = {value for fold, value in inner if fold == outer_fold}
        if observed_ids != expected_ids or {
            inner[(outer_fold, value)] for value in observed_ids
        } != {0, 1, 2}:
            raise ValueError(f"Stage62 inner coverage differs: outer{outer_fold}")
        for column in ("split_group_id", "scaffold_smiles"):
            folds: dict[str, set[int]] = defaultdict(set)
            for ligand_id in observed_ids:
                folds[by_id[ligand_id][column]].add(inner[(outer_fold, ligand_id)])
            if any(len(values) != 1 for values in folds.values()):
                raise ValueError(f"Stage62 {column} crosses inner folds")

    audit = {
        "status": "stage62_input_audit_ok",
        "target_id": "PPARD",
        "ligand_count": len(ligands),
        "label_counts": dict(sorted(Counter(row["label"] for row in ligands).items())),
        "pilot_ligand_count": len(pilot_ids),
        "remaining_ligand_count": len(remaining_ids),
        "receptor_count": len(receptors),
        "seed_count": 3,
        "score_row_count": len(score_rows),
        "outer_fold_count": 4,
        "inner_assignment_rows": len(inner_rows),
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
    }
    return ligands, receptors, score_rows, outer, inner, audit


def fit_paths(
    ranks: np.ndarray,
    labels: np.ndarray,
    maximum_size: int,
    alpha: float,
    beam_width: int,
) -> dict[str, Any]:
    singleton, complement = pair_coefficients(ranks, labels, alpha)
    exact, state_count, _ = exact_by_size(
        ranks.shape[2], maximum_size, singleton, complement
    )
    strong, search_records = classical_by_size(
        ranks.shape[2], maximum_size, singleton, complement, beam_width
    )
    direct = direct_greedy_by_size(
        ranks.shape[2], maximum_size, singleton, complement
    )
    linear = linear_by_size(ranks, labels, maximum_size, alpha)
    nested = metric_greedy_by_size(ranks, labels, maximum_size, alpha)
    return {
        "singleton": singleton,
        "complement": complement,
        "exact": exact,
        "strong": strong,
        "direct": direct,
        "linear": linear,
        "nested": nested,
        "state_count": state_count,
        "strong_search_records": search_records,
    }


def random_fixed_k(
    ranks: np.ndarray,
    labels: np.ndarray,
    subset_size: int,
    alpha: float,
    samples: int,
    seed: int,
) -> tuple[int, ...]:
    receptor_count = ranks.shape[2]
    target = min(samples, math.comb(receptor_count, subset_size))
    generator = random.Random(seed)
    candidates: set[tuple[int, ...]] = set()
    universe = list(range(receptor_count))
    while len(candidates) < target:
        candidates.add(tuple(sorted(generator.sample(universe, subset_size))))
    return min(
        candidates,
        key=lambda subset: (
            -bedroc_metrics(ranks, labels, subset, alpha)["robust_bedroc_composite"],
            subset,
        ),
    )


def one_standard_error(values: dict[int, list[float]]) -> dict[str, Any]:
    summary: dict[int, dict[str, float]] = {}
    for subset_size, observed in sorted(values.items()):
        if not observed:
            raise ValueError(f"empty Stage62 k cell: {subset_size}")
        mean = statistics.fmean(observed)
        standard_error = (
            statistics.stdev(observed) / math.sqrt(len(observed))
            if len(observed) > 1
            else 0.0
        )
        summary[subset_size] = {"mean": mean, "standard_error": standard_error}
    best_k = min(summary, key=lambda k: (-summary[k]["mean"], k))
    threshold = summary[best_k]["mean"] - summary[best_k]["standard_error"]
    selected_k = min(
        k for k in summary if summary[k]["mean"] >= threshold - TOLERANCE
    )
    return {
        "best_k": best_k,
        "selected_k": selected_k,
        "one_standard_error_threshold": threshold,
        "by_k": {str(key): value for key, value in summary.items()},
    }


def metric_record(
    outer_fold: int | str,
    method: str,
    subset: tuple[int, ...],
    train_ranks: np.ndarray,
    train_labels: np.ndarray,
    evaluation_ranks: np.ndarray,
    evaluation_labels: np.ndarray,
    receptor_ids: list[str],
    alpha: float,
    singleton: np.ndarray,
    complement: np.ndarray,
) -> dict[str, Any]:
    train = bedroc_metrics(train_ranks, train_labels, subset, alpha)
    evaluation = bedroc_metrics(evaluation_ranks, evaluation_labels, subset, alpha)
    return {
        "outer_fold": outer_fold,
        "method": method,
        "subset_size": len(subset),
        "selected_subset": subset_name(subset, receptor_ids),
        "train_qubo_objective": qubo_value(subset, singleton, complement),
        "train_robust_bedroc": train["robust_bedroc_composite"],
        "evaluation_primary_bedroc": evaluation["primary_bedroc"],
        "evaluation_mean_seed_bedroc": evaluation["mean_seed_bedroc"],
        "evaluation_worst_seed_bedroc": evaluation["worst_seed_bedroc"],
        "evaluation_robust_bedroc": evaluation["robust_bedroc_composite"],
    }


def gap_row(
    scope: str,
    outer_fold: int | str,
    inner_fold: int | str,
    subset_size: int,
    paths: dict[str, Any],
    receptor_ids: list[str],
) -> dict[str, Any]:
    exact = paths["exact"][subset_size]
    strong = paths["strong"][subset_size]
    direct = paths["direct"][subset_size]
    singleton = paths["singleton"]
    complement = paths["complement"]
    exact_value = qubo_value(exact, singleton, complement)
    strong_value = qubo_value(strong, singleton, complement)
    direct_value = qubo_value(direct, singleton, complement)
    return {
        "scope": scope,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "subset_size": subset_size,
        "exact_subset": subset_name(exact, receptor_ids),
        "strong_classical_subset": subset_name(strong, receptor_ids),
        "direct_greedy_subset": subset_name(direct, receptor_ids),
        "exact_objective": exact_value,
        "strong_classical_objective": strong_value,
        "direct_greedy_objective": direct_value,
        "exact_minus_strong_gap": exact_value - strong_value,
        "exact_minus_direct_greedy_gap": exact_value - direct_value,
    }


def method_subsets(
    paths: dict[str, Any],
    subset_size: int,
    ranks: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    random_samples: int,
    random_seed: int,
) -> dict[str, tuple[int, ...]]:
    return {
        "rank_pair_qubo_exact": paths["exact"][subset_size],
        "rank_pair_strong_classical": paths["strong"][subset_size],
        "rank_pair_direct_greedy": paths["direct"][subset_size],
        "best_single_receptor": paths["linear"][1],
        "bedroc_linear_topk": paths["linear"][subset_size],
        "bedroc_nested_greedy": paths["nested"][subset_size],
        "bedroc_random_search": random_fixed_k(
            ranks, labels, subset_size, alpha, random_samples, random_seed
        ),
        "all_receptors": tuple(range(ranks.shape[2])),
    }


def compute_analysis(config: dict[str, Any], root: Path) -> dict[str, Any]:
    ligands, receptors, score_rows, outer, inner, input_audit = validate_inputs(
        config, root
    )
    ligand_ids = [row["ligand_id"] for row in ligands]
    receptor_ids = [row["conformer_id"] for row in receptors]
    labels = np.asarray([int(row["label"] == "active") for row in ligands], dtype=int)
    scores = build_score_cube(score_rows, ligand_ids, receptor_ids)
    alpha = float(config["objective"]["bedroc_alpha"])
    candidate_k = [int(value) for value in config["nested_cv"]["candidate_k_values"]]
    maximum_size = max(candidate_k)
    execution = dict(config["execution"])
    beam_width = int(execution["classical_beam_width"])
    random_samples = int(execution["random_samples_per_size"])
    random_seed = int(execution["random_seed"])
    outer_fold_count = int(config["nested_cv"]["outer_fold_count"])
    inner_fold_count = int(config["nested_cv"]["inner_fold_count"])
    ligand_index = {value: index for index, value in enumerate(ligand_ids)}

    inner_rows: list[dict[str, Any]] = []
    inner_selection_rows: list[dict[str, Any]] = []
    outer_k_rows: list[dict[str, Any]] = []
    nested_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for outer_fold in range(outer_fold_count):
        outer_train = np.asarray(
            [outer[ligand_id] != outer_fold for ligand_id in ligand_ids], dtype=bool
        )
        outer_holdout = ~outer_train
        inner_values = {subset_size: [] for subset_size in candidate_k}
        for inner_fold in range(inner_fold_count):
            inner_holdout = np.zeros(len(ligands), dtype=bool)
            for ligand_id in ligand_ids:
                if outer[ligand_id] != outer_fold and inner[(outer_fold, ligand_id)] == inner_fold:
                    inner_holdout[ligand_index[ligand_id]] = True
            inner_train = outer_train & ~inner_holdout
            ranks = rank_cube(scores, inner_train)
            paths = fit_paths(
                ranks[:, inner_train, :],
                labels[inner_train],
                maximum_size,
                alpha,
                beam_width,
            )
            for subset_size in candidate_k:
                subset = paths["exact"][subset_size]
                train_metrics = bedroc_metrics(
                    ranks[:, inner_train, :], labels[inner_train], subset, alpha
                )
                holdout_metrics = bedroc_metrics(
                    ranks[:, inner_holdout, :], labels[inner_holdout], subset, alpha
                )
                value = holdout_metrics["robust_bedroc_composite"]
                inner_values[subset_size].append(value)
                inner_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "subset_size": subset_size,
                        "selected_subset": subset_name(subset, receptor_ids),
                        "train_qubo_objective": qubo_value(
                            subset, paths["singleton"], paths["complement"]
                        ),
                        "train_robust_bedroc": train_metrics["robust_bedroc_composite"],
                        "holdout_primary_bedroc": holdout_metrics["primary_bedroc"],
                        "holdout_mean_seed_bedroc": holdout_metrics["mean_seed_bedroc"],
                        "holdout_worst_seed_bedroc": holdout_metrics["worst_seed_bedroc"],
                        "holdout_robust_bedroc": value,
                    }
                )
                gap_rows.append(
                    gap_row(
                        "inner_train",
                        outer_fold,
                        inner_fold,
                        subset_size,
                        paths,
                        receptor_ids,
                    )
                )
        inner_choice = one_standard_error(inner_values)
        for subset_size in candidate_k:
            values = inner_choice["by_k"][str(subset_size)]
            inner_selection_rows.append(
                {
                    "outer_fold": outer_fold,
                    "subset_size": subset_size,
                    "mean_inner_holdout_robust_bedroc": values["mean"],
                    "standard_error": values["standard_error"],
                    "best_k": inner_choice["best_k"],
                    "selected_k": inner_choice["selected_k"],
                    "one_standard_error_threshold": inner_choice[
                        "one_standard_error_threshold"
                    ],
                    "within_one_standard_error": values["mean"]
                    >= inner_choice["one_standard_error_threshold"] - TOLERANCE,
                }
            )

        outer_ranks = rank_cube(scores, outer_train)
        outer_train_ranks = outer_ranks[:, outer_train, :]
        paths = fit_paths(
            outer_train_ranks,
            labels[outer_train],
            maximum_size,
            alpha,
            beam_width,
        )
        for subset_size in candidate_k:
            subset = paths["exact"][subset_size]
            holdout = bedroc_metrics(
                outer_ranks[:, outer_holdout, :], labels[outer_holdout], subset, alpha
            )
            outer_k_rows.append(
                {
                    "outer_fold": outer_fold,
                    "subset_size": subset_size,
                    "selected_subset": subset_name(subset, receptor_ids),
                    "selected_by_inner_cv": subset_size == inner_choice["selected_k"],
                    "train_qubo_objective": qubo_value(
                        subset, paths["singleton"], paths["complement"]
                    ),
                    "holdout_primary_bedroc": holdout["primary_bedroc"],
                    "holdout_mean_seed_bedroc": holdout["mean_seed_bedroc"],
                    "holdout_worst_seed_bedroc": holdout["worst_seed_bedroc"],
                    "holdout_robust_bedroc": holdout["robust_bedroc_composite"],
                }
            )
            gap_rows.append(
                gap_row(
                    "outer_train",
                    outer_fold,
                    "",
                    subset_size,
                    paths,
                    receptor_ids,
                )
            )
        selected_k = int(inner_choice["selected_k"])
        subsets = method_subsets(
            paths,
            selected_k,
            outer_train_ranks,
            labels[outer_train],
            alpha,
            random_samples,
            random_seed + outer_fold,
        )
        for method in METHOD_ORDER:
            nested_rows.append(
                metric_record(
                    outer_fold,
                    method,
                    subsets[method],
                    outer_train_ranks,
                    labels[outer_train],
                    outer_ranks[:, outer_holdout, :],
                    labels[outer_holdout],
                    receptor_ids,
                    alpha,
                    paths["singleton"],
                    paths["complement"],
                )
            )

    outer_values = {
        subset_size: [
            float(row["holdout_robust_bedroc"])
            for row in outer_k_rows
            if int(row["subset_size"]) == subset_size
        ]
        for subset_size in candidate_k
    }
    final_choice = one_standard_error(outer_values)
    full_mask = np.ones(len(ligands), dtype=bool)
    full_ranks = rank_cube(scores, full_mask)
    full_paths = fit_paths(full_ranks, labels, maximum_size, alpha, beam_width)
    for subset_size in candidate_k:
        gap_rows.append(
            gap_row("full_train", "full", "", subset_size, full_paths, receptor_ids)
        )
    final_k = int(final_choice["selected_k"])
    final_subsets = method_subsets(
        full_paths,
        final_k,
        full_ranks,
        labels,
        alpha,
        random_samples,
        random_seed + 1000,
    )
    final_rows = [
        metric_record(
            "full",
            method,
            final_subsets[method],
            full_ranks,
            labels,
            full_ranks,
            labels,
            receptor_ids,
            alpha,
            full_paths["singleton"],
            full_paths["complement"],
        )
        for method in METHOD_ORDER
    ]

    nested_by_method = {
        method: sorted(
            [row for row in nested_rows if row["method"] == method],
            key=lambda row: int(row["outer_fold"]),
        )
        for method in METHOD_ORDER
    }
    exact_values = [
        float(row["evaluation_robust_bedroc"])
        for row in nested_by_method["rank_pair_qubo_exact"]
    ]

    def fold_delta(method: str) -> list[float]:
        return [
            exact - float(row["evaluation_robust_bedroc"])
            for exact, row in zip(exact_values, nested_by_method[method], strict=True)
        ]

    single_delta = fold_delta("best_single_receptor")
    linear_delta = fold_delta("bedroc_linear_topk")
    greedy_delta = fold_delta("bedroc_nested_greedy")
    strong_delta = fold_delta("rank_pair_strong_classical")
    application_gate = dict(config["claim_gates"])["application_support"]
    application_checks = {
        "minimum_mean_outer_holdout_gain_over_best_single": statistics.fmean(
            single_delta
        )
        >= float(application_gate["minimum_mean_outer_holdout_gain_over_best_single"]),
        "minimum_positive_outer_holdout_gain_folds": sum(
            value > TOLERANCE for value in single_delta
        )
        >= int(application_gate["minimum_positive_outer_holdout_gain_folds"]),
        "minimum_mean_outer_holdout_gain_over_linear_topk": statistics.fmean(
            linear_delta
        )
        >= float(application_gate["minimum_mean_outer_holdout_gain_over_linear_topk"]),
        "minimum_mean_outer_holdout_gain_over_nested_bedroc_greedy": statistics.fmean(
            greedy_delta
        )
        >= float(
            application_gate[
                "minimum_mean_outer_holdout_gain_over_nested_bedroc_greedy"
            ]
        ),
    }
    application_supported = all(application_checks.values())
    positive_strong_gap_cells = sum(
        float(row["exact_minus_strong_gap"]) > TOLERANCE for row in gap_rows
    )
    novelty_gate = dict(config["claim_gates"])["optimization_novelty"]
    solver_novelty = positive_strong_gap_cells >= int(
        novelty_gate["minimum_positive_objective_gap_cells_over_strong_classical"]
    )
    performance = {
        "mean_nested_outer_rank_pair_qubo_exact_robust_bedroc": statistics.fmean(
            exact_values
        ),
        "mean_nested_outer_best_single_robust_bedroc": statistics.fmean(
            float(row["evaluation_robust_bedroc"])
            for row in nested_by_method["best_single_receptor"]
        ),
        "mean_nested_outer_linear_topk_robust_bedroc": statistics.fmean(
            float(row["evaluation_robust_bedroc"])
            for row in nested_by_method["bedroc_linear_topk"]
        ),
        "mean_nested_outer_bedroc_greedy_robust_bedroc": statistics.fmean(
            float(row["evaluation_robust_bedroc"])
            for row in nested_by_method["bedroc_nested_greedy"]
        ),
        "mean_gain_over_best_single": statistics.fmean(single_delta),
        "mean_gain_over_linear_topk": statistics.fmean(linear_delta),
        "mean_gain_over_nested_bedroc_greedy": statistics.fmean(greedy_delta),
        "mean_gain_over_strong_classical_same_qubo": statistics.fmean(strong_delta),
        "positive_gain_over_best_single_folds": sum(
            value > TOLERANCE for value in single_delta
        ),
    }
    decision = {
        "application_support_checks": application_checks,
        "transferred_qubo_application_supported": application_supported,
        "fresh_validation_authorized": application_supported,
        "locked_test_authorized": False,
        "optimization_novelty_supported": solver_novelty,
        "positive_objective_gap_cells_over_strong_classical": positive_strong_gap_cells,
        "quantum_hardware_authorized": False,
        "quantum_advantage_claim_authorized": False,
        "next_action": (
            "freeze one PPARD fresh-validation protocol without further outcome fitting"
            if application_supported
            else "stop PPARD outcome fitting and retain the result as a negative transfer test"
        ),
    }
    model_record = {
        "schema_version": "1.0",
        "status": "stage62_ppard_train240_final_model_frozen",
        "objective": config["objective"],
        "selected_k": final_k,
        "outer_k_selection": final_choice,
        "selected_subset": subset_name(
            final_subsets["rank_pair_qubo_exact"], receptor_ids
        ),
        "selected_subset_indices": list(final_subsets["rank_pair_qubo_exact"]),
        "selected_qubo_value": qubo_value(
            final_subsets["rank_pair_qubo_exact"],
            full_paths["singleton"],
            full_paths["complement"],
        ),
        "receptor_ids": receptor_ids,
        "singleton_coefficients": full_paths["singleton"].tolist(),
        "pair_complementarity_matrix": full_paths["complement"].tolist(),
        "exact_subsets_by_k": {
            str(key): subset_name(value, receptor_ids)
            for key, value in full_paths["exact"].items()
        },
        "strong_classical_subsets_by_k": {
            str(key): subset_name(value, receptor_ids)
            for key, value in full_paths["strong"].items()
        },
        "direct_greedy_subsets_by_k": {
            str(key): subset_name(value, receptor_ids)
            for key, value in full_paths["direct"].items()
        },
        "exact_state_count_k1_to_k6": int(full_paths["state_count"]),
        "coefficient_changes_after_stage60": 0,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
    }
    return {
        "input_audit": input_audit,
        "merged_score_rows": sorted(
            score_rows,
            key=lambda row: (
                SEED_IDS.index(row["seed_id"]),
                receptor_ids.index(row["receptor_id"]),
                ligand_ids.index(row["ligand_id"]),
            ),
        ),
        "inner_rows": inner_rows,
        "inner_selection_rows": inner_selection_rows,
        "outer_k_rows": outer_k_rows,
        "nested_rows": nested_rows,
        "gap_rows": gap_rows,
        "final_rows": final_rows,
        "final_choice": final_choice,
        "performance": performance,
        "decision": decision,
        "model_record": model_record,
    }


def write_report(path: Path, analysis: dict[str, Any], boundary: str) -> None:
    performance = analysis["performance"]
    decision = analysis["decision"]
    final_choice = analysis["final_choice"]
    lines = [
        "# Stage62 PPARD Train-240 frozen nested QUBO analysis",
        "",
        "## Nested outer-fold performance",
        "",
        "| Method | Mean robust BEDROC20 |",
        "|---|---:|",
        f"| Exact transferred QUBO | {performance['mean_nested_outer_rank_pair_qubo_exact_robust_bedroc']:.6f} |",
        f"| Best single receptor | {performance['mean_nested_outer_best_single_robust_bedroc']:.6f} |",
        f"| Linear Top-k | {performance['mean_nested_outer_linear_topk_robust_bedroc']:.6f} |",
        f"| Direct BEDROC greedy | {performance['mean_nested_outer_bedroc_greedy_robust_bedroc']:.6f} |",
        "",
        "## Frozen decisions",
        "",
        f"- Final one-standard-error k: {final_choice['selected_k']}.",
        f"- Application support gate: {'PASS' if decision['transferred_qubo_application_supported'] else 'NO-GO'}.",
        f"- Solver novelty gate: {'PASS' if decision['optimization_novelty_supported'] else 'NO-GO'}.",
        f"- Fresh validation authorized: {decision['fresh_validation_authorized']}.",
        "- Quantum hardware and quantum-advantage claims remain unauthorized.",
        "",
        boundary,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, dict(config["implementation"])["runner"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage62 implementation path differs")
    outputs = {key: root / str(value) for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for key, path in outputs.items() if key != "audit_json"):
        raise FileExistsError("Stage62 outputs exist; pass --overwrite")
    analysis = compute_analysis(config, root)
    write_csv(outputs["merged_scores_csv"], analysis["merged_score_rows"])
    write_csv(outputs["inner_k_metrics_csv"], analysis["inner_rows"])
    write_csv(outputs["inner_k_selection_csv"], analysis["inner_selection_rows"])
    write_csv(outputs["outer_k_metrics_csv"], analysis["outer_k_rows"])
    write_csv(outputs["nested_outer_metrics_csv"], analysis["nested_rows"])
    write_csv(outputs["objective_gap_cells_csv"], analysis["gap_rows"])
    write_csv(outputs["final_method_metrics_csv"], analysis["final_rows"])
    write_json(outputs["model_record_json"], analysis["model_record"])
    write_report(outputs["report_md"], analysis, config["interpretation_boundary"])
    analysis_fingerprint = canonical_sha256(
        {key: value for key, value in analysis.items() if key != "merged_score_rows"}
    )
    result = {
        "schema_version": "1.0",
        "status": "stage62_ppard_train240_frozen_nested_qubo_complete",
        "experiment_class": "prospective frozen-objective development application test",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation),
        "input_audit": analysis["input_audit"],
        "objective": config["objective"],
        "nested_cv": config["nested_cv"],
        "final_k_selection": analysis["final_choice"],
        "performance": analysis["performance"],
        "decision": analysis["decision"],
        "analysis_payload_sha256": analysis_fingerprint,
        "data_boundary": {
            "development_train_rows_read": 240,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {
            key: descriptor(root, path)
            for key, path in outputs.items()
            if key not in {"result_json", "audit_json"}
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
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
