"""Evaluate frozen QUBO objectives and classical baselines on PPARA Train-374."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage42d_bace1_large_pool_qubo_screen import (
    BitsetObjective,
    additive_selection,
    bedroc_metrics,
    direct_greedy as coverage_direct_greedy,
    exact_search as coverage_exact_search,
    rank_cube,
    strong_classical_search as coverage_strong_search,
)
from scripts.run_stage42f_bace1_rank_sensitive_pair_qubo import (
    classical_by_size as pair_classical_by_size,
    direct_greedy_by_size as pair_direct_greedy_by_size,
    exact_by_size as pair_exact_by_size,
    pair_coefficients,
    qubo_value,
)


SEED_IDS = ("seed0", "seed1", "seed2")
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verified(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage 53 input identity differs: {path}")
    return path


def build_score_cube(
    rows: list[dict[str, str]], ligand_ids: list[str], receptor_ids: list[str]
) -> np.ndarray:
    ligand_index = {value: index for index, value in enumerate(ligand_ids)}
    receptor_index = {value: index for index, value in enumerate(receptor_ids)}
    seed_index = {value: index for index, value in enumerate(SEED_IDS)}
    cube = np.full((3, len(ligand_ids), len(receptor_ids)), np.nan, dtype=float)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if row["target_id"] != "PPARA":
            raise ValueError("Stage 53 requires the amended PPARA target ID")
        key = (row["seed_id"], row["ligand_id"], row["receptor_id"])
        if key in seen:
            raise ValueError(f"duplicate Stage 53 score key: {key}")
        seen.add(key)
        cube[
            seed_index[row["seed_id"]],
            ligand_index[row["ligand_id"]],
            receptor_index[row["receptor_id"]],
        ] = float(row["gpu_score"])
    expected = 3 * len(ligand_ids) * len(receptor_ids)
    if len(seen) != expected or not np.isfinite(cube).all():
        raise ValueError("Stage 53 score cube is incomplete")
    return cube


def subset_name(subset: tuple[int, ...], receptor_ids: list[str]) -> str:
    return "+".join(receptor_ids[index] for index in subset)


def robust_value(
    ranks: np.ndarray, labels: np.ndarray, subset: tuple[int, ...], alpha: float
) -> float:
    return bedroc_metrics(ranks, labels, subset, alpha)["robust_bedroc_composite"]


def metric_greedy_by_size(
    ranks: np.ndarray, labels: np.ndarray, maximum_size: int, alpha: float
) -> dict[int, tuple[int, ...]]:
    receptor_count = ranks.shape[2]
    current = min(
        ((index,) for index in range(receptor_count)),
        key=lambda subset: (-robust_value(ranks, labels, subset, alpha), subset),
    )
    output = {1: current}
    for size in range(2, maximum_size + 1):
        selected = set(current)
        candidates = [
            tuple(sorted((*current, added)))
            for added in range(receptor_count)
            if added not in selected
        ]
        current = min(
            candidates,
            key=lambda subset: (-robust_value(ranks, labels, subset, alpha), subset),
        )
        output[size] = current
    return output


def linear_by_size(
    ranks: np.ndarray, labels: np.ndarray, maximum_size: int, alpha: float
) -> dict[int, tuple[int, ...]]:
    values = [
        robust_value(ranks, labels, (index,), alpha)
        for index in range(ranks.shape[2])
    ]
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return {
        size: tuple(sorted(order[:size])) for size in range(1, maximum_size + 1)
    }


def random_by_size(
    ranks: np.ndarray,
    labels: np.ndarray,
    maximum_size: int,
    alpha: float,
    samples_per_size: int,
    seed: int,
) -> dict[int, tuple[int, ...]]:
    generator = random.Random(seed)
    receptor_count = ranks.shape[2]
    universe = list(range(receptor_count))
    output: dict[int, tuple[int, ...]] = {}
    for size in range(1, maximum_size + 1):
        target = min(samples_per_size, math.comb(receptor_count, size))
        candidates: set[tuple[int, ...]] = set()
        while len(candidates) < target:
            candidates.add(tuple(sorted(generator.sample(universe, size))))
        output[size] = min(
            candidates,
            key=lambda subset: (-robust_value(ranks, labels, subset, alpha), subset),
        )
    return output


def best_metric_subset(
    subsets: dict[int, tuple[int, ...]],
    ranks: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> tuple[int, ...]:
    return min(
        subsets.values(),
        key=lambda subset: (
            -robust_value(ranks, labels, subset, alpha),
            len(subset),
            subset,
        ),
    )


def best_pair_subset(
    subsets: dict[int, tuple[int, ...]],
    singleton: np.ndarray,
    complement: np.ndarray,
) -> tuple[int, ...]:
    return min(
        subsets.values(),
        key=lambda subset: (
            -qubo_value(subset, singleton, complement),
            len(subset),
            subset,
        ),
    )


def metric_record(
    scope: str,
    fold: int | str,
    method: str,
    subset: tuple[int, ...],
    train_ranks: np.ndarray,
    train_labels: np.ndarray,
    evaluation_ranks: np.ndarray,
    evaluation_labels: np.ndarray,
    receptor_ids: list[str],
    alpha: float,
    objective_value: float | None = None,
) -> dict[str, Any]:
    train_metrics = bedroc_metrics(train_ranks, train_labels, subset, alpha)
    evaluation_metrics = bedroc_metrics(
        evaluation_ranks, evaluation_labels, subset, alpha
    )
    return {
        "scope": scope,
        "outer_fold": fold,
        "method": method,
        "subset_size": len(subset),
        "selected_subset": subset_name(subset, receptor_ids),
        "train_objective": objective_value,
        "train_robust_bedroc": train_metrics["robust_bedroc_composite"],
        "evaluation_primary_bedroc": evaluation_metrics["primary_bedroc"],
        "evaluation_mean_seed_bedroc": evaluation_metrics["mean_seed_bedroc"],
        "evaluation_worst_seed_bedroc": evaluation_metrics["worst_seed_bedroc"],
        "evaluation_robust_bedroc": evaluation_metrics["robust_bedroc_composite"],
    }


def frozen_method_subsets(
    train_ranks: np.ndarray,
    train_labels: np.ndarray,
    objective: dict[str, Any],
    maximum_size: int,
    alpha: float,
    beam_width: int,
    random_samples: int,
    random_seed: int,
) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    receptor_count = train_ranks.shape[2]
    singleton, complement = pair_coefficients(train_ranks, train_labels, alpha)
    pair_exact, pair_states, pair_elapsed = pair_exact_by_size(
        receptor_count, maximum_size, singleton, complement
    )
    pair_classical, pair_records = pair_classical_by_size(
        receptor_count, maximum_size, singleton, complement, beam_width
    )
    pair_greedy = pair_direct_greedy_by_size(
        receptor_count, maximum_size, singleton, complement
    )

    favorable = train_ranks < float(objective["favorable_rank_fraction"])
    coverage = BitsetObjective(favorable, train_labels, objective)
    coverage_best, coverage_exact_by_size, coverage_states, coverage_elapsed = (
        coverage_exact_search(coverage, 1, maximum_size)
    )
    coverage_strong, coverage_search = coverage_strong_search(
        coverage, 1, maximum_size, beam_width
    )
    coverage_greedy_path = coverage_direct_greedy(coverage, maximum_size)
    coverage_additive = additive_selection(coverage, 1, maximum_size)

    linear = linear_by_size(train_ranks, train_labels, maximum_size, alpha)
    nested = metric_greedy_by_size(train_ranks, train_labels, maximum_size, alpha)
    random_subsets = random_by_size(
        train_ranks,
        train_labels,
        maximum_size,
        alpha,
        random_samples,
        random_seed,
    )
    methods = {
        "rank_pair_qubo_exact": best_pair_subset(
            pair_exact, singleton, complement
        ),
        "rank_pair_strong_classical": best_pair_subset(
            pair_classical, singleton, complement
        ),
        "rank_pair_direct_greedy": best_pair_subset(
            pair_greedy, singleton, complement
        ),
        "coverage_qubo_exact": coverage_best,
        "coverage_strong_classical": coverage_strong,
        "coverage_direct_greedy": min(
            coverage_greedy_path,
            key=lambda subset: (-coverage.score(subset)[0], len(subset), subset),
        ),
        "coverage_linear_additive": coverage_additive,
        "bedroc_linear_topk": best_metric_subset(
            linear, train_ranks, train_labels, alpha
        ),
        "bedroc_nested_greedy": best_metric_subset(
            nested, train_ranks, train_labels, alpha
        ),
        "bedroc_random_search": best_metric_subset(
            random_subsets, train_ranks, train_labels, alpha
        ),
        "best_single_receptor": linear[1],
        "all_receptors": tuple(range(receptor_count)),
    }
    method_objectives: dict[str, float | None] = {
        "rank_pair_qubo_exact": qubo_value(
            methods["rank_pair_qubo_exact"], singleton, complement
        ),
        "rank_pair_strong_classical": qubo_value(
            methods["rank_pair_strong_classical"], singleton, complement
        ),
        "rank_pair_direct_greedy": qubo_value(
            methods["rank_pair_direct_greedy"], singleton, complement
        ),
        "coverage_qubo_exact": coverage.score(methods["coverage_qubo_exact"])[0],
        "coverage_strong_classical": coverage.score(
            methods["coverage_strong_classical"]
        )[0],
        "coverage_direct_greedy": coverage.score(
            methods["coverage_direct_greedy"]
        )[0],
        "coverage_linear_additive": coverage.score(
            methods["coverage_linear_additive"]
        )[0],
    }
    diagnostics = {
        "pair_exact_by_size": pair_exact,
        "pair_classical_by_size": pair_classical,
        "pair_greedy_by_size": pair_greedy,
        "pair_singleton": singleton,
        "pair_complement": complement,
        "pair_state_count": pair_states,
        "pair_exact_seconds": pair_elapsed,
        "pair_classical_records": pair_records,
        "coverage_exact_by_size": coverage_exact_by_size,
        "coverage_greedy_by_size": {
            size: subset for size, subset in enumerate(coverage_greedy_path, start=1)
        },
        "coverage_scorer": coverage,
        "coverage_state_count": coverage_states,
        "coverage_exact_seconds": coverage_elapsed,
        "coverage_search": coverage_search,
        "linear_by_size": linear,
        "nested_by_size": nested,
        "random_by_size": random_subsets,
        "method_objectives": method_objectives,
    }
    return methods, diagnostics


def write_report(
    path: Path,
    full_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    boundary: str,
) -> None:
    lines = [
        "# Stage53 PPARA large-pool QUBO transfer",
        "",
        "| Method | k | Full robust BEDROC20 |",
        "|---|---:|---:|",
    ]
    for row in sorted(
        full_rows, key=lambda value: -float(value["evaluation_robust_bedroc"])
    ):
        lines.append(
            f"| {row['method']} | {row['subset_size']} | "
            f"{float(row['evaluation_robust_bedroc']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Frozen QUBO application transfer: **{'PASS' if decision['frozen_qubo_application_transfer_supported'] else 'NO-GO'}**.",
            "",
            f"Solver novelty over strong classical search: **{'PASS' if decision['solver_novelty_detected'] else 'NO-GO'}**.",
            "",
            boundary,
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
        raise ValueError("Stage 53 implementation identity differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    amendment = read_json(inputs["stage52c_amendment"])
    if amendment["status"] != "stage52c_ppara_target_id_amendment_ok" or not amendment[
        "decision"
    ]["stage53_train_only_method_comparison_authorized"]:
        raise ValueError("Stage 52c did not authorize Stage 53")
    frozen_coverage = read_json(inputs["stage37_objective_config"])["objective"]
    frozen_pair = read_json(inputs["stage42f_objective_config"])["objective"]
    if frozen_coverage != config["objectives"]["coverage_qubo"]:
        raise ValueError("Stage 53 changed the frozen coverage objective")
    if frozen_pair != config["objectives"]["rank_pair_qubo"]:
        raise ValueError("Stage 53 changed the frozen rank-pair objective")
    outputs = {key: root / value for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage 53 outputs exist; pass --overwrite")

    ligands = read_csv(inputs["ligand_manifest"])
    receptors = read_csv(inputs["receptor_manifest"])
    score_rows = read_csv(inputs["corrected_scores"])
    ligand_ids = [row["ligand_id"] for row in ligands]
    receptor_ids = [row["conformer_id"] for row in receptors]
    labels = np.asarray([int(row["label"] == "active") for row in ligands], dtype=int)
    scores = build_score_cube(score_rows, ligand_ids, receptor_ids)
    folds = make_frozen_group_folds(
        ligands,
        int(config["screen"]["outer_fold_count"]),
        int(config["screen"]["fold_seed"]),
    )
    fold_assignments = [
        {
            "ligand_id": row["ligand_id"],
            "label": row["label"],
            "split_group_id": row["split_group_id"],
            "outer_fold": folds[row["ligand_id"]],
        }
        for row in ligands
    ]
    maximum_size = int(config["screen"]["maximum_subset_size"])
    alpha = float(config["screen"]["bedroc_alpha"])
    beam_width = int(config["screen"]["classical_beam_width"])
    random_samples = int(config["screen"]["random_samples_per_size"])
    random_seed = int(config["screen"]["random_seed"])
    selection_rows: list[dict[str, Any]] = []
    landscape_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for fold in range(int(config["screen"]["outer_fold_count"])):
        train_mask = np.asarray([folds[value] != fold for value in ligand_ids])
        holdout_mask = ~train_mask
        ranks = rank_cube(scores, train_mask)
        train_ranks = ranks[:, train_mask, :]
        holdout_ranks = ranks[:, holdout_mask, :]
        train_labels = labels[train_mask]
        holdout_labels = labels[holdout_mask]
        methods, diagnostics = frozen_method_subsets(
            train_ranks,
            train_labels,
            config["objectives"]["coverage_qubo"],
            maximum_size,
            alpha,
            beam_width,
            random_samples,
            random_seed + fold,
        )
        for method, subset in methods.items():
            selection_rows.append(
                metric_record(
                    "outer_holdout",
                    fold,
                    method,
                    subset,
                    train_ranks,
                    train_labels,
                    holdout_ranks,
                    holdout_labels,
                    receptor_ids,
                    alpha,
                    diagnostics["method_objectives"].get(method),
                )
            )
        for size in range(1, maximum_size + 1):
            pair_singleton = diagnostics["pair_singleton"]
            pair_complement = diagnostics["pair_complement"]
            coverage = diagnostics["coverage_scorer"]
            for method, subset in (
                ("rank_pair_qubo_exact", diagnostics["pair_exact_by_size"][size]),
                ("rank_pair_strong_classical", diagnostics["pair_classical_by_size"][size]),
                ("rank_pair_direct_greedy", diagnostics["pair_greedy_by_size"][size]),
                ("coverage_qubo_exact", diagnostics["coverage_exact_by_size"][size]),
                ("coverage_direct_greedy", diagnostics["coverage_greedy_by_size"][size]),
                ("bedroc_linear_topk", diagnostics["linear_by_size"][size]),
                ("bedroc_nested_greedy", diagnostics["nested_by_size"][size]),
                ("bedroc_random_search", diagnostics["random_by_size"][size]),
            ):
                objective_value = None
                if method.startswith("rank_pair"):
                    objective_value = qubo_value(
                        subset, pair_singleton, pair_complement
                    )
                elif method.startswith("coverage"):
                    objective_value = coverage.score(subset)[0]
                landscape_rows.append(
                    metric_record(
                        "outer_holdout_fixed_k",
                        fold,
                        method,
                        subset,
                        train_ranks,
                        train_labels,
                        holdout_ranks,
                        holdout_labels,
                        receptor_ids,
                        alpha,
                        objective_value,
                    )
                )
        print(json.dumps({"outer_fold": fold, "status": "complete"}), flush=True)

    full_ranks = rank_cube(scores, np.ones(len(ligands), dtype=bool))
    full_methods, full_diagnostics = frozen_method_subsets(
        full_ranks,
        labels,
        config["objectives"]["coverage_qubo"],
        maximum_size,
        alpha,
        beam_width,
        random_samples,
        random_seed,
    )
    full_rows = [
        metric_record(
            "full_data",
            "full",
            method,
            subset,
            full_ranks,
            labels,
            full_ranks,
            labels,
            receptor_ids,
            alpha,
            full_diagnostics["method_objectives"].get(method),
        )
        for method, subset in full_methods.items()
    ]
    selection_rows.extend(full_rows)

    def fold_values(method: str) -> list[float]:
        return [
            float(row["evaluation_robust_bedroc"])
            for row in selection_rows
            if row["scope"] == "outer_holdout" and row["method"] == method
        ]

    rank_values = fold_values("rank_pair_qubo_exact")
    single_values = fold_values("best_single_receptor")
    linear_values = fold_values("bedroc_linear_topk")
    nested_values = fold_values("bedroc_nested_greedy")
    rank_strong_values = fold_values("rank_pair_strong_classical")
    coverage_values = fold_values("coverage_qubo_exact")
    coverage_strong_values = fold_values("coverage_strong_classical")
    rank_minus_single = [a - b for a, b in zip(rank_values, single_values, strict=True)]
    rank_minus_linear = [a - b for a, b in zip(rank_values, linear_values, strict=True)]
    rank_minus_nested = [a - b for a, b in zip(rank_values, nested_values, strict=True)]
    rank_minus_strong = [
        a - b for a, b in zip(rank_values, rank_strong_values, strict=True)
    ]
    coverage_minus_strong = [
        a - b
        for a, b in zip(coverage_values, coverage_strong_values, strict=True)
    ]
    fixed_rank_rows = [
        row
        for row in landscape_rows
        if row["method"] == "rank_pair_qubo_exact"
    ]
    fixed_rank_strong = {
        (row["outer_fold"], row["subset_size"]): row
        for row in landscape_rows
        if row["method"] == "rank_pair_strong_classical"
    }
    fixed_rank_greedy = {
        (row["outer_fold"], row["subset_size"]): row
        for row in landscape_rows
        if row["method"] == "rank_pair_direct_greedy"
    }
    positive_over_strong = sum(
        float(row["train_objective"])
        - float(fixed_rank_strong[(row["outer_fold"], row["subset_size"])]["train_objective"])
        > TOLERANCE
        for row in fixed_rank_rows
    )
    positive_over_greedy = sum(
        float(row["train_objective"])
        - float(fixed_rank_greedy[(row["outer_fold"], row["subset_size"])]["train_objective"])
        > TOLERANCE
        for row in fixed_rank_rows
    )
    gate = config["support_gate"]
    checks = {
        "minimum_mean_holdout_over_single": statistics.fmean(rank_minus_single)
        >= float(gate["minimum_mean_holdout_over_single"]),
        "minimum_positive_holdout_over_single_folds": sum(
            value > 0 for value in rank_minus_single
        )
        >= int(gate["minimum_positive_holdout_over_single_folds"]),
        "nonnegative_mean_holdout_over_linear": statistics.fmean(rank_minus_linear)
        >= float(gate["minimum_mean_holdout_over_linear"]),
        "nonnegative_mean_holdout_over_nested_greedy": statistics.fmean(
            rank_minus_nested
        )
        >= float(gate["minimum_mean_holdout_over_nested_greedy"]),
    }
    application_supported = all(checks.values())
    solver_novelty = positive_over_strong >= int(
        gate["minimum_positive_objective_gap_cells_over_strong_classical"]
    )
    full_by_method = {row["method"]: row for row in full_rows}
    decision = {
        "frozen_qubo_application_transfer_supported": application_supported,
        "solver_novelty_detected": solver_novelty,
        "checks": checks,
        "mean_holdout_rank_pair_qubo": statistics.fmean(rank_values),
        "mean_holdout_rank_pair_minus_single": statistics.fmean(rank_minus_single),
        "mean_holdout_rank_pair_minus_linear": statistics.fmean(rank_minus_linear),
        "mean_holdout_rank_pair_minus_nested_greedy": statistics.fmean(
            rank_minus_nested
        ),
        "mean_holdout_rank_pair_minus_strong_classical": statistics.fmean(
            rank_minus_strong
        ),
        "mean_holdout_coverage_minus_strong_classical": statistics.fmean(
            coverage_minus_strong
        ),
        "positive_holdout_over_single_fold_count": sum(
            value > 0 for value in rank_minus_single
        ),
        "positive_rank_pair_objective_gap_cells_over_direct_greedy": positive_over_greedy,
        "positive_rank_pair_objective_gap_cells_over_strong_classical": positive_over_strong,
        "full_data_rank_pair_over_single_bedroc_gain": float(
            full_by_method["rank_pair_qubo_exact"]["evaluation_robust_bedroc"]
        )
        - float(full_by_method["best_single_receptor"]["evaluation_robust_bedroc"]),
        "fresh_validation_authorized": False,
        "new_independent_preregistration_justified": application_supported,
        "same_data_weight_retuning_authorized": False,
        "quantum_hardware_authorized": False,
    }

    write_csv(outputs["fold_assignments_csv"], fold_assignments)
    write_csv(outputs["selection_metrics_csv"], selection_rows)
    write_csv(outputs["fixed_k_landscape_csv"], landscape_rows)
    write_report(outputs["report_md"], full_rows, decision, config["interpretation_boundary"])
    result = {
        "schema_version": "1.0",
        "status": "stage53_ppara_large_pool_qubo_transfer_complete",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation),
        "objectives": config["objectives"],
        "input_statistics": {
            "receptor_count": len(receptor_ids),
            "ligand_count": len(ligand_ids),
            "active_count": int(labels.sum()),
            "decoy_count": int((labels == 0).sum()),
            "seed_count": 3,
            "state_count_k1_to_k6": full_diagnostics["pair_state_count"],
        },
        "full_data_methods": {
            row["method"]: {
                "selected_subset": row["selected_subset"],
                "subset_size": row["subset_size"],
                "robust_bedroc": row["evaluation_robust_bedroc"],
            }
            for row in full_rows
        },
        "decision": decision,
        "runtime_seconds": time.perf_counter() - started,
        "data_boundary": {
            "train_rows_read": len(ligand_ids),
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {
            key: descriptor(root, path)
            for key, path in outputs.items()
            if key != "result_json"
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
