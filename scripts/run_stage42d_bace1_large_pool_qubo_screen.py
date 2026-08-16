"""Screen the frozen robust functional QUBO objective on BACE1 Train-266."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import write_json  # noqa: F401 (deduped)
import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.diagnose_stage19e_cross_target_qubo_v2 import vectorized_bedroc
from scripts.run_stage05_mk14_method_gate import make_frozen_group_folds
from scripts.run_stage37_cross_target_robust_functional_qubo import fit_rank_transform


SEED_IDS = ("seed0", "seed1", "seed2")
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))




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
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verified(root: Path, descriptor_value: dict[str, Any]) -> Path:
    path = root / str(descriptor_value["path"])
    if not path.is_file() or sha256(path) != str(descriptor_value["sha256"]).upper():
        raise ValueError(f"Stage 42d input identity differs: {path}")
    return path


def build_score_cube(
    score_rows: list[dict[str, str]], ligand_ids: list[str], receptor_ids: list[str]
) -> np.ndarray:
    ligand_index = {value: index for index, value in enumerate(ligand_ids)}
    receptor_index = {value: index for index, value in enumerate(receptor_ids)}
    seed_index = {value: index for index, value in enumerate(SEED_IDS)}
    cube = np.full((3, len(ligand_ids), len(receptor_ids)), np.nan, dtype=float)
    seen: set[tuple[str, str, str]] = set()
    for row in score_rows:
        key = (row["seed_id"], row["ligand_id"], row["receptor_id"])
        if key in seen:
            raise ValueError(f"duplicate Stage 42c score key: {key}")
        seen.add(key)
        cube[
            seed_index[row["seed_id"]],
            ligand_index[row["ligand_id"]],
            receptor_index[row["receptor_id"]],
        ] = float(row["gpu_score"])
    if len(seen) != 27132 or not np.isfinite(cube).all():
        raise ValueError("Stage 42c score cube is incomplete")
    return cube


def rank_cube(scores: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(scores)
    for seed_index in range(scores.shape[0]):
        for receptor_index in range(scores.shape[2]):
            values = scores[seed_index, :, receptor_index]
            ranks[seed_index, :, receptor_index] = fit_rank_transform(
                values[train_mask], values
            )
    return ranks


class BitsetObjective:
    def __init__(self, favorable: np.ndarray, labels: np.ndarray, objective: dict[str, Any]):
        if favorable.shape[:2] != (3, len(labels)):
            raise ValueError("favorable-hit dimensions differ")
        self.objective = objective
        self.active_mask = self._mask(labels == 1)
        self.decoy_mask = self._mask(labels == 0)
        self.active_count = int(np.sum(labels == 1))
        self.decoy_count = int(np.sum(labels == 0))
        self.receptor_count = favorable.shape[2]
        self.hits = [
            [self._mask(favorable[seed, :, receptor]) for receptor in range(self.receptor_count)]
            for seed in range(3)
        ]
        self.cache: dict[tuple[int, ...], tuple[float, dict[str, float]]] = {}

    @staticmethod
    def _mask(values: np.ndarray) -> int:
        output = 0
        for index in np.flatnonzero(values):
            output |= 1 << int(index)
        return output

    def score(
        self, subset: tuple[int, ...], cache: bool = True
    ) -> tuple[float, dict[str, float]]:
        if cache:
            cached = self.cache.get(subset)
            if cached is not None:
                return cached
        once = [0, 0, 0]
        twice = [0, 0, 0]
        for receptor in subset:
            for seed in range(3):
                hit = self.hits[seed][receptor]
                twice[seed] |= once[seed] & hit
                once[seed] |= hit
        covered_majority = (once[0] & once[1]) | (once[0] & once[2]) | (once[1] & once[2])
        covered_all = once[0] & once[1] & once[2]
        double_majority = (twice[0] & twice[1]) | (twice[0] & twice[2]) | (twice[1] & twice[2])
        exposed = once[0] | once[1] | once[2]
        components = {
            "active_majority_seed_coverage": (covered_majority & self.active_mask).bit_count() / self.active_count,
            "active_all_seed_coverage": (covered_all & self.active_mask).bit_count() / self.active_count,
            "active_double_receptor_majority_seed_support": (double_majority & self.active_mask).bit_count() / self.active_count,
            "decoy_any_seed_exposure": (exposed & self.decoy_mask).bit_count() / self.decoy_count,
        }
        weights = self.objective["weights"]
        value = (
            float(weights["active_majority_seed_coverage"]) * components["active_majority_seed_coverage"]
            + float(weights["active_all_seed_coverage"]) * components["active_all_seed_coverage"]
            + float(weights["active_double_receptor_majority_seed_support"])
            * components["active_double_receptor_majority_seed_support"]
            - float(weights["decoy_any_seed_exposure"]) * components["decoy_any_seed_exposure"]
            - float(weights["receptor_cost"]) * len(subset) / int(self.objective["maximum_subset_size"])
        )
        result = (value, components)
        if cache:
            self.cache[subset] = result
        return result


def better(first: tuple[float, tuple[int, ...]], second: tuple[float, tuple[int, ...]]) -> bool:
    if first[0] > second[0] + TOLERANCE:
        return True
    if abs(first[0] - second[0]) <= TOLERANCE:
        return (len(first[1]), first[1]) < (len(second[1]), second[1])
    return False


def exact_search(
    scorer: BitsetObjective, minimum_size: int, maximum_size: int
) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]], int, float]:
    started = time.perf_counter()
    best = (-math.inf, tuple())
    best_by_size: dict[int, tuple[int, ...]] = {}
    state_count = 0
    for size in range(minimum_size, maximum_size + 1):
        size_best = (-math.inf, tuple())
        for subset in itertools.combinations(range(scorer.receptor_count), size):
            candidate = (scorer.score(subset, cache=False)[0], subset)
            state_count += 1
            if better(candidate, size_best):
                size_best = candidate
            if better(candidate, best):
                best = candidate
        best_by_size[size] = size_best[1]
    return best[1], best_by_size, state_count, time.perf_counter() - started


def direct_greedy(scorer: BitsetObjective, maximum_size: int) -> list[tuple[int, ...]]:
    current: tuple[int, ...] = tuple()
    path: list[tuple[int, ...]] = []
    for _ in range(maximum_size):
        selected = set(current)
        candidates = [tuple(sorted((*current, item))) for item in range(scorer.receptor_count) if item not in selected]
        current = min(candidates, key=lambda subset: (-scorer.score(subset)[0], subset))
        path.append(current)
    return path


def local_improve(
    scorer: BitsetObjective, start: tuple[int, ...], minimum_size: int, maximum_size: int
) -> tuple[int, ...]:
    current = start
    while True:
        selected = set(current)
        neighbors: set[tuple[int, ...]] = set()
        if len(current) < maximum_size:
            neighbors.update(tuple(sorted((*current, item))) for item in range(scorer.receptor_count) if item not in selected)
        if len(current) > minimum_size:
            neighbors.update(tuple(item for item in current if item != removed) for removed in current)
        for removed in current:
            neighbors.update(
                tuple(sorted((selected - {removed}) | {added}))
                for added in range(scorer.receptor_count)
                if added not in selected
            )
        improving = [subset for subset in neighbors if scorer.score(subset)[0] > scorer.score(current)[0] + TOLERANCE]
        if not improving:
            return current
        current = min(improving, key=lambda subset: (-scorer.score(subset)[0], len(subset), subset))


def strong_classical_search(
    scorer: BitsetObjective, minimum_size: int, maximum_size: int, beam_width: int
) -> tuple[tuple[int, ...], dict[str, int]]:
    starts: set[tuple[int, ...]] = {(index,) for index in range(scorer.receptor_count)}
    beam = sorted(starts, key=lambda subset: (-scorer.score(subset)[0], subset))[:beam_width]
    for _size in range(2, maximum_size + 1):
        expanded = {
            tuple(sorted((*subset, item)))
            for subset in beam
            for item in range(scorer.receptor_count)
            if item not in subset
        }
        beam = sorted(expanded, key=lambda subset: (-scorer.score(subset)[0], subset))[:beam_width]
        starts.update(beam)
    for initial in range(scorer.receptor_count):
        current = (initial,)
        starts.add(current)
        while len(current) < maximum_size:
            selected = set(current)
            candidates = [tuple(sorted((*current, item))) for item in range(scorer.receptor_count) if item not in selected]
            current = min(candidates, key=lambda subset: (-scorer.score(subset)[0], subset))
            starts.add(current)
    endpoints = {local_improve(scorer, start, minimum_size, maximum_size) for start in starts}
    best = min(endpoints, key=lambda subset: (-scorer.score(subset)[0], len(subset), subset))
    return best, {"start_state_count": len(starts), "local_endpoint_count": len(endpoints)}


def additive_selection(scorer: BitsetObjective, minimum_size: int, maximum_size: int) -> tuple[int, ...]:
    order = sorted(range(scorer.receptor_count), key=lambda item: (-scorer.score((item,))[0], item))
    candidates = [tuple(sorted(order[:size])) for size in range(minimum_size, maximum_size + 1)]
    return min(candidates, key=lambda subset: (-sum(scorer.score((item,))[0] for item in subset), len(subset), subset))


def random_search(
    scorer: BitsetObjective, minimum_size: int, maximum_size: int, samples_per_size: int, seed: int
) -> tuple[int, ...]:
    generator = random.Random(seed)
    candidates: set[tuple[int, ...]] = set()
    universe = list(range(scorer.receptor_count))
    for size in range(minimum_size, maximum_size + 1):
        target = min(samples_per_size, math.comb(scorer.receptor_count, size))
        while sum(len(value) == size for value in candidates) < target:
            candidates.add(tuple(sorted(generator.sample(universe, size))))
    return min(candidates, key=lambda subset: (-scorer.score(subset)[0], len(subset), subset))


def bedroc_metrics(ranks: np.ndarray, labels: np.ndarray, subset: tuple[int, ...], alpha: float) -> dict[str, float]:
    per_seed_rank = np.min(ranks[:, :, subset], axis=2)
    seed_values = vectorized_bedroc(per_seed_rank.T, labels, alpha)
    consensus = np.median(per_seed_rank, axis=0)
    primary = float(vectorized_bedroc(consensus[:, None], labels, alpha)[0])
    return {
        "primary_bedroc": primary,
        "mean_seed_bedroc": float(np.mean(seed_values)),
        "worst_seed_bedroc": float(np.min(seed_values)),
        "robust_bedroc_composite": float((primary + np.mean(seed_values) + np.min(seed_values)) / 3.0),
    }


def subset_name(subset: tuple[int, ...], receptor_ids: list[str]) -> str:
    return "+".join(receptor_ids[index] for index in subset)


def selection_row(
    method: str,
    subset: tuple[int, ...],
    scorer: BitsetObjective,
    ranks: np.ndarray,
    labels: np.ndarray,
    receptor_ids: list[str],
    alpha: float,
) -> dict[str, Any]:
    value, components = scorer.score(subset)
    return {
        "method": method,
        "selected_subset": subset_name(subset, receptor_ids),
        "subset_size": len(subset),
        "objective": value,
        **components,
        **bedroc_metrics(ranks, labels, subset, alpha),
    }


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 42d implementation identity differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    matrix_audit = read_json(inputs["stage42c_audit"])
    if matrix_audit.get("status") != "independent_stage42c_bace1_train266_unidock_matrix_audit_ok":
        raise ValueError("Stage 42c independent matrix audit did not pass")
    prior_objective = read_json(inputs["frozen_stage37_objective"])["objective"]
    if prior_objective != config["objective"]:
        raise ValueError("Stage 42d changed the previously frozen objective")
    outputs = {key: root / value for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage 42d outputs exist; pass --overwrite")

    ligands = read_csv(inputs["ligand_manifest"])
    receptors = read_csv(inputs["receptor_manifest"])
    score_rows = read_csv(inputs["scores"])
    ligand_ids = [row["ligand_id"] for row in ligands]
    receptor_ids = [row["conformer_id"] for row in receptors]
    labels = np.asarray([int(row["label"] == "active") for row in ligands], dtype=int)
    if len(ligands) != 266 or Counter(labels) != Counter({0: 133, 1: 133}) or len(receptors) != 34:
        raise ValueError("Stage 42d input dimensions differ")
    if {row["split"] for row in ligands} != {"train"}:
        raise ValueError("Stage 42d crossed the train boundary")
    scores = build_score_cube(score_rows, ligand_ids, receptor_ids)
    fold_by_ligand = make_frozen_group_folds(
        ligands, int(config["screen"]["outer_fold_count"]), int(config["screen"]["fold_seed"])
    )
    fold_rows = [
        {"ligand_id": row["ligand_id"], "label": row["label"], "split_group_id": row["split_group_id"], "outer_fold": fold_by_ligand[row["ligand_id"]]}
        for row in ligands
    ]

    objective = config["objective"]
    minimum_size = int(objective["minimum_subset_size"])
    maximum_size = int(objective["maximum_subset_size"])
    alpha = float(config["screen"]["bedroc_alpha"])
    fold_metrics: list[dict[str, Any]] = []
    for fold in range(int(config["screen"]["outer_fold_count"])):
        train_mask = np.asarray([fold_by_ligand[value] != fold for value in ligand_ids])
        holdout_mask = ~train_mask
        ranks = rank_cube(scores, train_mask)
        train_scorer = BitsetObjective(
            ranks[:, train_mask, :] <= float(objective["favorable_rank_fraction"]), labels[train_mask], objective
        )
        exact, _, state_count, elapsed = exact_search(train_scorer, minimum_size, maximum_size)
        classical, search_record = strong_classical_search(
            train_scorer, minimum_size, maximum_size, int(config["screen"]["classical_beam_width"])
        )
        holdout_scorer = BitsetObjective(
            ranks[:, holdout_mask, :] <= float(objective["favorable_rank_fraction"]), labels[holdout_mask], objective
        )
        exact_train = train_scorer.score(exact)[0]
        classical_train = train_scorer.score(classical)[0]
        exact_bedroc = bedroc_metrics(ranks[:, holdout_mask, :], labels[holdout_mask], exact, alpha)
        classical_bedroc = bedroc_metrics(ranks[:, holdout_mask, :], labels[holdout_mask], classical, alpha)
        fold_metrics.append({
            "outer_fold": fold,
            "train_ligand_count": int(train_mask.sum()),
            "holdout_ligand_count": int(holdout_mask.sum()),
            "state_count": state_count,
            "exact_subset": subset_name(exact, receptor_ids),
            "classical_subset": subset_name(classical, receptor_ids),
            "exact_subset_size": len(exact),
            "classical_subset_size": len(classical),
            "train_exact_objective": exact_train,
            "train_classical_objective": classical_train,
            "train_exact_minus_classical_gap": exact_train - classical_train,
            "holdout_exact_objective": holdout_scorer.score(exact)[0],
            "holdout_classical_objective": holdout_scorer.score(classical)[0],
            "holdout_objective_delta": holdout_scorer.score(exact)[0] - holdout_scorer.score(classical)[0],
            "holdout_exact_robust_bedroc": exact_bedroc["robust_bedroc_composite"],
            "holdout_classical_robust_bedroc": classical_bedroc["robust_bedroc_composite"],
            "holdout_robust_bedroc_delta": exact_bedroc["robust_bedroc_composite"] - classical_bedroc["robust_bedroc_composite"],
            "exact_enumeration_seconds": elapsed,
            **search_record,
        })
        print(json.dumps(fold_metrics[-1], sort_keys=True), flush=True)

    full_mask = np.ones(len(ligands), dtype=bool)
    full_ranks = rank_cube(scores, full_mask)
    full_scorer = BitsetObjective(
        full_ranks <= float(objective["favorable_rank_fraction"]), labels, objective
    )
    exact, exact_by_size, state_count, full_elapsed = exact_search(full_scorer, minimum_size, maximum_size)
    classical, classical_record = strong_classical_search(
        full_scorer, minimum_size, maximum_size, int(config["screen"]["classical_beam_width"])
    )
    direct_path = direct_greedy(full_scorer, maximum_size)
    direct = min(direct_path, key=lambda subset: (-full_scorer.score(subset)[0], len(subset), subset))
    additive = additive_selection(full_scorer, minimum_size, maximum_size)
    random_best = random_search(
        full_scorer, minimum_size, maximum_size, int(config["screen"]["random_samples_per_size"]), int(config["screen"]["random_seed"])
    )
    best_single = exact_by_size[1]
    all_receptors = tuple(range(len(receptor_ids)))
    methods = [
        ("exact_qubo_objective_optimum", exact),
        ("beam64_multistart_add_drop_swap", classical),
        ("direct_greedy", direct),
        ("exact_additive_singleton", additive),
        ("random_matched_budget_best", random_best),
        ("best_single_receptor", best_single),
        ("all_34_receptors", all_receptors),
    ]
    selection_metrics = [
        selection_row(method, subset, full_scorer, full_ranks, labels, receptor_ids, alpha)
        for method, subset in methods
    ]
    for size, subset in exact_by_size.items():
        selection_metrics.append(
            selection_row(f"exact_qubo_objective_k{size}", subset, full_scorer, full_ranks, labels, receptor_ids, alpha)
        )

    positive_gap_folds = sum(row["train_exact_minus_classical_gap"] > TOLERANCE for row in fold_metrics)
    mean_holdout_objective_delta = statistics.fmean(row["holdout_objective_delta"] for row in fold_metrics)
    mean_holdout_bedroc_delta = statistics.fmean(row["holdout_robust_bedroc_delta"] for row in fold_metrics)
    full_gap = full_scorer.score(exact)[0] - full_scorer.score(classical)[0]
    exact_row = selection_metrics[0]
    single_row = next(row for row in selection_metrics if row["method"] == "best_single_receptor")
    gate = config["support_gate"]
    checks = {
        "minimum_positive_train_gap_folds": positive_gap_folds >= int(gate["minimum_positive_train_gap_folds"]),
        "positive_full_data_exact_gap": full_gap >= float(gate["minimum_full_data_exact_gap"]),
        "nonnegative_mean_holdout_objective_delta": mean_holdout_objective_delta >= float(gate["minimum_mean_holdout_objective_delta"]),
        "nonnegative_mean_holdout_bedroc_delta": mean_holdout_bedroc_delta >= float(gate["minimum_mean_holdout_robust_bedroc_delta"]),
        "minimum_exact_over_single_bedroc_gain": exact_row["robust_bedroc_composite"] - single_row["robust_bedroc_composite"] >= float(gate["minimum_exact_over_single_robust_bedroc_gain"]),
    }
    supported = all(checks.values())
    decision = {
        "frozen_objective_supported_on_bace1": supported,
        "exact_differs_from_strong_classical_full_data": exact != classical,
        "positive_train_gap_fold_count": positive_gap_folds,
        "mean_holdout_objective_delta": mean_holdout_objective_delta,
        "mean_holdout_robust_bedroc_delta": mean_holdout_bedroc_delta,
        "full_data_exact_minus_classical_gap": full_gap,
        "exact_over_single_robust_bedroc_gain": exact_row["robust_bedroc_composite"] - single_row["robust_bedroc_composite"],
        "checks": checks,
        "sparse_auxiliary_qubo_encoding_authorized": supported,
        "fresh_validation_authorized": supported,
        "quantum_hardware_authorized": False,
        "objective_retuning_on_same_bace1_outcomes_authorized": False,
    }
    write_csv(outputs["fold_assignments_csv"], fold_rows)
    write_csv(outputs["fold_metrics_csv"], fold_metrics)
    write_csv(outputs["selection_metrics_csv"], selection_metrics)
    report_lines = [
        "# Stage42d BACE1 large-pool QUBO screen",
        "",
        "The Stage37 robust functional objective was reused without BACE1-specific weight tuning.",
        "",
        "| Method | k | Objective | Robust BEDROC20 | Subset |",
        "|---|---:|---:|---:|---|",
    ]
    for row in selection_metrics[:7]:
        report_lines.append(
            f"| {row['method']} | {row['subset_size']} | {row['objective']:.8f} | {row['robust_bedroc_composite']:.6f} | {row['selected_subset']} |"
        )
    report_lines.extend(["", "## Decision", "", f"Frozen objective support gate: **{'PASS' if supported else 'NO-GO'}**.", "", config["interpretation_boundary"], ""])
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report_lines), encoding="ascii")
    result = {
        "schema_version": "1.0",
        "status": "stage42d_bace1_large_pool_qubo_screen_complete",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, Path(__file__).resolve()),
        "input_statistics": {
            "receptor_count": len(receptors),
            "ligand_count": len(ligands),
            "active_count": int(labels.sum()),
            "decoy_count": int((labels == 0).sum()),
            "seed_count": 3,
            "full_state_count_k1_to_k6": state_count,
        },
        "objective": objective,
        "fold_metrics": fold_metrics,
        "full_data": {
            "exact_subset": subset_name(exact, receptor_ids),
            "classical_subset": subset_name(classical, receptor_ids),
            "exact_objective": full_scorer.score(exact)[0],
            "classical_objective": full_scorer.score(classical)[0],
            "exact_enumeration_seconds": full_elapsed,
            "classical_search": classical_record,
        },
        "decision": decision,
        "data_boundary": {
            "train_rows_read": 266,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {
            key: descriptor(root, path)
            for key, path in outputs.items()
            if key not in {"result_json"}
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage42d_bace1_large_pool_qubo_screen.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
