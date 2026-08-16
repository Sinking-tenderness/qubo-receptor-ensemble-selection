"""Test BEDROC-aligned continuous and rank-bin QUBO set objectives."""

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
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from scripts.run_stage42d_bace1_large_pool_qubo_screen import bedroc_metrics, rank_cube
from scripts.run_stage64_cross_target_uncertainty_shrunk_qubo import (
    K_VALUES,
    TOLERANCE,
    load_target,
    pairwise_jaccard,
)
from scripts.run_stage66_cross_target_auxiliary_coverage_qubo import (
    beam_swap_by_size,
    direct_greedy_by_size,
    slack_weights,
)


SOLVER_BEAM = "objective_beam_swap"
SOLVER_GREEDY = "same_objective_direct_greedy"
SOLVER_PAIR_OFF = "pair_off_baseline"
CONTINUOUS_ID = "continuous_reference"


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
        raise ValueError(f"Stage67 frozen identity differs: {path}")
    return path


def subset_name(subset: tuple[int, ...], receptor_ids: list[str]) -> str:
    return "+".join(receptor_ids[index] for index in subset)


class RankUtilityObjective:
    """Mean seed-wise active-minus-decoy exponential best-rank utility."""

    def __init__(
        self,
        ranks: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        alpha: float,
        bin_count: int | None,
    ):
        utilities = np.exp(-float(alpha) * ranks[:, mask, :])
        if bin_count is not None:
            utilities = np.floor(float(bin_count) * utilities + 1e-12) / float(
                bin_count
            )
        self.utilities = utilities
        self.labels = labels[mask]
        self.active = self.labels == 1
        self.decoy = self.labels == 0
        if not int(np.sum(self.active)) or not int(np.sum(self.decoy)):
            raise ValueError("Stage67 objective requires both label classes")
        self.receptor_count = ranks.shape[2]
        self.bin_count = bin_count
        self.cache: dict[tuple[int, ...], tuple[float, dict[str, float]]] = {}

    def score(self, subset: tuple[int, ...]) -> tuple[float, dict[str, float]]:
        subset = tuple(sorted(subset))
        cached = self.cache.get(subset)
        if cached is not None:
            return cached
        best = np.max(self.utilities[:, :, subset], axis=2)
        active_per_seed = np.mean(best[:, self.active], axis=1)
        decoy_per_seed = np.mean(best[:, self.decoy], axis=1)
        discrimination = active_per_seed - decoy_per_seed
        value = float(np.mean(discrimination))
        components = {
            "mean_active_utility": float(np.mean(active_per_seed)),
            "mean_decoy_utility": float(np.mean(decoy_per_seed)),
            "worst_seed_discrimination": float(np.min(discrimination)),
            "mean_seed_discrimination": value,
        }
        result = (value, components)
        self.cache[subset] = result
        return result


def objective_id(bin_count: int | None) -> str:
    return CONTINUOUS_ID if bin_count is None else f"rankbin_b{bin_count}"


def metric_row(
    target_id: str,
    outer_fold: int,
    bin_count: int | None,
    solver_id: str,
    subset_size: int,
    subset: tuple[int, ...],
    receptor_ids: list[str],
    train_scorer: RankUtilityObjective,
    holdout_scorer: RankUtilityObjective,
    train_continuous: RankUtilityObjective,
    holdout_continuous: RankUtilityObjective,
    ranks: np.ndarray,
    labels: np.ndarray,
    holdout_mask: np.ndarray,
    alpha: float,
    search_record: dict[str, int],
) -> dict[str, Any]:
    train_value, train_components = train_scorer.score(subset)
    holdout_value, holdout_components = holdout_scorer.score(subset)
    train_continuous_value = train_continuous.score(subset)[0]
    holdout_continuous_value = holdout_continuous.score(subset)[0]
    metrics = bedroc_metrics(
        ranks[:, holdout_mask, :], labels[holdout_mask], subset, alpha
    )
    return {
        "target_id": target_id,
        "outer_fold": outer_fold,
        "objective_id": objective_id(bin_count),
        "bin_count": "continuous" if bin_count is None else bin_count,
        "solver_id": solver_id,
        "subset_size": subset_size,
        "selected_subset": subset_name(subset, receptor_ids),
        "train_objective": train_value,
        "train_continuous_objective": train_continuous_value,
        "train_quantization_error": train_value - train_continuous_value,
        "train_mean_active_utility": train_components["mean_active_utility"],
        "train_mean_decoy_utility": train_components["mean_decoy_utility"],
        "train_worst_seed_discrimination": train_components[
            "worst_seed_discrimination"
        ],
        "holdout_objective": holdout_value,
        "holdout_continuous_objective": holdout_continuous_value,
        "holdout_quantization_error": holdout_value - holdout_continuous_value,
        "holdout_worst_seed_discrimination": holdout_components[
            "worst_seed_discrimination"
        ],
        "holdout_primary_bedroc": metrics["primary_bedroc"],
        "holdout_mean_seed_bedroc": metrics["mean_seed_bedroc"],
        "holdout_worst_seed_bedroc": metrics["worst_seed_bedroc"],
        "holdout_robust_bedroc": metrics["robust_bedroc_composite"],
        "search_start_state_count": search_record["start_state_count"],
        "search_local_endpoint_count": search_record["local_endpoint_count"],
    }


def target_and_global_summaries(
    rows: list[dict[str, Any]],
    objective_ids: list[str],
    target_order: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_off = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): float(
            row["holdout_robust_bedroc"]
        )
        for row in rows
        if row["solver_id"] == SOLVER_PAIR_OFF
    }
    greedy = {
        (
            row["target_id"],
            int(row["outer_fold"]),
            row["objective_id"],
            int(row["subset_size"]),
        ): row
        for row in rows
        if row["solver_id"] == SOLVER_GREEDY
    }
    continuous = {
        (row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row
        for row in rows
        if row["solver_id"] == SOLVER_BEAM
        and row["objective_id"] == CONTINUOUS_ID
    }
    target_rows: list[dict[str, Any]] = []
    for target_id in target_order:
        for current_id in objective_ids:
            selected = [
                row
                for row in rows
                if row["target_id"] == target_id
                and row["objective_id"] == current_id
                and row["solver_id"] == SOLVER_BEAM
                and int(row["subset_size"]) >= 2
            ]
            pair_gains = [
                float(row["holdout_robust_bedroc"])
                - pair_off[
                    (target_id, int(row["outer_fold"]), int(row["subset_size"]))
                ]
                for row in selected
            ]
            greedy_rows = [
                greedy[
                    (
                        target_id,
                        int(row["outer_fold"]),
                        current_id,
                        int(row["subset_size"]),
                    )
                ]
                for row in selected
            ]
            continuous_rows = [
                continuous[
                    (target_id, int(row["outer_fold"]), int(row["subset_size"]))
                ]
                for row in selected
            ]
            target_rows.append(
                {
                    "target_id": target_id,
                    "objective_id": current_id,
                    "fixed_k_cell_count": len(selected),
                    "mean_fixed_k_holdout_robust_bedroc": statistics.fmean(
                        float(row["holdout_robust_bedroc"]) for row in selected
                    ),
                    "mean_gain_over_pair_off": statistics.fmean(pair_gains),
                    "minimum_fold_k_gain_over_pair_off": min(pair_gains),
                    "nonnegative_fold_k_gain_over_pair_off_count": sum(
                        value >= -TOLERANCE for value in pair_gains
                    ),
                    "mean_gain_over_same_objective_greedy": statistics.fmean(
                        float(row["holdout_robust_bedroc"])
                        - float(greedy_row["holdout_robust_bedroc"])
                        for row, greedy_row in zip(selected, greedy_rows)
                    ),
                    "minimum_train_objective_gain_over_greedy": min(
                        float(row["train_objective"])
                        - float(greedy_row["train_objective"])
                        for row, greedy_row in zip(selected, greedy_rows)
                    ),
                    "selection_difference_count_vs_greedy": sum(
                        row["selected_subset"] != greedy_row["selected_subset"]
                        for row, greedy_row in zip(selected, greedy_rows)
                    ),
                    "mean_absolute_train_quantization_error": statistics.fmean(
                        abs(float(row["train_quantization_error"])) for row in selected
                    ),
                    "mean_subset_jaccard_vs_continuous": statistics.fmean(
                        len(
                            set(str(row["selected_subset"]).split("+"))
                            & set(str(reference["selected_subset"]).split("+"))
                        )
                        / len(
                            set(str(row["selected_subset"]).split("+"))
                            | set(str(reference["selected_subset"]).split("+"))
                        )
                        for row, reference in zip(selected, continuous_rows)
                    ),
                    "mean_holdout_bedroc_gap_vs_continuous": statistics.fmean(
                        float(row["holdout_robust_bedroc"])
                        - float(reference["holdout_robust_bedroc"])
                        for row, reference in zip(selected, continuous_rows)
                    ),
                    "mean_fixed_k_selection_jaccard": statistics.fmean(
                        pairwise_jaccard(
                            [
                                str(row["selected_subset"])
                                for row in selected
                                if int(row["subset_size"]) == subset_size
                            ]
                        )
                        for subset_size in range(2, 7)
                    ),
                }
            )
    lookup = {
        (row["target_id"], row["objective_id"]): row for row in target_rows
    }
    global_rows: list[dict[str, Any]] = []
    for order, current_id in enumerate(objective_ids):
        selected = [lookup[(target_id, current_id)] for target_id in target_order]
        gains = [float(row["mean_gain_over_pair_off"]) for row in selected]
        global_rows.append(
            {
                "objective_order": order,
                "objective_id": current_id,
                "mean_target_gain_over_pair_off": statistics.fmean(gains),
                "worst_target_gain_over_pair_off": min(gains),
                "nonnegative_target_count_over_pair_off": sum(
                    value >= -TOLERANCE for value in gains
                ),
                "positive_target_count_over_pair_off": sum(
                    value > TOLERANCE for value in gains
                ),
                "mean_target_gain_over_same_objective_greedy": statistics.fmean(
                    float(row["mean_gain_over_same_objective_greedy"])
                    for row in selected
                ),
                "minimum_train_objective_gain_over_greedy": min(
                    float(row["minimum_train_objective_gain_over_greedy"])
                    for row in selected
                ),
                "selection_difference_count_vs_greedy": sum(
                    int(row["selection_difference_count_vs_greedy"])
                    for row in selected
                ),
                "mean_absolute_train_quantization_error": statistics.fmean(
                    float(row["mean_absolute_train_quantization_error"])
                    for row in selected
                ),
                "mean_subset_jaccard_vs_continuous": statistics.fmean(
                    float(row["mean_subset_jaccard_vs_continuous"])
                    for row in selected
                ),
                "mean_holdout_bedroc_gap_vs_continuous": statistics.fmean(
                    float(row["mean_holdout_bedroc_gap_vs_continuous"])
                    for row in selected
                ),
                "mean_target_selection_jaccard": statistics.fmean(
                    float(row["mean_fixed_k_selection_jaccard"])
                    for row in selected
                ),
            }
        )
    return target_rows, global_rows


def incidence_hex(indices: np.ndarray) -> str:
    value = 0
    for index in indices:
        value |= 1 << int(index)
    return format(value, "x")


def compact_rankbin_model(
    target_id: str,
    target: dict[str, Any],
    alpha: float,
    bin_count: int,
    reference_k: int,
    beam_width: int,
    certificate_count: int,
    certificate_seed: int,
    cardinality_penalty: float,
    constraint_penalty: float,
) -> tuple[dict[str, Any], float]:
    mask = np.ones(len(target["ligand_ids"]), dtype=bool)
    ranks = rank_cube(target["scores"], mask)
    scorer = RankUtilityObjective(ranks, target["labels"], mask, alpha, bin_count)
    selected, search_records = beam_swap_by_size(scorer, max(K_VALUES), beam_width)
    levels = np.floor(
        float(bin_count) * np.exp(-float(alpha) * ranks) + 1e-12
    ).astype(int)
    active_count = int(np.sum(target["labels"] == 1))
    decoy_count = int(np.sum(target["labels"] == 0))
    states: list[dict[str, Any]] = []
    slack_count = 0
    implication_edges = 0
    raw_quadratic_terms = math.comb(len(target["receptor_ids"]), 2)
    for seed_index in range(levels.shape[0]):
        for ligand_index, (ligand_id, label) in enumerate(
            zip(target["ligand_ids"], target["labels"])
        ):
            for level in range(1, bin_count + 1):
                incidence = np.flatnonzero(levels[seed_index, ligand_index, :] >= level)
                if not len(incidence):
                    continue
                is_active = int(label) == 1
                state_slack_count = len(slack_weights(len(incidence))) if is_active else 0
                slack_count += state_slack_count
                if is_active:
                    variable_count = 1 + state_slack_count + len(incidence)
                    raw_quadratic_terms += math.comb(variable_count, 2)
                else:
                    implication_edges += len(incidence)
                    raw_quadratic_terms += len(incidence)
                states.append(
                    {
                        "state_id": f"s{seed_index}_l{ligand_index}_b{level}",
                        "seed_index": seed_index,
                        "ligand_id": ligand_id,
                        "label": "active" if is_active else "decoy",
                        "utility_level": level,
                        "utility_threshold": level / bin_count,
                        "objective_weight": 1.0
                        / (3.0 * bin_count * (active_count if is_active else decoy_count)),
                        "incidence_hex": incidence_hex(incidence),
                        "incidence_count": len(incidence),
                        "slack_bit_count": state_slack_count,
                    }
                )
    active_states = sum(state["label"] == "active" for state in states)
    decoy_states = len(states) - active_states
    selected_subset = selected[reference_k]

    def factorized_energy(subset: tuple[int, ...]) -> float:
        selected_mask = sum(1 << int(index) for index in subset)
        value = cardinality_penalty * (len(subset) - reference_k) ** 2
        for state in states:
            exposed = bool(int(state["incidence_hex"], 16) & selected_mask)
            if state["label"] == "active":
                value -= float(state["objective_weight"]) * int(exposed)
            else:
                value += float(state["objective_weight"]) * int(exposed)
        return float(value)

    generator = random.Random(certificate_seed)
    subsets = {selected_subset}
    for _ in range(certificate_count):
        subsets.add(
            tuple(
                sorted(
                    generator.sample(
                        range(len(target["receptor_ids"])), reference_k
                    )
                )
            )
        )
    residual = max(
        abs(factorized_energy(subset) + scorer.score(subset)[0])
        for subset in subsets
    )
    core = {
        "target_id": target_id,
        "bin_count": bin_count,
        "reference_k": reference_k,
        "receptor_ids": target["receptor_ids"],
        "states": states,
        "penalties": {
            "cardinality_penalty": cardinality_penalty,
            "constraint_penalty": constraint_penalty,
        },
        "factorized_qubo": {
            "cardinality": "A*(sum_i x_i-k)^2",
            "active_or": "P*(y+s-sum_i h_i*x_i)^2-weight*y",
            "decoy_or": "P*x_i*(1-z)+weight*z for each incidence edge",
            "expansion_status": "exact factorized quadratic form; sparse expansion deferred",
        },
    }
    record = {
        **core,
        "compact_model_sha256": canonical_sha256(core),
        "selected_subset": subset_name(selected_subset, target["receptor_ids"]),
        "selected_objective": scorer.score(selected_subset)[0],
        "selected_factorized_energy": factorized_energy(selected_subset),
        "equivalence_max_residual": residual,
        "equivalence_state_count": len(subsets),
        "search_record": search_records[reference_k],
        "variable_counts": {
            "receptor_x": len(target["receptor_ids"]),
            "active_y": active_states,
            "decoy_z": decoy_states,
            "active_slack": slack_count,
            "total": len(target["receptor_ids"])
            + active_states
            + decoy_states
            + slack_count,
        },
        "implication_edge_count": implication_edges,
        "raw_quadratic_term_count_before_coalescing": raw_quadratic_terms,
    }
    return record, residual


def build_model_record(
    config: dict[str, Any],
    targets: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], float]:
    development = config["development"]
    encoding = config["qubo_encoding"]
    models: dict[str, Any] = {}
    maximum_residual = 0.0
    for target_index, target_id in enumerate(development["target_order"]):
        record, residual = compact_rankbin_model(
            target_id,
            targets[target_id],
            float(development["bedroc_alpha"]),
            int(encoding["reference_bin_count"]),
            int(encoding["reference_model_k"]),
            int(development["classical_beam_width"]),
            int(encoding["certificate_random_subset_count"]),
            int(encoding["certificate_seed_base"]) + target_index,
            float(encoding["cardinality_penalty"]),
            float(encoding["constraint_penalty"]),
        )
        models[target_id] = record
        maximum_residual = max(maximum_residual, residual)
    return {
        "schema_version": "1.0",
        "algorithm_id": "bedroc20-exponential-rankbin-auxiliary-qubo-v1",
        "status": "posthoc_full_train_factorized_qubo_record",
        "objective": {
            "continuous": (
                "mean_s(mean_active(max_i exp(-alpha*r_sli))-"
                "mean_decoy(max_i exp(-alpha*r_sli)))"
            ),
            "rankbin": (
                "replace exp(-alpha*r) with floor(B*exp(-alpha*r))/B; each "
                "unit level is an OR auxiliary variable"
            ),
            "alpha": development["bedroc_alpha"],
            "reference_bin_count": encoding["reference_bin_count"],
        },
        "hardware_execution": False,
        "sparse_expansion_performed": False,
        "targets": models,
    }, maximum_residual


def report_text(result: dict[str, Any]) -> str:
    continuous = result["continuous_reference"]
    rankbin = result["rankbin_reference"]
    gate = result["route_gate"]
    return f"""# Stage67 BEDROC-aligned rank-bin QUBO

## Scope

Stage67 uses four consumed historical development matrices only. It performs no
new docking, protected-data read, or quantum-hardware execution.

## Question

Can a QUBO preserve the continuous best-rank signal that Stage66 lost? The
continuous reference uses exp(-20*r). Rank-bin QUBOs approximate the same
function at B=4, 8, 16, and 32 levels without tuning biological weights.

## Continuous objective ceiling

- Mean target gain over pair-off: {continuous['mean_target_gain_over_pair_off']:+.6f}
- Worst-target gain: {continuous['worst_target_gain_over_pair_off']:+.6f}
- Nonnegative targets: {continuous['nonnegative_target_count_over_pair_off']}/4

## B=32 QUBO approximation

- Mean target gain over pair-off: {rankbin['mean_target_gain_over_pair_off']:+.6f}
- Worst-target gain: {rankbin['worst_target_gain_over_pair_off']:+.6f}
- Mean subset Jaccard versus continuous reference: {rankbin['mean_subset_jaccard_vs_continuous']:.6f}
- Mean absolute training-objective quantization error: {rankbin['mean_absolute_train_quantization_error']:.6f}

## Decision

- Continuous objective supported: **{'PASS' if gate['continuous_objective_supported'] else 'NO-GO'}**
- B=32 rank-bin QUBO frozen: **{'PASS' if gate['rankbin_qubo_freeze_authorized'] else 'NO-GO'}**

This stage tests objective fidelity, not quantum speedup or quantum advantage.
"""


def compute_analysis(config: dict[str, Any], root: Path) -> dict[str, Any]:
    stage64_config_path = verified(root, config["inputs"]["stage64_config"])
    stage66_result_path = verified(root, config["inputs"]["stage66_result"])
    stage66_audit_path = verified(root, config["inputs"]["stage66_audit"])
    stage66_metrics_path = verified(root, config["inputs"]["stage66_fixed_k_metrics"])
    stage64_config = read_json(stage64_config_path)
    stage66_result = read_json(stage66_result_path)
    stage66_audit = read_json(stage66_audit_path)
    if stage66_result.get("status") != "stage66_cross_target_auxiliary_coverage_qubo_complete":
        raise ValueError("Stage67 source Stage66 result did not complete")
    if stage66_audit.get("status") != "stage66_cross_target_auxiliary_coverage_qubo_independent_audit_ok":
        raise ValueError("Stage67 source Stage66 audit did not pass")
    if stage66_result["freeze_gate"]["coverage_objective_freeze_authorized"]:
        raise ValueError("Stage67 requires the Stage66 coverage no-go")
    development = config["development"]
    target_order = [str(value) for value in development["target_order"]]
    bin_counts = [int(value) for value in development["bin_counts"]]
    objective_ids = [CONTINUOUS_ID] + [objective_id(value) for value in bin_counts]
    targets = {
        target_id: load_target(root, target_id, stage64_config["targets"][target_id])
        for target_id in target_order
    }
    rows: list[dict[str, Any]] = []
    for target_id in target_order:
        target = targets[target_id]
        ligand_ids = target["ligand_ids"]
        labels = target["labels"]
        receptor_ids = target["receptor_ids"]
        for outer_fold in range(int(development["outer_fold_count"])):
            train_mask = np.asarray(
                [target["outer"][ligand_id] != outer_fold for ligand_id in ligand_ids]
            )
            holdout_mask = ~train_mask
            ranks = rank_cube(target["scores"], train_mask)
            train_continuous = RankUtilityObjective(
                ranks,
                labels,
                train_mask,
                float(development["bedroc_alpha"]),
                None,
            )
            holdout_continuous = RankUtilityObjective(
                ranks,
                labels,
                holdout_mask,
                float(development["bedroc_alpha"]),
                None,
            )
            for bin_count in [None, *bin_counts]:
                train_scorer = (
                    train_continuous
                    if bin_count is None
                    else RankUtilityObjective(
                        ranks,
                        labels,
                        train_mask,
                        float(development["bedroc_alpha"]),
                        bin_count,
                    )
                )
                holdout_scorer = (
                    holdout_continuous
                    if bin_count is None
                    else RankUtilityObjective(
                        ranks,
                        labels,
                        holdout_mask,
                        float(development["bedroc_alpha"]),
                        bin_count,
                    )
                )
                greedy = direct_greedy_by_size(train_scorer, max(K_VALUES))
                selected, records = beam_swap_by_size(
                    train_scorer,
                    max(K_VALUES),
                    int(development["classical_beam_width"]),
                )
                for subset_size in K_VALUES:
                    rows.append(
                        metric_row(
                            target_id,
                            outer_fold,
                            bin_count,
                            SOLVER_BEAM,
                            subset_size,
                            selected[subset_size],
                            receptor_ids,
                            train_scorer,
                            holdout_scorer,
                            train_continuous,
                            holdout_continuous,
                            ranks,
                            labels,
                            holdout_mask,
                            float(development["bedroc_alpha"]),
                            records[subset_size],
                        )
                    )
                    rows.append(
                        metric_row(
                            target_id,
                            outer_fold,
                            bin_count,
                            SOLVER_GREEDY,
                            subset_size,
                            greedy[subset_size],
                            receptor_ids,
                            train_scorer,
                            holdout_scorer,
                            train_continuous,
                            holdout_continuous,
                            ranks,
                            labels,
                            holdout_mask,
                            float(development["bedroc_alpha"]),
                            {"start_state_count": 1, "local_endpoint_count": 1},
                        )
                    )
            print(
                json.dumps(
                    {
                        "target_id": target_id,
                        "outer_fold": outer_fold,
                        "objective_count": len(objective_ids),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    source_pair_off = [
        row
        for row in read_csv(stage66_metrics_path)
        if row["solver_id"] == SOLVER_PAIR_OFF
    ]
    for source in source_pair_off:
        rows.append(
            {
                "target_id": source["target_id"],
                "outer_fold": int(source["outer_fold"]),
                "objective_id": "pair_off",
                "bin_count": "",
                "solver_id": SOLVER_PAIR_OFF,
                "subset_size": int(source["subset_size"]),
                "selected_subset": source["selected_subset"],
                "train_objective": source["train_set_objective"],
                "train_continuous_objective": "",
                "train_quantization_error": "",
                "train_mean_active_utility": "",
                "train_mean_decoy_utility": "",
                "train_worst_seed_discrimination": "",
                "holdout_objective": "",
                "holdout_continuous_objective": "",
                "holdout_quantization_error": "",
                "holdout_worst_seed_discrimination": "",
                "holdout_primary_bedroc": source["holdout_primary_bedroc"],
                "holdout_mean_seed_bedroc": source["holdout_mean_seed_bedroc"],
                "holdout_worst_seed_bedroc": source["holdout_worst_seed_bedroc"],
                "holdout_robust_bedroc": source["holdout_robust_bedroc"],
                "search_start_state_count": source["search_start_state_count"],
                "search_local_endpoint_count": source["search_local_endpoint_count"],
            }
        )
    expected = (
        len(target_order)
        * int(development["outer_fold_count"])
        * len(objective_ids)
        * len(K_VALUES)
        * 2
        + 96
    )
    if len(rows) != expected or len(source_pair_off) != 96:
        raise ValueError("Stage67 metric dimensions differ")
    target_rows, global_rows = target_and_global_summaries(
        rows, objective_ids, target_order
    )
    global_lookup = {row["objective_id"]: row for row in global_rows}
    continuous = global_lookup[CONTINUOUS_ID]
    reference_id = objective_id(int(config["qubo_encoding"]["reference_bin_count"]))
    rankbin = global_lookup[reference_id]
    model_record, maximum_residual = build_model_record(config, targets)
    thresholds = config["route_gate"]

    def performance_checks(row: dict[str, Any], prefix: str) -> dict[str, bool]:
        return {
            f"{prefix}_minimum_mean_target_gain_over_pair_off": float(
                row["mean_target_gain_over_pair_off"]
            )
            >= float(thresholds["minimum_mean_target_gain_over_pair_off"])
            - TOLERANCE,
            f"{prefix}_minimum_worst_target_gain_over_pair_off": float(
                row["worst_target_gain_over_pair_off"]
            )
            >= float(thresholds["minimum_worst_target_gain_over_pair_off"])
            - TOLERANCE,
            f"{prefix}_minimum_nonnegative_target_count_over_pair_off": int(
                row["nonnegative_target_count_over_pair_off"]
            )
            >= int(thresholds["minimum_nonnegative_target_count_over_pair_off"]),
        }

    continuous_checks = performance_checks(continuous, "continuous")
    rankbin_checks = performance_checks(rankbin, "rankbin")
    fidelity_checks = {
        "rankbin_minimum_mean_subset_jaccard_vs_continuous": float(
            rankbin["mean_subset_jaccard_vs_continuous"]
        )
        >= float(thresholds["minimum_mean_subset_jaccard_vs_continuous"])
        - TOLERANCE,
        "rankbin_maximum_mean_absolute_train_quantization_error": float(
            rankbin["mean_absolute_train_quantization_error"]
        )
        <= float(thresholds["maximum_mean_absolute_train_quantization_error"])
        + TOLERANCE,
        "maximum_factorized_qubo_energy_residual": maximum_residual
        <= float(thresholds["maximum_factorized_qubo_energy_residual"])
        + TOLERANCE,
    }
    continuous_supported = all(continuous_checks.values())
    rankbin_frozen = continuous_supported and all(rankbin_checks.values()) and all(
        fidelity_checks.values()
    )
    checks = {**continuous_checks, **rankbin_checks, **fidelity_checks}
    if rankbin_frozen:
        next_action = "freeze B=32 rank-bin QUBO for a genuinely new target preregistration"
    elif continuous_supported:
        next_action = "retain the continuous objective but redesign or increase QUBO resolution before external testing"
    else:
        next_action = "stop the min-rank exponential-discrimination QUBO route on these targets"
    model_record["status"] = (
        "rankbin_qubo_frozen_for_new_target_preregistration"
        if rankbin_frozen
        else "rankbin_qubo_not_frozen"
    )
    result = {
        "schema_version": "1.0",
        "status": "stage67_bedroc_rankbin_qubo_complete",
        "experiment_class": "posthoc cross-target train-only objective fidelity adjudication",
        "objective_count": len(objective_ids),
        "fixed_k_metric_count": len(rows),
        "pair_off_reproduction_cell_count": len(source_pair_off),
        "continuous_reference": continuous,
        "rankbin_reference": rankbin,
        "route_gate": {
            "checks": checks,
            "continuous_objective_supported": continuous_supported,
            "rankbin_qubo_freeze_authorized": rankbin_frozen,
        },
        "qubo_model_audit": {
            "target_model_count": len(model_record["targets"]),
            "maximum_factorized_energy_residual": maximum_residual,
            "total_variable_count_at_b32_k3": sum(
                int(row["variable_counts"]["total"])
                for row in model_record["targets"].values()
            ),
            "maximum_target_variable_count_at_b32_k3": max(
                int(row["variable_counts"]["total"])
                for row in model_record["targets"].values()
            ),
            "hardware_execution": False,
        },
        "decision": {
            "new_target_preregistration_authorized": rankbin_frozen,
            "fresh_validation_authorized": False,
            "new_docking_authorized": False,
            "quantum_hardware_authorized": False,
            "same_target_retuning_authorized": False,
            "next_action": next_action,
        },
        "data_boundary": {
            "historical_development_targets_read": 4,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "target_input_audits": {
            target_id: {
                "ligand_count": len(targets[target_id]["ligand_ids"]),
                "receptor_count": len(targets[target_id]["receptor_ids"]),
                "score_row_count": 3
                * len(targets[target_id]["ligand_ids"])
                * len(targets[target_id]["receptor_ids"]),
                "input_descriptors": targets[target_id]["input_descriptors"],
            }
            for target_id in target_order
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    return {
        "rows": rows,
        "target_rows": target_rows,
        "global_rows": global_rows,
        "model_record": model_record,
        "result": result,
    }


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    runner = verified(root, config["implementation"]["runner"])
    if runner.resolve() != Path(__file__).resolve():
        raise ValueError("Stage67 runner identity differs")
    for value in config["implementation"].values():
        verified(root, value)
    outputs = {key: root / value for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage67 outputs exist; pass --overwrite")
    analysis = compute_analysis(config, root)
    write_csv(outputs["fixed_k_metrics_csv"], analysis["rows"])
    write_csv(outputs["target_summary_csv"], analysis["target_rows"])
    write_csv(outputs["resolution_summary_csv"], analysis["global_rows"])
    write_json(outputs["model_record_json"], analysis["model_record"])
    result = analysis["result"]
    result["config"] = descriptor(root, config_path)
    result["implementation"] = descriptor(root, runner)
    result["outputs"] = {
        key: descriptor(root, path)
        for key, path in outputs.items()
        if key not in {"result_json", "audit_json", "report_md"}
    }
    result["analysis_payload_sha256"] = canonical_sha256(
        {
            "target_summary": analysis["target_rows"],
            "resolution_summary": analysis["global_rows"],
            "route_gate": result["route_gate"],
        }
    )
    write_json(outputs["result_json"], result)
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text(report_text(result), encoding="ascii")
    result["outputs"]["report_md"] = descriptor(root, outputs["report_md"])
    write_json(outputs["result_json"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage67_bedroc_rankbin_qubo.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
