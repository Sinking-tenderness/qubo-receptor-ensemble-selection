"""Replay completed docking matrices under a hidden, sequential task budget.

This is a post-hoc policy replay, not a new docking experiment.  The selector
can see only scores from tasks selected in earlier rounds.  Labels are loaded
only by the final metric evaluator and never enter the surrogate or QUBO.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.linear_model import BayesianRidge


META_COLUMNS = {"ligand_id", "label", "selection_role", "split_group_id"}
POLICIES = [
    "random",
    "predicted_mean",
    "predictive_uncertainty",
    "qubo_direct_greedy",
    "qubo_greedy_one_swap",
    "qubo_exact_milp",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def resolve(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_matrix(path: Path) -> tuple[list[str], list[dict[str, str]], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"matrix has no header: {path}")
        receptors = [field for field in reader.fieldnames if field not in META_COLUMNS]
        rows = list(reader)
    if not receptors or not rows:
        raise ValueError(f"matrix has no receptor columns or rows: {path}")
    values = np.asarray([[float(row[name]) for name in receptors] for row in rows], dtype=float)
    return receptors, rows, values


def read_config(root: Path, path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for target, spec in config["targets"].items():
        matrix = resolve(root, spec["matrix"]["path"])
        manifest = resolve(root, spec["ligand_manifest"]["path"])
        if sha256(matrix) != spec["matrix"]["sha256"]:
            raise ValueError(f"matrix hash mismatch for {target}")
        if sha256(manifest) != spec["ligand_manifest"]["sha256"]:
            raise ValueError(f"manifest hash mismatch for {target}")
    return config


def bedroc(scores: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    mask = np.isfinite(scores) & (labels >= 0)
    scores = scores[mask]
    labels = labels[mask]
    if scores.size == 0:
        return math.nan
    order = np.argsort(scores, kind="stable")
    ordered_labels = labels[order]
    total = int(ordered_labels.size)
    active_ranks = [i for i, label in enumerate(ordered_labels, start=1) if label == 1]
    active_total = len(active_ranks)
    if active_total == 0 or active_total == total:
        return math.nan

    def exp_sum(ranks: list[int]) -> float:
        return sum(math.exp(-alpha * rank / total) for rank in ranks)

    all_weights = [math.exp(-alpha * rank / total) for rank in range(1, total + 1)]
    random_expected = active_total * sum(all_weights) / total
    observed = exp_sum(active_ranks) / random_expected
    best = exp_sum(list(range(1, active_total + 1))) / random_expected
    worst = exp_sum(list(range(total - active_total + 1, total + 1))) / random_expected
    return (observed - worst) / (best - worst)


def standardize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    scale = float(np.std(values))
    if scale < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / scale


def initial_cross(receptor_count: int, batch_ids: list[str], target: str, seed: int) -> set[tuple[int, str]]:
    digest = hashlib.sha256(f"{target}|{seed}".encode("utf-8")).digest()
    anchor_receptor = int.from_bytes(digest[:4], "big") % receptor_count
    anchor_batch = batch_ids[int.from_bytes(digest[4:8], "big") % len(batch_ids)]
    tasks = {(receptor_index, anchor_batch) for receptor_index in range(receptor_count)}
    tasks.update({(anchor_receptor, batch_id) for batch_id in batch_ids})
    return tasks


def design_matrix(
    tasks: list[tuple[int, str]],
    receptor_count: int,
    batch_ids: list[str],
) -> np.ndarray:
    batch_index = {batch_id: index for index, batch_id in enumerate(batch_ids)}
    columns = 1 + max(0, receptor_count - 1) + max(0, len(batch_ids) - 1)
    matrix = np.zeros((len(tasks), columns), dtype=float)
    for row_index, (receptor_index, batch_id) in enumerate(tasks):
        matrix[row_index, 0] = 1.0
        if receptor_index > 0:
            matrix[row_index, receptor_index] = 1.0
        batch_offset = receptor_count
        batch_value = batch_index[batch_id]
        if batch_value > 0:
            matrix[row_index, batch_offset + batch_value - 1] = 1.0
    return matrix


def predict(
    observed: set[tuple[int, str]],
    responses: dict[tuple[int, str], float],
    all_tasks: list[tuple[int, str]],
    receptor_count: int,
    batch_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    if len(observed) < 3:
        global_mean = float(np.mean(list(responses.values()))) if responses else 0.0
        return np.full(len(all_tasks), global_mean), np.full(len(all_tasks), 1.0)
    ordered = sorted(observed)
    x_train = design_matrix(ordered, receptor_count, batch_ids)
    y_train = np.asarray([responses[task] for task in ordered], dtype=float)
    x_all = design_matrix(all_tasks, receptor_count, batch_ids)
    try:
        model = BayesianRidge(fit_intercept=False, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
        model.fit(x_train, y_train)
        mean, std = model.predict(x_all, return_std=True)
        return np.asarray(mean), np.maximum(np.asarray(std), 1e-9)
    except (ValueError, FloatingPointError):
        global_mean = float(np.mean(y_train))
        return np.full(len(all_tasks), global_mean), np.full(len(all_tasks), max(float(np.std(y_train)), 1e-9))


def objective(selected: list[int], utility: np.ndarray, redundancy: np.ndarray, weight: float) -> float:
    if not selected:
        return -math.inf
    node = float(np.mean(utility[selected]))
    if len(selected) < 2:
        return node
    pairs = [float(redundancy[i, j]) for offset, i in enumerate(selected) for j in selected[offset + 1 :]]
    return node - weight * float(np.mean(pairs))


def greedy_selection(utility: np.ndarray, redundancy: np.ndarray, count: int, weight: float) -> list[int]:
    selected: list[int] = []
    remaining = set(range(len(utility)))
    while remaining and len(selected) < count:
        best = max(remaining, key=lambda index: (objective(selected + [index], utility, redundancy, weight), -index))
        selected.append(best)
        remaining.remove(best)
    return selected


def one_swap_selection(utility: np.ndarray, redundancy: np.ndarray, count: int, weight: float) -> list[int]:
    selected = greedy_selection(utility, redundancy, count, weight)
    best_value = objective(selected, utility, redundancy, weight)
    while True:
        selected_set = set(selected)
        best_swap: tuple[float, int, int] | None = None
        for old in sorted(selected_set):
            for new in range(len(utility)):
                if new in selected_set:
                    continue
                candidate = [new if index == old else index for index in selected]
                value = objective(candidate, utility, redundancy, weight)
                proposal = (value, -old, -new)
                if value > best_value + 1e-12 and (best_swap is None or proposal > best_swap):
                    best_swap = proposal
        if best_swap is None:
            return selected
        _, old_key, new_key = best_swap
        old = -old_key
        new = -new_key
        selected = [new if index == old else index for index in selected]
        best_value = objective(selected, utility, redundancy, weight)


def exact_selection(utility: np.ndarray, redundancy: np.ndarray, count: int, weight: float) -> tuple[list[int], float, str]:
    if count <= 0:
        return [], -math.inf, "empty"
    if count == 1:
        selected = [int(np.argmax(utility))]
        return selected, objective(selected, utility, redundancy, weight), "scipy_milp_exact"
    n = len(utility)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_count = len(pairs)
    c = np.concatenate([-utility / count, weight * np.asarray([redundancy[i, j] for i, j in pairs]) / (count * (count - 1) / 2)])
    integrality = np.ones(n + pair_count)
    lower = np.zeros(n + pair_count)
    upper = np.ones(n + pair_count)
    rows: list[np.ndarray] = []
    lows: list[float] = []
    highs: list[float] = []
    equality = np.zeros(n + pair_count)
    equality[:n] = 1.0
    rows.append(equality)
    lows.append(float(count))
    highs.append(float(count))
    for pair_index, (i, j) in enumerate(pairs):
        z = n + pair_index
        row = np.zeros(n + pair_count)
        row[z] = 1.0
        row[i] = -1.0
        rows.append(row)
        lows.append(-math.inf)
        highs.append(0.0)
        row = np.zeros(n + pair_count)
        row[z] = 1.0
        row[j] = -1.0
        rows.append(row)
        lows.append(-math.inf)
        highs.append(0.0)
        row = np.zeros(n + pair_count)
        row[i] = 1.0
        row[j] = 1.0
        row[z] = -1.0
        rows.append(row)
        lows.append(-math.inf)
        highs.append(1.0)
    result = milp(c, integrality=integrality, bounds=Bounds(lower, upper), constraints=LinearConstraint(np.asarray(rows), lows, highs), options={"time_limit": 30.0})
    if result.x is None:
        fallback = one_swap_selection(utility, redundancy, count, weight)
        return fallback, objective(fallback, utility, redundancy, weight), "fallback_one_swap"
    selected = [index for index, value in enumerate(result.x[:n]) if value >= 0.5]
    selected = selected[:count]
    return selected, objective(selected, utility, redundancy, weight), "scipy_milp_exact" if result.success else "scipy_milp_incumbent"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def run_target(root: Path, target: str, spec: dict[str, Any], batch_rows: list[dict[str, str]], config: dict[str, Any], trajectories: list[dict[str, Any]], checkpoints: list[dict[str, Any]], solver_rows: list[dict[str, Any]]) -> dict[str, Any]:
    receptors, matrix_rows, scores = read_matrix(resolve(root, spec["matrix"]["path"]))
    manifest_rows = read_csv(resolve(root, spec["ligand_manifest"]["path"]))
    manifest_by_id = {row["ligand_id"]: row for row in manifest_rows}
    ligand_ids = [row["ligand_id"] for row in matrix_rows]
    if set(ligand_ids) != set(manifest_by_id):
        raise ValueError(f"ligand manifest mismatch for {target}")
    batch_by_ligand: dict[str, str] = {}
    batch_sizes: dict[str, int] = {}
    smiles_by_batch: dict[str, list[str]] = {}
    for row in batch_rows:
        if row["target_id"] != target:
            continue
        batch_by_ligand[row["ligand_id"]] = row["chemistry_batch_id"]
        batch_sizes[row["chemistry_batch_id"]] = int(row["batch_size"])
        smiles_by_batch.setdefault(row["chemistry_batch_id"], []).append(row["canonical_smiles"])
    if set(ligand_ids) != set(batch_by_ligand):
        raise ValueError(f"chemistry batch manifest mismatch for {target}")
    batch_ids = sorted(batch_sizes)
    ligand_indices_by_batch = {batch_id: [index for index, ligand_id in enumerate(ligand_ids) if batch_by_ligand[ligand_id] == batch_id] for batch_id in batch_ids}
    batch_summary = json.loads(resolve(root, "data/stage96_multitarget_balanced_chemistry_batches_summary.json").read_text(encoding="utf-8"))["targets"][target]
    centroid_similarity = batch_summary["centroid_tanimoto_similarity"]
    labels = np.asarray([1 if manifest_by_id[ligand_id]["label"] in {"active", "high"} else 0 if manifest_by_id[ligand_id]["label"] in {"decoy", "low"} else -1 for ligand_id in ligand_ids], dtype=int)
    eval_mask = labels >= 0
    oracle_scores = np.min(scores, axis=1)
    oracle_bedroc = bedroc(oracle_scores, labels, 20.0)
    total_cost = int(sum(batch_sizes.values()) * len(receptors))
    all_tasks = [(receptor_index, batch_id) for receptor_index in range(len(receptors)) for batch_id in batch_ids]
    initial_costs: dict[int, int] = {}
    target_summary: dict[str, Any] = {"target_id": target, "ligand_count": len(ligand_ids), "receptor_count": len(receptors), "batch_count": len(batch_ids), "total_task_cost": total_cost, "oracle_bedroc_alpha_20": oracle_bedroc, "policies": {}}
    for replay_seed in config["replay"]["replay_seeds"]:
        initial = initial_cross(len(receptors), batch_ids, target, int(replay_seed))
        for policy in POLICIES:
            rng = random.Random(int(replay_seed) + sum(ord(c) for c in policy))
            observed = set(initial)
            responses: dict[tuple[int, str], float] = {}
            current = np.full(len(ligand_ids), np.inf)
            for receptor_index, batch_id in sorted(initial):
                indices = ligand_indices_by_batch[batch_id]
                task_scores = scores[indices, receptor_index]
                quartile_count = max(1, math.ceil(len(task_scores) * 0.25))
                responses[(receptor_index, batch_id)] = float(-np.mean(np.sort(task_scores)[:quartile_count]))
                current[indices] = np.minimum(current[indices], task_scores)
            cost = int(sum(batch_sizes[batch_id] for _, batch_id in initial))
            round_index = 0
            checkpoints_seen: set[float] = set()
            policy_rows: list[dict[str, Any]] = []
            initial_bedroc = bedroc(current, labels, 20.0)
            while cost < total_cost * float(config["replay"]["maximum_task_fraction"]):
                unobserved = [task for task in all_tasks if task not in observed]
                if not unobserved:
                    break
                mean, std = predict(observed, responses, unobserved, len(receptors), batch_ids)
                desired_cost = max(1, math.ceil(total_cost * float(config["replay"]["round_batch_task_fraction"])))
                median_cost = max(1, int(np.median(list(batch_sizes.values()))))
                select_count = max(1, math.ceil(desired_cost / median_cost))
                select_count = min(select_count, len(unobserved))
                if policy == "random":
                    selected_positions = rng.sample(range(len(unobserved)), select_count)
                    selected_positions.sort()
                    selected_tasks = [unobserved[index] for index in selected_positions]
                    solver_info = None
                else:
                    if policy == "predicted_mean":
                        order = np.argsort(-mean, kind="stable")
                        selected_tasks = [unobserved[int(index)] for index in order[:select_count]]
                        solver_info = None
                    elif policy == "predictive_uncertainty":
                        order = np.argsort(-std, kind="stable")
                        selected_tasks = [unobserved[int(index)] for index in order[:select_count]]
                        solver_info = None
                    else:
                        node_raw = 0.65 * standardize(mean) + 0.35 * standardize(std)
                        order = np.argsort(-node_raw, kind="stable")[: int(config["qubo_batch_objective"]["candidate_pool_size"])]
                        candidates = [unobserved[int(index)] for index in order]
                        utility = standardize(node_raw[order])
                        candidate_similarity = np.zeros((len(candidates), len(candidates)), dtype=float)
                        for i, (receptor_i, batch_i) in enumerate(candidates):
                            for j, (receptor_j, batch_j) in enumerate(candidates):
                                same_receptor = 1.0 if receptor_i == receptor_j else 0.0
                                centroid = float(centroid_similarity[batch_i][batch_j])
                                candidate_similarity[i, j] = 0.5 * same_receptor + 0.5 * centroid
                        q_weight = float(config["qubo_batch_objective"]["redundancy_weight"])
                        direct = greedy_selection(utility, candidate_similarity, select_count, q_weight)
                        swap = one_swap_selection(utility, candidate_similarity, select_count, q_weight)
                        exact, exact_value, solver_name = exact_selection(utility, candidate_similarity, select_count, q_weight)
                        direct_value = objective(direct, utility, candidate_similarity, q_weight)
                        swap_value = objective(swap, utility, candidate_similarity, q_weight)
                        solver_rows.append({"target_id": target, "replay_seed": replay_seed, "policy_round": round_index, "candidate_count": len(candidates), "select_count": select_count, "direct_greedy_objective": direct_value, "greedy_one_swap_objective": swap_value, "exact_milp_objective": exact_value, "exact_minus_one_swap": exact_value - swap_value, "solver_status": solver_name})
                        if policy == "qubo_direct_greedy":
                            selected_tasks = [candidates[index] for index in direct]
                        elif policy == "qubo_greedy_one_swap":
                            selected_tasks = [candidates[index] for index in swap]
                        else:
                            selected_tasks = [candidates[index] for index in exact]
                        solver_info = {"candidate_count": len(candidates), "select_count": select_count, "direct_objective": direct_value, "one_swap_objective": swap_value, "exact_objective": exact_value, "exact_minus_one_swap": exact_value - swap_value, "solver_status": solver_name}
                for receptor_index, batch_id in selected_tasks:
                    if (receptor_index, batch_id) in observed:
                        continue
                    indices = ligand_indices_by_batch[batch_id]
                    task_scores = scores[indices, receptor_index]
                    quartile_count = max(1, math.ceil(len(task_scores) * 0.25))
                    responses[(receptor_index, batch_id)] = float(-np.mean(np.sort(task_scores)[:quartile_count]))
                    observed.add((receptor_index, batch_id))
                    current[indices] = np.minimum(current[indices], task_scores)
                    cost += batch_sizes[batch_id]
                round_index += 1
                current_bedroc = bedroc(current, labels, 20.0)
                row = {"target_id": target, "policy": policy, "replay_seed": replay_seed, "round": round_index, "task_fraction": cost / total_cost, "task_cost": cost, "tasks_observed": len(observed), "bedroc_alpha_20": current_bedroc, "oracle_bedroc_alpha_20": oracle_bedroc, "selected_task_ids": "|".join(f"{receptors[r]}::{b}" for r, b in sorted(observed)), "solver_info": json.dumps(solver_info, sort_keys=True) if solver_info else ""}
                policy_rows.append(row)
                trajectories.append(row)
                for fraction in config["replay"]["checkpoint_task_fractions"]:
                    fraction = float(fraction)
                    if fraction not in checkpoints_seen and cost / total_cost >= fraction:
                        denominator = oracle_bedroc - initial_bedroc
                        recovery = (current_bedroc - initial_bedroc) / denominator if denominator > 0 else math.nan
                        checkpoints.append({"target_id": target, "policy": policy, "replay_seed": replay_seed, "checkpoint_fraction": fraction, "actual_task_fraction": cost / total_cost, "task_cost": cost, "tasks_observed": len(observed), "bedroc_alpha_20": current_bedroc, "oracle_bedroc_alpha_20": oracle_bedroc, "relative_recovery": recovery})
                        checkpoints_seen.add(fraction)
            target_summary["policies"].setdefault(policy, []).append({"replay_seed": replay_seed, "initial_task_fraction": sum(batch_sizes[b] for _, b in initial) / total_cost, "final_task_fraction": cost / total_cost, "initial_bedroc_alpha_20": initial_bedroc, "final_bedroc_alpha_20": float(policy_rows[-1]["bedroc_alpha_20"]) if policy_rows else initial_bedroc, "initial_tasks": len(initial), "final_tasks": len(observed)})
    return target_summary


def finite_mean(values: list[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage96_multitarget_adaptive_docking_replay.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = read_config(root, args.config if args.config.is_absolute() else root / args.config)
    batch_rows = read_csv(resolve(root, config["outputs"]["cluster_manifest"]))
    trajectories: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    target_summaries = []
    for target, spec in config["targets"].items():
        target_summaries.append(run_target(root, target, spec, batch_rows, config, trajectories, checkpoints, solver_rows))
    output_dir = root / "results/runs/stage96_multitarget_adaptive_docking_replay"
    write_csv(output_dir / "trajectories.csv", trajectories)
    write_csv(output_dir / "checkpoints.csv", checkpoints)
    write_csv(output_dir / "qubo_solver_comparisons.csv", solver_rows)

    primary_fraction = float(config["go_gate"]["primary_checkpoint_task_fraction"])
    gate_rows = []
    for target in config["targets"]:
        q_rows = [row for row in checkpoints if row["target_id"] == target and row["policy"] == "qubo_exact_milp" and float(row["checkpoint_fraction"]) == primary_fraction]
        non_q_rows = [row for row in checkpoints if row["target_id"] == target and row["policy"] in {"random", "predicted_mean", "predictive_uncertainty"} and float(row["checkpoint_fraction"]) == primary_fraction]
        by_seed = {int(row["replay_seed"]): float(row["bedroc_alpha_20"]) for row in q_rows}
        baseline_by_seed: dict[int, float] = {}
        for seed in by_seed:
            baseline_by_seed[seed] = max(float(row["bedroc_alpha_20"]) for row in non_q_rows if int(row["replay_seed"]) == seed)
        gains = [by_seed[seed] - baseline_by_seed[seed] for seed in by_seed]
        positive_count = sum(gain >= float(config["go_gate"]["minimum_qubo_exact_mean_bedroc_gain_over_best_nonqubo_per_target"]) for gain in gains)
        savings: list[float] = []
        for seed in by_seed:
            q_recovery = [row for row in checkpoints if row["target_id"] == target and row["policy"] == "qubo_exact_milp" and int(row["replay_seed"]) == seed and math.isfinite(float(row["relative_recovery"])) and float(row["relative_recovery"]) >= 0.95]
            baseline_recovery = [row for row in checkpoints if row["target_id"] == target and row["policy"] in {"random", "predicted_mean", "predictive_uncertainty"} and int(row["replay_seed"]) == seed and math.isfinite(float(row["relative_recovery"])) and float(row["relative_recovery"]) >= 0.95]
            if q_recovery and baseline_recovery:
                q_fraction = min(float(row["actual_task_fraction"]) for row in q_recovery)
                baseline_fraction = min(float(row["actual_task_fraction"]) for row in baseline_recovery)
                savings.append((baseline_fraction - q_fraction) / baseline_fraction)
        mean_saving = finite_mean(savings)
        saving_passes = bool(savings) and mean_saving >= float(config["go_gate"]["minimum_task_saving_to_95_percent_relative_recovery"])
        gain_passes = positive_count >= int(config["go_gate"]["minimum_positive_replay_seed_count_per_target"]) and finite_mean(gains) >= float(config["go_gate"]["minimum_qubo_exact_mean_bedroc_gain_over_best_nonqubo_per_target"])
        target_gate = {"target_id": target, "primary_checkpoint_fraction": primary_fraction, "qubo_exact_mean_bedroc": finite_mean(list(by_seed.values())), "best_nonqubo_mean_bedroc": finite_mean(list(baseline_by_seed.values())), "mean_gain": finite_mean(gains), "seed_gains": gains, "positive_seed_count": positive_count, "gain_gate_passes": gain_passes, "relative_recovery_saving_values": savings, "mean_task_saving_to_95_percent_relative_recovery": mean_saving, "task_saving_gate_passes": saving_passes, "passes_policy_gate": gain_passes and saving_passes}
        gate_rows.append(target_gate)
    policy_gate = sum(bool(row["passes_policy_gate"]) for row in gate_rows) >= int(config["go_gate"]["minimum_target_count_passing_policy_gate"])
    solver_gaps = [float(row["exact_minus_one_swap"]) for row in solver_rows]
    solver_value = any(gap > 1e-9 for gap in solver_gaps)
    result = {"schema_version": "1.0", "experiment_id": config["experiment_id"], "status": "stage96_replay_complete", "data_boundary": config["data_boundary"], "target_summaries": target_summaries, "policy_gate": {"targets": gate_rows, "passes": policy_gate}, "solver_value": {"max_exact_minus_one_swap": max(solver_gaps) if solver_gaps else math.nan, "positive_comparison_count": sum(gap > 1e-9 for gap in solver_gaps), "passes": solver_value}, "metric_reference_caveat": {"full_matrix_minimum_score_is_an_upper_bound_on_bedroc": False, "interpretation": "The all-receptor minimum-score aggregation is a fixed full-matrix reference, not a BEDROC oracle. Relative recovery is undefined when that reference is not above the seed-specific initial BEDROC."}, "hardware_authorization": {"authorized": False, "reason": "posthoc replay cannot authorize hardware; a passing gate would require a frozen prospective target and independent validation"}, "audit": {"labels_used_by_selector": False, "docking_scores_revealed_only_after_task_selection": True, "synthetic_scores": 0, "new_docking_jobs": 0, "fresh_validation_rows": 0}}
    result_path = root / config["outputs"]["result_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(json_safe(result), indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    report_path = root / config["outputs"]["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Stage96 多靶点自适应对接回放", "", "本报告是对已完成 Uni-Dock 矩阵的隐藏顺序回放，不新增对接，不使用新鲜验证数据。策略只能看到自己已经选择的受体×化学批次任务的分数。", "", "## 结果", ""]
    for row in gate_rows:
        lines.append(f"- {row['target_id']}：0.20 预算下 QUBO exact 与最佳非 QUBO 的平均 BEDROC 差值 `{row['mean_gain']:.6f}`；通过策略门：`{row['passes_policy_gate']}`。")
    lines.extend([f"", f"- 两靶点策略门总结果：`{policy_gate}`。", f"- QUBO exact 相对 one-swap 的最大目标值差：`{max(solver_gaps) if solver_gaps else math.nan:.8f}`；存在正差异：`{solver_value}`。", "", "## 解释", "", "若策略门失败，不能通过调参或更换量子硬件挽救本批数据；应停止在这些矩阵上继续后验优化。若策略门通过但 exact 与 one-swap 没有正差异，则只能报告自适应对接的经典应用价值，不能报告量子求解价值。", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "policy_gate": policy_gate, "solver_value": solver_value, "trajectory_rows": len(trajectories), "checkpoint_rows": len(checkpoints), "solver_rows": len(solver_rows), "result": str(result_path), "report": str(report_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
