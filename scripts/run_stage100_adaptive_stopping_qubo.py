"""Evaluate conservative, adaptive receptor-count stopping around Stage99 QUBOs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage99_qubo_objective_repair_screen as s99


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest().upper()


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inner_profiles(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: dict[str, Any],
    parent: dict[str, Any],
) -> tuple[dict[int, list[float]], list[dict[str, Any]]]:
    sizes = [int(k) for k in config["adaptive_cardinality"]["candidate_k_values"]]
    scores = {k: [] for k in sizes}
    records: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=int(config["adaptive_cardinality"]["inner_folds"]))
    for inner_fold, (train_index, valid_index) in enumerate(splitter.split(values, groups=groups), start=1):
        train_index = np.asarray([index for index in train_index if labels[index] >= 0])
        valid_index = np.asarray([index for index in valid_index if labels[index] >= 0])
        train_ranks = s99.percentile_ranks(values[train_index], values[train_index])
        valid_ranks = s99.percentile_ranks(values[train_index], values[valid_index])
        node, pair = s99.coefficients(train_ranks, labels[train_index], parent, "repair")
        for k in sizes:
            subset = s99.exact_select(values.shape[1], k, node, pair)
            score = s99.bedroc(
                s99.aggregate(valid_ranks[:, list(subset)], "minimum"),
                labels[valid_index],
                float(parent["objective"]["bedroc_alpha"]),
            )
            scores[k].append(score)
            records.append({"inner_fold": inner_fold, "k": k, "bedroc_alpha20": score})
    return scores, records


def choose_maximum_mean(scores: dict[int, list[float]]) -> int:
    return max(sorted(scores), key=lambda k: (float(np.mean(scores[k])), -k))


def choose_one_standard_error(scores: dict[int, list[float]]) -> tuple[int, dict[str, float]]:
    best_k = choose_maximum_mean(scores)
    best_mean = float(np.mean(scores[best_k]))
    best_se = standard_error(scores[best_k])
    threshold = best_mean - best_se
    eligible = [k for k in sorted(scores) if float(np.mean(scores[k])) >= threshold]
    return min(eligible), {"best_k": best_k, "best_mean": best_mean, "best_se": best_se, "threshold": threshold}


def choose_marginal_lcb(scores: dict[int, list[float]]) -> tuple[int, list[dict[str, float | bool]]]:
    selected = min(scores)
    decisions: list[dict[str, float | bool]] = []
    ordered = sorted(scores)
    for previous, current in zip(ordered, ordered[1:]):
        differences = np.asarray(scores[current], dtype=float) - np.asarray(scores[previous], dtype=float)
        mean_gain = float(np.mean(differences))
        gain_se = standard_error(differences.tolist())
        lower_bound = mean_gain - gain_se
        should_continue = lower_bound > 0.0
        decisions.append({"from_k": previous, "to_k": current, "mean_gain": mean_gain, "gain_se": gain_se, "lower_bound": lower_bound, "continue": should_continue})
        if not should_continue:
            break
        selected = current
    return selected, decisions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage100_adaptive_stopping_qubo.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    parent_path = root / config["parent"]["config"]
    if sha256(parent_path) != config["parent"]["config_sha256"]:
        raise ValueError("Stage99 parent config hash mismatch")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    fold_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    target_ids = list(parent["targets"])
    for target, spec in parent["targets"].items():
        receptors, rows, values = s99.load_target(root, spec)
        labels = s99.label_array(rows)
        groups = np.asarray([row.get(spec["group_field"], row.get("split_group_id", row["ligand_id"])) for row in rows])
        outer = GroupKFold(n_splits=int(config["adaptive_cardinality"]["outer_folds"]))
        for fold, (train_index, test_index) in enumerate(outer.split(values, groups=groups), start=1):
            train_index = np.asarray([index for index in train_index if labels[index] >= 0])
            test_index = np.asarray([index for index in test_index if labels[index] >= 0])
            train_values = values[train_index]
            train_labels = labels[train_index]
            profiles, profile_records = inner_profiles(train_values, train_labels, groups[train_index], config, parent)
            one_se_k, one_se_details = choose_one_standard_error(profiles)
            marginal_k, marginal_details = choose_marginal_lcb(profiles)
            maximum_mean_k = choose_maximum_mean(profiles)
            for record in profile_records:
                inner_rows.append({"target_id": target, "outer_fold": fold, **record})
            train_ranks = s99.percentile_ranks(train_values, train_values)
            test_ranks = s99.percentile_ranks(train_values, values[test_index])
            node, pair = s99.coefficients(train_ranks, train_labels, parent, "repair")
            choices = {
                "single": 1,
                "fixed_k3": 3,
                "maximum_inner_mean": maximum_mean_k,
                "one_standard_error_smallest_k": one_se_k,
                "sequential_paired_marginal_lcb": marginal_k,
            }
            for method, k in choices.items():
                exact = s99.exact_select(len(receptors), k, node, pair)
                local = s99.greedy_swap(len(receptors), k, node, pair)
                exact_bedroc = s99.bedroc(s99.aggregate(test_ranks[:, list(exact)], "minimum"), labels[test_index], float(parent["objective"]["bedroc_alpha"]))
                local_bedroc = s99.bedroc(s99.aggregate(test_ranks[:, list(local)], "minimum"), labels[test_index], float(parent["objective"]["bedroc_alpha"]))
                fold_rows.append({
                    "target_id": target,
                    "fold": fold,
                    "method": method,
                    "selected_k": k,
                    "selected_receptors": s99.subset_name(exact, receptors),
                    "outer_bedroc_alpha20": exact_bedroc,
                    "one_swap_bedroc_alpha20": local_bedroc,
                    "exact_minus_one_swap_objective": s99.q_value(exact, node, pair) - s99.q_value(local, node, pair),
                    "one_se_best_k": int(one_se_details["best_k"]),
                    "one_se_threshold": one_se_details["threshold"],
                    "marginal_decisions_json": json.dumps(marginal_details, separators=(",", ":")),
                    "selector_used_outer_test_labels": False,
                })
    methods = ["single", "fixed_k3", "maximum_inner_mean", "one_standard_error_smallest_k", "sequential_paired_marginal_lcb"]
    target_rows: list[dict[str, Any]] = []
    for target in target_ids:
        for method in methods:
            selected = [row for row in fold_rows if row["target_id"] == target and row["method"] == method]
            target_rows.append({
                "target_id": target,
                "method": method,
                "mean_outer_bedroc_alpha20": float(np.mean([row["outer_bedroc_alpha20"] for row in selected])),
                "std_outer_bedroc_alpha20": float(np.std([row["outer_bedroc_alpha20"] for row in selected], ddof=1)),
                "mean_selected_k": float(np.mean([row["selected_k"] for row in selected])),
                "selected_k_values": "|".join(str(row["selected_k"]) for row in selected),
                "fold_count": len(selected),
            })
    def target_mean(target: str, method: str) -> float:
        row = next(value for value in target_rows if value["target_id"] == target and value["method"] == method)
        return float(row["mean_outer_bedroc_alpha20"])
    primary_rows = []
    for target in target_ids:
        adaptive = target_mean(target, "one_standard_error_smallest_k")
        single = target_mean(target, "single")
        primary_rows.append({"target_id": target, "adaptive_bedroc": adaptive, "single_bedroc": single, "gain": adaptive - single})
    gains = [row["gain"] for row in primary_rows]
    nontrivial_count = sum(int(row["selected_k"]) > 1 for row in fold_rows if row["method"] == "one_standard_error_smallest_k")
    gate = {
        "positive_target_count_at_0p02": sum(gain >= 0.02 for gain in gains),
        "mean_gain_over_single": float(np.mean(gains)),
        "worst_target_gain": float(np.min(gains)),
        "nontrivial_selected_fold_count": nontrivial_count,
    }
    thresholds = config["gate"]
    gate["passes"] = gate["positive_target_count_at_0p02"] >= int(thresholds["minimum_positive_target_count_at_0p02"]) and gate["mean_gain_over_single"] >= float(thresholds["minimum_mean_gain_over_single"]) and gate["worst_target_gain"] >= float(thresholds["minimum_worst_target_gain"]) and nontrivial_count >= int(thresholds["minimum_nontrivial_selected_fold_count"])
    outputs = config["outputs"]
    write_csv(root / outputs["fold_csv"], fold_rows)
    write_csv(root / outputs["inner_csv"], inner_rows)
    write_csv(root / outputs["target_csv"], target_rows)
    result = {
        "schema_version": "1.0",
        "status": "stage100_adaptive_stopping_qubo_complete",
        "primary_rule": config["adaptive_cardinality"]["primary_rule"],
        "target_ids": sorted(target_ids),
        "primary_target_rows": primary_rows,
        "gate": gate,
        "data_boundary": {"historical_consumed_mk14_rows_read_posthoc": 1576, "protected_fresh_validation_rows_read": 0, "locked_test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0, "outer_test_labels_used_by_selector": False},
        "single_shot_variable_k_formulation": config["single_shot_variable_k_formulation"],
        "interpretation": "Stage100 tests whether statistical stopping can replace a fixed receptor count. It is a post-hoc nested analysis and cannot authorize hardware or quantum-advantage claims.",
    }
    (root / outputs["result_json"]).parent.mkdir(parents=True, exist_ok=True)
    (root / outputs["result_json"]).write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lines = ["# Stage100 adaptive stopping QUBO", "", "The receptor count is no longer fixed. The primary rule selects the smallest k within one standard error of the best inner-fold BEDROC.", "", "| Target | Adaptive BEDROC | Single BEDROC | Gain |", "|---|---:|---:|---:|"]
    lines.extend(f"| {row['target_id']} | {row['adaptive_bedroc']:.6f} | {row['single_bedroc']:.6f} | {row['gain']:+.6f} |" for row in primary_rows)
    lines.extend(["", f"Mean gain: `{gate['mean_gain_over_single']:+.6f}`", "", f"Worst-target gain: `{gate['worst_target_gain']:+.6f}`", "", f"Nontrivial selections: `{nontrivial_count}/25`", "", f"Go/No-Go: `{'GO' if gate['passes'] else 'NO-GO'}`", ""])
    (root / outputs["report_md"]).parent.mkdir(parents=True, exist_ok=True)
    (root / outputs["report_md"]).write_text("\n".join(lines), encoding="ascii")
    print(json.dumps({"status": result["status"], "gate": gate, "primary_target_rows": primary_rows}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
