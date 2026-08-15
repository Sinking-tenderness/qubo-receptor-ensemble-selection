"""Diagnose and calibrate adaptive receptor-count marginal signals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def standard_error(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def correlation(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    inner = np.asarray([row["inner_mean_gain"] for row in rows], dtype=float)
    outer = np.asarray([row["outer_gain"] for row in rows], dtype=float)
    spear = spearmanr(inner, outer)
    pear = pearsonr(inner, outer)
    return {
        "edge_count": len(rows),
        "spearman_r": float(spear.statistic),
        "spearman_p": float(spear.pvalue),
        "pearson_r": float(pear.statistic),
        "pearson_p": float(pear.pvalue),
        "sign_accuracy": float(np.mean(np.sign(inner) == np.sign(outer))),
        "inner_positive_count": int(np.sum(inner > 0.0)),
        "outer_positive_count": int(np.sum(outer > 0.0)),
        "false_positive_count": int(np.sum((inner > 0.0) & (outer <= 0.0))),
    }


def build_edges(
    inner_rows: list[dict[str, str]],
    outer_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], float]]:
    profiles: dict[tuple[str, int, int], dict[int, float]] = defaultdict(dict)
    for row in inner_rows:
        profiles[(row["target_id"], int(row["outer_fold"]), int(row["k"]))][int(row["inner_fold"])] = float(row["bedroc_alpha20"])
    scores: dict[tuple[str, int, int], float] = {}
    for row in outer_rows:
        if row["method"] == "repair_pair_qubo_exact":
            scores[(row["target_id"], int(row["fold"]), int(row["ensemble_size"]))] = float(row["primary_bedroc_alpha20"])
    edges: list[dict[str, Any]] = []
    for target in sorted({key[0] for key in scores}):
        for fold in range(1, 6):
            for current in (2, 3):
                previous = current - 1
                previous_profile = profiles[(target, fold, previous)]
                current_profile = profiles[(target, fold, current)]
                inner_folds = sorted(set(previous_profile) & set(current_profile))
                differences = np.asarray([current_profile[index] - previous_profile[index] for index in inner_folds], dtype=float)
                previous_values = np.asarray([previous_profile[index] for index in inner_folds], dtype=float)
                current_values = np.asarray([current_profile[index] for index in inner_folds], dtype=float)
                edges.append({
                    "target_id": target,
                    "outer_fold": fold,
                    "from_k": previous,
                    "to_k": current,
                    "inner_mean_gain": float(np.mean(differences)),
                    "inner_gain_se": standard_error(differences),
                    "inner_worst_gain": float(np.min(differences)),
                    "inner_best_gain": float(np.max(differences)),
                    "inner_positive_fraction": float(np.mean(differences > 0.0)),
                    "previous_inner_mean": float(np.mean(previous_values)),
                    "current_inner_mean": float(np.mean(current_values)),
                    "transition_is_k3": int(current == 3),
                    "inner_fold_gains": "|".join(f"{value:.12g}" for value in differences),
                    "outer_previous_bedroc": scores[(target, fold, previous)],
                    "outer_current_bedroc": scores[(target, fold, current)],
                    "outer_gain": scores[(target, fold, current)] - scores[(target, fold, previous)],
                    "loto_predicted_outer_gain": "",
                    "loto_training_targets": "",
                })
    return edges, scores


def add_loto_predictions(edges: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    features = list(config["calibration"]["features"])
    alpha = float(config["calibration"]["alpha"])
    target_models: dict[str, Any] = {}
    targets = sorted({row["target_id"] for row in edges})
    for held_target in targets:
        train = [row for row in edges if row["target_id"] != held_target]
        test = [row for row in edges if row["target_id"] == held_target]
        pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        x_train = np.asarray([[float(row[name]) for name in features] for row in train], dtype=float)
        y_train = np.asarray([row["outer_gain"] for row in train], dtype=float)
        x_test = np.asarray([[float(row[name]) for name in features] for row in test], dtype=float)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        training_targets = sorted(set(targets) - {held_target})
        for row, prediction in zip(test, predictions):
            row["loto_predicted_outer_gain"] = float(prediction)
            row["loto_training_targets"] = "|".join(training_targets)
        ridge = pipeline.named_steps["ridge"]
        target_models[held_target] = {
            "training_targets": training_targets,
            "intercept": float(ridge.intercept_),
            "standardized_coefficients": {name: float(value) for name, value in zip(features, ridge.coef_)},
        }
    return target_models


def select_sequential(
    target: str,
    fold: int,
    edge_lookup: dict[tuple[str, int, int], dict[str, Any]],
    should_continue: Callable[[dict[str, Any]], bool],
) -> int:
    selected = 1
    for current in (2, 3):
        edge = edge_lookup[(target, fold, current)]
        if not should_continue(edge):
            break
        selected = current
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage101_marginal_transfer_gate.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    parents = config["parents"]
    for path_key, hash_key in (
        ("stage100_config", "stage100_config_sha256"),
        ("stage100_result", "stage100_result_sha256"),
        ("inner_profiles", "inner_profiles_sha256"),
        ("outer_fold_scores", "outer_fold_scores_sha256"),
    ):
        if sha256(root / parents[path_key]) != parents[hash_key]:
            raise ValueError(f"parent hash mismatch: {path_key}")
    inner_rows = read_csv(root / parents["inner_profiles"])
    outer_rows = read_csv(root / parents["outer_fold_scores"])
    edges, scores = build_edges(inner_rows, outer_rows)
    models = add_loto_predictions(edges, config)
    edge_lookup = {(row["target_id"], int(row["outer_fold"]), int(row["to_k"])): row for row in edges}
    targets = sorted({row["target_id"] for row in edges})
    policy_rules: dict[str, Callable[[dict[str, Any]], bool] | None] = {}
    for z in config["policy_sensitivity"]["lcb_z_values"]:
        name = f"lcb_z{str(z).replace('.', 'p')}"
        policy_rules[name] = lambda edge, z=float(z): float(edge["inner_mean_gain"]) - z * float(edge["inner_gain_se"]) > 0.0
    policy_rules.update({
        "all_inner_positive": lambda edge: float(edge["inner_worst_gain"]) > 0.0,
        "two_of_three_positive": lambda edge: float(edge["inner_positive_fraction"]) >= (2.0 / 3.0),
        "loto_ridge": lambda edge: float(edge["loto_predicted_outer_gain"]) > 0.0,
        "always_single": None,
        "outer_oracle_k": None,
    })
    fold_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for policy, rule in policy_rules.items():
        for target in targets:
            target_scores: list[float] = []
            single_scores: list[float] = []
            selected_values: list[int] = []
            for fold in range(1, 6):
                if policy == "always_single":
                    selected = 1
                elif policy == "outer_oracle_k":
                    selected = max((1, 2, 3), key=lambda k: (scores[(target, fold, k)], -k))
                else:
                    assert rule is not None
                    selected = select_sequential(target, fold, edge_lookup, rule)
                score = scores[(target, fold, selected)]
                single = scores[(target, fold, 1)]
                target_scores.append(score)
                single_scores.append(single)
                selected_values.append(selected)
                fold_rows.append({
                    "policy": policy,
                    "target_id": target,
                    "outer_fold": fold,
                    "selected_k": selected,
                    "outer_bedroc_alpha20": score,
                    "single_bedroc_alpha20": single,
                    "gain_over_single": score - single,
                    "uses_held_target_outer_labels": policy == "outer_oracle_k",
                })
            policy_rows.append({
                "policy": policy,
                "target_id": target,
                "mean_outer_bedroc_alpha20": float(np.mean(target_scores)),
                "single_bedroc_alpha20": float(np.mean(single_scores)),
                "gain_over_single": float(np.mean(target_scores) - np.mean(single_scores)),
                "mean_selected_k": float(np.mean(selected_values)),
                "selected_k_values": "|".join(map(str, selected_values)),
                "nontrivial_fold_count": int(np.sum(np.asarray(selected_values) > 1)),
                "uses_held_target_outer_labels": policy == "outer_oracle_k",
            })
    correlations = {
        "all_edges": correlation(edges),
        "k1_to_k2": correlation([row for row in edges if row["to_k"] == 2]),
        "k2_to_k3": correlation([row for row in edges if row["to_k"] == 3]),
    }
    aggregate_policies: dict[str, Any] = {}
    for policy in policy_rules:
        rows = [row for row in policy_rows if row["policy"] == policy]
        gains = np.asarray([row["gain_over_single"] for row in rows], dtype=float)
        aggregate_policies[policy] = {
            "mean_target_gain": float(np.mean(gains)),
            "worst_target_gain": float(np.min(gains)),
            "positive_target_count_at_0p02": int(np.sum(gains >= 0.02)),
            "nontrivial_fold_count": int(sum(row["nontrivial_fold_count"] for row in rows)),
        }
    loto = aggregate_policies["loto_ridge"]
    gate_spec = config["candidate_gate"]
    loto["candidate_gate_passes"] = (
        loto["mean_target_gain"] >= float(gate_spec["minimum_loto_mean_target_gain"])
        and loto["worst_target_gain"] >= float(gate_spec["minimum_loto_worst_target_gain"])
        and loto["positive_target_count_at_0p02"] >= int(gate_spec["minimum_positive_target_count_at_0p02"])
        and loto["nontrivial_fold_count"] >= int(gate_spec["minimum_nontrivial_fold_count"])
    )
    outputs = config["outputs"]
    write_csv(root / outputs["edge_csv"], edges)
    write_csv(root / outputs["policy_csv"], policy_rows)
    write_csv(root / outputs["fold_csv"], fold_rows)
    result = {
        "schema_version": "1.0",
        "status": "stage101_marginal_transfer_gate_complete",
        "correlations": correlations,
        "aggregate_policies": aggregate_policies,
        "loto_models": models,
        "oracle_ceiling_mean_target_gain": aggregate_policies["outer_oracle_k"]["mean_target_gain"],
        "decision": {
            "loto_candidate_gate_passes": loto["candidate_gate_passes"],
            "hardware_authorized": False,
            "same_matrix_threshold_tuning_allowed": False,
            "next_action": config["decision_policy"]["if_loto_candidate_passes"] if loto["candidate_gate_passes"] else config["decision_policy"]["if_loto_candidate_fails"],
        },
        "data_boundary": {
            "protected_fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
            "held_target_outer_labels_visible_to_loto_model": False,
        },
    }
    result_path = root / outputs["result_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage101 marginal transfer gate",
        "",
        "Stage101 asks whether inner-fold evidence predicts the held-out value of adding another receptor. It does not tune a new QUBO on held-out labels.",
        "",
        "| Transition | Spearman | p-value | Sign accuracy | False positives |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in (("k=1 to 2", "k1_to_k2"), ("k=2 to 3", "k2_to_k3"), ("All", "all_edges")):
        row = correlations[key]
        lines.append(f"| {label} | {row['spearman_r']:+.3f} | {row['spearman_p']:.4f} | {row['sign_accuracy']:.2%} | {row['false_positive_count']} |")
    lines.extend(["", "## Policy outcomes", "", "| Policy | Mean target gain | Worst target gain | Nontrivial folds |", "|---|---:|---:|---:|"])
    for policy, row in aggregate_policies.items():
        lines.append(f"| {policy} | {row['mean_target_gain']:+.6f} | {row['worst_target_gain']:+.6f} | {row['nontrivial_fold_count']}/25 |")
    lines.extend([
        "",
        f"LOTO candidate gate: `{'GO' if loto['candidate_gate_passes'] else 'NO-GO'}`",
        "",
        f"Outer-oracle adaptive-k ceiling: `{aggregate_policies['outer_oracle_k']['mean_target_gain']:+.6f}` mean target gain.",
        "",
        "Interpretation: a useful variable-k solution exists in these folds, but the current inner-fold marginal signal does not identify it reliably. No more threshold tuning is allowed on these same matrices.",
        "",
    ])
    report_path = root / outputs["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="ascii")
    print(json.dumps({"status": result["status"], "correlations": correlations, "loto": loto, "oracle_ceiling": result["oracle_ceiling_mean_target_gain"], "decision": result["decision"]}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
