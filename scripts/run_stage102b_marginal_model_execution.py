"""Execute the operational Stage102B adaptive-cardinality development analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_stage102a_phase_a_results as s102a
from scripts import run_stage100_adaptive_stopping_qubo as s100
from scripts import run_stage99_qubo_objective_repair_screen as s99


@dataclass(frozen=True)
class TargetData:
    target_id: str
    receptors: list[str]
    rows: list[dict[str, str]]
    values: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    outer_folds: np.ndarray | None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def validate_parent_hashes(root: Path, config: dict[str, Any]) -> None:
    for key, expected in config["parent"].items():
        if not key.endswith("_sha256"):
            continue
        path_key = key.removesuffix("_sha256")
        path = root / config["parent"][path_key]
        if sha256(path) != expected:
            raise ValueError(f"parent hash mismatch: {path_key}")


def validate_stage102a_inputs(root: Path, config: dict[str, Any]) -> None:
    for relative, expected in config["stage102a_inputs"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Stage102A input hash mismatch: {relative}")


def load_targets(root: Path, config: dict[str, Any]) -> dict[str, TargetData]:
    stage99 = read_json(root / config["parent"]["stage99_config"])
    target_data: dict[str, TargetData] = {}
    for target_id, spec in stage99["targets"].items():
        receptors, rows, values = s99.load_target(root, spec)
        labels = s99.label_array(rows)
        groups = np.asarray(
            [row.get(spec["group_field"], row.get("split_group_id", row["ligand_id"])) for row in rows]
        )
        target_data[target_id] = TargetData(
            target_id=target_id,
            receptors=receptors,
            rows=rows,
            values=values,
            labels=labels,
            groups=groups,
            outer_folds=None,
        )
    received_root = root / config["targets"]["stage102a_received_root"]
    for target_id in config["targets"]["stage102a_targets"]:
        receptors, rows, values = s102a.load_target(received_root, target_id)
        target_data[target_id] = TargetData(
            target_id=target_id,
            receptors=receptors,
            rows=rows,
            values=values,
            labels=s99.label_array(rows),
            groups=np.asarray([row["split_group_id"] for row in rows]),
            outer_folds=np.asarray([int(row["outer_fold"]) for row in rows]),
        )
    return target_data


def outer_splits(data: TargetData) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if data.outer_folds is not None:
        return [
            (
                int(fold),
                np.flatnonzero(data.outer_folds != fold),
                np.flatnonzero(data.outer_folds == fold),
            )
            for fold in sorted(set(data.outer_folds.tolist()))
        ]
    splitter = GroupKFold(n_splits=5)
    return [
        (fold, train_index, test_index)
        for fold, (train_index, test_index) in enumerate(
            splitter.split(data.values, groups=data.groups), start=1
        )
    ]


def jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    return len(set(left) & set(right)) / len(set(left) | set(right))


def q_gap(count: int, k: int, node: np.ndarray, pair: np.ndarray) -> float:
    """Measure exact-QUBO separation from a deterministic one-swap solution.

    Unlike an exhaustive best-versus-second-best search, this scales to PPARG-MD96
    while retaining an interpretable indicator of classical local-search agreement.
    """
    exact = s99.exact_select(count, k, node, pair)
    local = s99.greedy_swap(count, k, node, pair)
    return float(s99.q_value(exact, node, pair) - s99.q_value(local, node, pair))


def group_bootstrap_deltas(
    previous_scores: np.ndarray,
    current_scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    iterations: int,
    seed: int,
    alpha: float,
) -> np.ndarray:
    group_to_index: dict[str, np.ndarray] = {
        group: np.flatnonzero(groups == group) for group in sorted(set(groups.tolist()))
    }
    group_names = np.asarray(sorted(group_to_index), dtype=object)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        selected = rng.choice(group_names, size=len(group_names), replace=True)
        indices = np.concatenate([group_to_index[group] for group in selected])
        previous = s99.bedroc(previous_scores[indices], labels[indices], alpha)
        current = s99.bedroc(current_scores[indices], labels[indices], alpha)
        if math.isfinite(previous) and math.isfinite(current):
            deltas.append(current - previous)
    if len(deltas) < max(20, iterations // 2):
        raise ValueError("too few valid group-bootstrap replicates")
    return np.asarray(deltas, dtype=float)


def rescue_contrast(
    previous_scores: np.ndarray,
    current_scores: np.ndarray,
    labels: np.ndarray,
    fraction: float,
) -> tuple[float, float, float]:
    top_count = max(1, int(math.ceil(fraction * len(labels))))
    previous_top = set(np.argsort(previous_scores, kind="stable")[:top_count].tolist())
    current_top = set(np.argsort(current_scores, kind="stable")[:top_count].tolist())
    rescued = current_top - previous_top
    active_count = int(np.sum(labels == 1))
    decoy_count = int(np.sum(labels == 0))
    active = sum(labels[index] == 1 for index in rescued) / active_count
    decoy = sum(labels[index] == 0 for index in rescued) / decoy_count
    return float(active), float(decoy), float(active - decoy)


def build_inner_features(
    data: TargetData,
    outer_fold: int,
    outer_train_index: np.ndarray,
    stage99: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[int, list[float]], dict[int, dict[str, Any]]]:
    values = data.values[outer_train_index]
    labels = data.labels[outer_train_index]
    groups = data.groups[outer_train_index]
    candidate_k = [int(value) for value in config["objective"]["candidate_k_values"]]
    splitter = GroupKFold(n_splits=3)
    oof = {k: np.full(len(values), np.nan) for k in candidate_k}
    profiles = {k: [] for k in candidate_k}
    subsets = {k: [] for k in candidate_k}
    gaps = {k: [] for k in candidate_k}
    for inner_fold, (inner_train, inner_valid) in enumerate(
        splitter.split(values, groups=groups), start=1
    ):
        train_ranks = s99.percentile_ranks(values[inner_train], values[inner_train])
        valid_ranks = s99.percentile_ranks(values[inner_train], values[inner_valid])
        node, pair = s99.coefficients(train_ranks, labels[inner_train], stage99, "repair")
        for k in candidate_k:
            subset = s99.exact_select(len(data.receptors), k, node, pair)
            subsets[k].append(subset)
            gaps[k].append(q_gap(len(data.receptors), k, node, pair))
            aggregate = s99.aggregate(valid_ranks[:, list(subset)], "minimum")
            oof[k][inner_valid] = aggregate
            profiles[k].append(
                s99.bedroc(aggregate, labels[inner_valid], float(stage99["objective"]["bedroc_alpha"]))
            )
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise ValueError(f"{data.target_id}/fold{outer_fold}: incomplete out-of-inner-fold ranks")
    features: dict[int, dict[str, Any]] = {}
    bootstrap = config["feature_construction"]["bootstrap"]
    alpha = float(stage99["objective"]["bedroc_alpha"])
    for previous, current in zip(candidate_k, candidate_k[1:]):
        differences = np.asarray(profiles[current]) - np.asarray(profiles[previous])
        deltas = group_bootstrap_deltas(
            oof[previous],
            oof[current],
            labels,
            groups,
            int(bootstrap["iterations"]),
            stable_seed(
                bootstrap["random_seed_namespace"], data.target_id, outer_fold, previous, current
            ),
            alpha,
        )
        active1, decoy1, contrast1 = rescue_contrast(oof[previous], oof[current], labels, 0.01)
        active5, decoy5, contrast5 = rescue_contrast(oof[previous], oof[current], labels, 0.05)
        pairs = list(itertools.combinations(subsets[current], 2))
        stability = float(np.mean([jaccard(left, right) for left, right in pairs])) if pairs else 1.0
        correlation = spearmanr(oof[previous], oof[current]).statistic
        features[current] = {
            "target_id": data.target_id,
            "outer_fold": outer_fold,
            "from_k": previous,
            "to_k": current,
            "inner_mean_gain": float(np.mean(differences)),
            "inner_gain_se": s100.standard_error(differences.tolist()),
            "inner_worst_gain": float(np.min(differences)),
            "inner_best_gain": float(np.max(differences)),
            "inner_positive_fraction": float(np.mean(differences > 0.0)),
            "bootstrap_mean_gain": float(np.mean(deltas)),
            "bootstrap_ci95_lower": float(np.quantile(deltas, bootstrap["ci_lower_quantile"])),
            "bootstrap_positive_probability": float(np.mean(deltas > 0.0)),
            "bootstrap_worst_gain": float(np.min(deltas)),
            "selected_subset_jaccard": stability,
            "active_rescue_top1": active1,
            "decoy_rescue_top1": decoy1,
            "active_rescue_contrast_top1": contrast1,
            "active_rescue_top5": active5,
            "decoy_rescue_top5": decoy5,
            "active_rescue_contrast_top5": contrast5,
            "aggregate_rank_spearman": safe_float(float(correlation)),
            "qubo_optimum_gap": float(np.mean(gaps[current])),
            "incremental_receptor_cost": 1.0 / len(data.receptors),
            "transition_is_k3": int(current == 3),
            "inner_fold_subsets": "|".join(
                s99.subset_name(subset, data.receptors) for subset in subsets[current]
            ),
        }
    return profiles, features


def select_sequential(features: dict[int, dict[str, Any]], rule: str) -> int:
    selected = 1
    for current in (2, 3):
        row = features[current]
        continue_choice = edge_continue_decision(row, rule)
        row[f"{rule}_continue"] = bool(continue_choice)
        if not continue_choice:
            break
        selected = current
    return selected


def edge_continue_decision(row: dict[str, Any], rule: str) -> bool:
    """Evaluate a single transition, including transitions after an early stop."""
    if rule == "mechanistic_bootstrap_lcb":
        return bool(
            row["bootstrap_ci95_lower"] > 0.0
            and (row["active_rescue_contrast_top1"] + row["active_rescue_contrast_top5"]) / 2.0 > 0.0
        )
    if rule == "target_held_out_l2_ridge":
        return bool(float(row["ridge_predicted_outer_gain"]) > 0.0)
    raise ValueError(f"unknown sequential rule: {rule}")


def direct_bedroc_forward_greedy(
    ranks: np.ndarray,
    labels: np.ndarray,
    k: int,
    alpha: float,
) -> tuple[int, ...]:
    selected: tuple[int, ...] = ()
    while len(selected) < k:
        candidates = [
            tuple(sorted((*selected, index)))
            for index in range(ranks.shape[1])
            if index not in selected
        ]
        selected = max(
            candidates,
            key=lambda subset: (
                s99.bedroc(s99.aggregate(ranks[:, list(subset)], "minimum"), labels, alpha),
                tuple(-index for index in subset),
            ),
        )
    return selected


def add_ridge_predictions(edge_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    feature_names = config["candidate_rules"]["target_held_out_l2_ridge"]["features"]
    targets = sorted({str(row["target_id"]) for row in edge_rows})
    models: dict[str, Any] = {}
    for held_target in targets:
        train = [row for row in edge_rows if row["target_id"] != held_target]
        held = [row for row in edge_rows if row["target_id"] == held_target]
        x_train = np.asarray([[float(row[name]) for name in feature_names] for row in train], dtype=float)
        y_train = np.asarray([float(row["outer_gain"]) for row in train], dtype=float)
        x_held = np.asarray([[float(row[name]) for name in feature_names] for row in held], dtype=float)
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x_train, y_train)
        predictions = model.predict(x_held)
        ridge = model.named_steps["ridge"]
        for row, prediction in zip(held, predictions):
            row["ridge_predicted_outer_gain"] = float(prediction)
            row["ridge_training_targets"] = "|".join(target for target in targets if target != held_target)
        models[held_target] = {
            "training_targets": [target for target in targets if target != held_target],
            "intercept": float(ridge.intercept_),
            "standardized_coefficients": {
                name: float(value) for name, value in zip(feature_names, ridge.coef_)
            },
        }
    return models


def outer_results_for_target(
    data: TargetData,
    stage99: dict[str, Any],
    stage100: dict[str, Any],
    edge_lookup: dict[tuple[str, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    alpha = float(stage99["objective"]["bedroc_alpha"])
    result: list[dict[str, Any]] = []
    for outer_fold, train_index, test_index in outer_splits(data):
        train_values = data.values[train_index]
        train_labels = data.labels[train_index]
        train_ranks = s99.percentile_ranks(train_values, train_values)
        test_ranks = s99.percentile_ranks(train_values, data.values[test_index])
        node, pair = s99.coefficients(train_ranks, train_labels, stage99, "repair")
        exact = {k: s99.exact_select(len(data.receptors), k, node, pair) for k in (1, 2, 3)}
        profiles = {k: [] for k in (1, 2, 3)}
        for current in (2, 3):
            edge = edge_lookup[(data.target_id, outer_fold, current)]
            profiles[current] = [0.0] * 3
            profiles[current - 1] = [0.0] * 3
            # The stage100 policies use the same inner values stored by the feature builder.
            gains = np.asarray(json.loads(edge["inner_fold_gains_json"]), dtype=float)
            base = np.asarray(json.loads(edge["previous_inner_scores_json"]), dtype=float)
            profiles[current - 1] = base.tolist()
            profiles[current] = (base + gains).tolist()
        one_se_k, _ = s100.choose_one_standard_error(profiles)
        stage100_lcb_k, _ = s100.choose_marginal_lcb(profiles)
        mechanical_k = select_sequential(
            {2: edge_lookup[(data.target_id, outer_fold, 2)], 3: edge_lookup[(data.target_id, outer_fold, 3)]},
            "mechanistic_bootstrap_lcb",
        )
        ridge_k = select_sequential(
            {2: edge_lookup[(data.target_id, outer_fold, 2)], 3: edge_lookup[(data.target_id, outer_fold, 3)]},
            "target_held_out_l2_ridge",
        )
        exact_scores = {
            k: s99.bedroc(
                s99.aggregate(test_ranks[:, list(exact[k])], "minimum"), data.labels[test_index], alpha
            )
            for k in (1, 2, 3)
        }
        policy_k = {
            "single": 1,
            "fixed_k2": 2,
            "fixed_k3": 3,
            "one_standard_error_smallest_k": one_se_k,
            "stage100_sequential_lcb": stage100_lcb_k,
            "mechanistic_bootstrap_lcb": mechanical_k,
            "target_held_out_l2_ridge": ridge_k,
            "outer_oracle_k": max((1, 2, 3), key=lambda k: (exact_scores[k], -k)),
        }
        for policy, selected_k in policy_k.items():
            choices = {"exact_qubo": exact[selected_k]}
            if policy not in {"single", "outer_oracle_k"}:
                choices["direct_bedroc_forward_greedy"] = direct_bedroc_forward_greedy(
                    train_ranks, train_labels, selected_k, alpha
                )
                choices["mean_singleton_topk"] = tuple(
                    np.argsort(-node, kind="stable")[:selected_k].tolist()
                )
            for method, subset in choices.items():
                score = s99.bedroc(
                    s99.aggregate(test_ranks[:, list(subset)], "minimum"),
                    data.labels[test_index],
                    alpha,
                )
                result.append(
                    {
                        "target_id": data.target_id,
                        "outer_fold": outer_fold,
                        "policy": policy,
                        "selection_method": method,
                        "selected_k": selected_k,
                        "selected_receptors": s99.subset_name(subset, data.receptors),
                        "outer_bedroc_alpha20": score,
                        "gain_over_train_selected_single": score - exact_scores[1],
                        "uses_outer_labels_for_selection": policy == "outer_oracle_k",
                    }
                )
    return result


def target_summary(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in sorted({str(row["target_id"]) for row in fold_rows}):
        for policy in sorted({str(row["policy"]) for row in fold_rows}):
            for method in sorted({str(row["selection_method"]) for row in fold_rows}):
                selected = [
                    row
                    for row in fold_rows
                    if row["target_id"] == target
                    and row["policy"] == policy
                    and row["selection_method"] == method
                ]
                if not selected:
                    continue
                scores = np.asarray([row["outer_bedroc_alpha20"] for row in selected], dtype=float)
                gains = np.asarray([row["gain_over_train_selected_single"] for row in selected], dtype=float)
                rows.append(
                    {
                        "target_id": target,
                        "policy": policy,
                        "selection_method": method,
                        "mean_outer_bedroc_alpha20": float(np.mean(scores)),
                        "std_outer_bedroc_alpha20": float(np.std(scores, ddof=1)),
                        "mean_gain_over_train_selected_single": float(np.mean(gains)),
                        "mean_selected_k": float(np.mean([row["selected_k"] for row in selected])),
                        "selected_k_values": "|".join(str(row["selected_k"]) for row in selected),
                        "fold_count": len(selected),
                    }
                )
    return rows


def policy_gate(
    policy: str,
    summaries: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        row
        for row in summaries
        if row["policy"] == policy and row["selection_method"] == "exact_qubo"
    ]
    gains = np.asarray([row["mean_gain_over_train_selected_single"] for row in rows], dtype=float)
    nontrivial = sum(
        int(value) > 1
        for row in rows
        for value in row["selected_k_values"].split("|")
    )
    new_targets = set(config["targets"]["stage102a_targets"])
    policy_key = f"{policy}_continue"
    new_edges = [row for row in edge_rows if row["target_id"] in new_targets]
    sign_accuracy = float(
        np.mean(
            [bool(row[policy_key]) == (float(row["outer_gain"]) > 0.0) for row in new_edges]
        )
    )
    metrics = {
        "mean_target_gain": float(np.mean(gains)),
        "worst_target_gain": float(np.min(gains)),
        "target_count_gain_at_least_0p02": int(np.sum(gains >= 0.02)),
        "nontrivial_outer_fold_count": int(nontrivial),
        "new_target_count_with_positive_gain": int(
            sum(row["mean_gain_over_train_selected_single"] > 0.0 for row in rows if row["target_id"] in new_targets)
        ),
        "new_target_marginal_sign_accuracy": sign_accuracy,
    }
    gate = config["phase_a_gate"]
    checks = {
        "mean_target_gain": metrics["mean_target_gain"] >= float(gate["minimum_mean_target_gain_over_train_selected_single"]),
        "worst_target_gain": metrics["worst_target_gain"] >= float(gate["minimum_worst_target_gain"]),
        "positive_target_count": metrics["target_count_gain_at_least_0p02"] >= int(gate["minimum_target_count_with_gain_at_least_0p02"]),
        "nontrivial_outer_fold_count": metrics["nontrivial_outer_fold_count"] >= int(gate["minimum_nontrivial_outer_fold_count"]),
        "new_positive_target_count": metrics["new_target_count_with_positive_gain"] >= int(gate["minimum_new_target_count_with_positive_gain"]),
        "new_target_sign_accuracy": metrics["new_target_marginal_sign_accuracy"] >= float(gate["minimum_new_target_sign_accuracy"]),
    }
    return {"metrics": metrics, "checks": checks, "passes": all(checks.values())}


def render_report(
    result: dict[str, Any],
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    lines = [
        "# Stage102B marginal-model execution",
        "",
        "Stage102B operationalizes the two candidate rules named in Stage102. It is a posthoc development amendment, not a PARP1 confirmation or hardware release.",
        "",
        "## Candidate outcomes",
        "",
        "| Policy | Mean target gain | Worst target gain | Positive targets at +0.02 | Nontrivial folds | New-target sign accuracy | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for policy, decision in result["candidate_decisions"].items():
        metrics = decision["metrics"]
        lines.append(
            f"| {policy} | {metrics['mean_target_gain']:+.6f} | {metrics['worst_target_gain']:+.6f} | "
            f"{metrics['target_count_gain_at_least_0p02']} | {metrics['nontrivial_outer_fold_count']} | "
            f"{metrics['new_target_marginal_sign_accuracy']:.2%} | {'GO' if decision['passes'] else 'NO-GO'} |"
        )
    lines.extend([
        "",
        "## Exact-QUBO target outcomes",
        "",
        "| Target | Policy | BEDROC20 | Gain over single | Selected k |",
        "| --- | --- | ---: | ---: | --- |",
    ])
    for row in summaries:
        if row["selection_method"] != "exact_qubo" or row["policy"] not in result["candidate_decisions"]:
            continue
        lines.append(
            f"| {row['target_id']} | {row['policy']} | {row['mean_outer_bedroc_alpha20']:.6f} | "
            f"{row['mean_gain_over_train_selected_single']:+.6f} | {row['selected_k_values']} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        result["decision"]["next_action"],
        "",
        "No new docking, PARP1 rows, locked-test rows, or quantum-hardware jobs were used.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/stage102b_marginal_model_execution_amendment01.json")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config = read_json(root / args.config)
    validate_parent_hashes(root, config)
    validate_stage102a_inputs(root, config)
    stage99 = read_json(root / config["parent"]["stage99_config"])
    stage100 = read_json(root / config["parent"]["stage100_config"])
    targets = load_targets(root, config)

    edge_rows: list[dict[str, Any]] = []
    inner_profiles: dict[tuple[str, int], dict[int, list[float]]] = {}
    for data in targets.values():
        for outer_fold, train_index, _ in outer_splits(data):
            profiles, feature_map = build_inner_features(data, outer_fold, train_index, stage99, config)
            inner_profiles[(data.target_id, outer_fold)] = profiles
            for current, row in feature_map.items():
                previous = current - 1
                row["inner_fold_gains_json"] = json.dumps(
                    (np.asarray(profiles[current]) - np.asarray(profiles[previous])).tolist(), separators=(",", ":")
                )
                row["previous_inner_scores_json"] = json.dumps(profiles[previous], separators=(",", ":"))
                train_values = data.values[train_index]
                train_labels = data.labels[train_index]
                test_index = next(
                    test for fold, _, test in outer_splits(data) if fold == outer_fold
                )
                train_ranks = s99.percentile_ranks(train_values, train_values)
                test_ranks = s99.percentile_ranks(train_values, data.values[test_index])
                node, pair = s99.coefficients(train_ranks, train_labels, stage99, "repair")
                exact_previous = s99.exact_select(len(data.receptors), previous, node, pair)
                exact_current = s99.exact_select(len(data.receptors), current, node, pair)
                row["outer_previous_bedroc"] = s99.bedroc(
                    s99.aggregate(test_ranks[:, list(exact_previous)], "minimum"),
                    data.labels[test_index],
                    float(stage99["objective"]["bedroc_alpha"]),
                )
                row["outer_current_bedroc"] = s99.bedroc(
                    s99.aggregate(test_ranks[:, list(exact_current)], "minimum"),
                    data.labels[test_index],
                    float(stage99["objective"]["bedroc_alpha"]),
                )
                row["outer_gain"] = row["outer_current_bedroc"] - row["outer_previous_bedroc"]
                edge_rows.append(row)

    ridge_models = add_ridge_predictions(edge_rows, config)
    for row in edge_rows:
        for policy in ("mechanistic_bootstrap_lcb", "target_held_out_l2_ridge"):
            row[f"{policy}_continue"] = edge_continue_decision(row, policy)
    edge_lookup = {
        (str(row["target_id"]), int(row["outer_fold"]), int(row["to_k"])): row
        for row in edge_rows
    }
    fold_rows: list[dict[str, Any]] = []
    for data in targets.values():
        fold_rows.extend(outer_results_for_target(data, stage99, stage100, edge_lookup))
    summaries = target_summary(fold_rows)
    decisions = {
        policy: policy_gate(policy, summaries, edge_rows, config)
        for policy in ("mechanistic_bootstrap_lcb", "target_held_out_l2_ridge")
    }
    passing = [policy for policy, decision in decisions.items() if decision["passes"]]
    if passing:
        winner = sorted(
            passing,
            key=lambda policy: (
                -decisions[policy]["metrics"]["mean_target_gain"],
                -decisions[policy]["metrics"]["worst_target_gain"],
                decisions[policy]["metrics"]["nontrivial_outer_fold_count"],
                policy != "mechanistic_bootstrap_lcb",
            ),
        )[0]
        next_action = (
            f"Candidate {winner} passed the amended development gate. It may be reviewed for an untouched PARP1 protocol; "
            "PARP1 is not automatically released by this development result."
        )
    else:
        winner = None
        next_action = config["phase_a_gate"]["failure_action"]
    outputs = config["outputs"]
    result = {
        "schema_version": "1.0",
        "status": "stage102b_marginal_model_execution_complete",
        "evidence_status": config["evidence_status"],
        "target_ids": sorted(targets),
        "candidate_decisions": decisions,
        "selected_candidate": winner,
        "decision": {
            "phase_a_gate_passes": winner is not None,
            "parp1_released": False,
            "quantum_hardware_authorized": False,
            "next_action": next_action,
        },
        "data_boundary": config["data_boundary"],
        "interpretation": "This analysis evaluates whether frozen operational marginal rules generalize across seven development targets. It cannot establish confirmation or quantum advantage.",
    }
    write_csv(root / outputs["edge_csv"], edge_rows)
    write_csv(root / outputs["fold_csv"], fold_rows)
    write_csv(root / outputs["target_csv"], summaries)
    model_path = root / outputs["model_json"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(ridge_models, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    result_path = root / outputs["result_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report_path = root / outputs["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result, summaries, config), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
