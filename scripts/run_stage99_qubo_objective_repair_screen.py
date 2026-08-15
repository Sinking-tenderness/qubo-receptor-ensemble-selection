"""Screen one frozen, robust pair-QUBO repair on existing docking matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold


META_COLUMNS = {"ligand_id", "label", "selection_role", "split_group_id", "target_id"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_matrix(path: Path) -> tuple[list[str], list[dict[str, str]], np.ndarray]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"empty score matrix: {path}")
    receptors = [name for name in rows[0] if name not in META_COLUMNS]
    values = np.asarray([[float(row[name]) for name in receptors] for row in rows], dtype=float)
    return receptors, rows, values


def load_target(root: Path, spec: dict[str, Any]) -> tuple[list[str], list[dict[str, str]], np.ndarray]:
    matrix_paths = [spec["matrix"]] if "matrix" in spec else spec["matrix_parts"]
    manifest_paths = [spec["manifest"]] if "manifest" in spec else spec["manifest_parts"]
    all_rows: list[dict[str, str]] = []
    all_values: list[np.ndarray] = []
    receptors: list[str] | None = None
    for index, (matrix_path, manifest_path) in enumerate(zip(matrix_paths, manifest_paths)):
        matrix_file = root / matrix_path
        manifest_file = root / manifest_path
        expected_matrix = spec["matrix_sha256"] if isinstance(spec["matrix_sha256"], str) else spec["matrix_sha256"][index]
        expected_manifest = spec["manifest_sha256"] if isinstance(spec["manifest_sha256"], str) else spec["manifest_sha256"][index]
        if sha256(matrix_file) != expected_matrix:
            raise ValueError(f"matrix hash mismatch: {matrix_path}")
        if sha256(manifest_file) != expected_manifest:
            raise ValueError(f"manifest hash mismatch: {manifest_path}")
        current_receptors, matrix_rows, values = read_matrix(matrix_file)
        manifest_rows = read_csv(manifest_file)
        by_id = {row["ligand_id"]: row for row in manifest_rows}
        if receptors is None:
            receptors = current_receptors
        elif receptors != current_receptors:
            raise ValueError("receptor columns differ between matrix parts")
        if len(matrix_rows) != len(manifest_rows):
            raise ValueError("matrix and manifest row counts differ")
        for row in matrix_rows:
            if row["ligand_id"] not in by_id:
                raise ValueError(f"ligand missing from manifest: {row['ligand_id']}")
            merged = dict(by_id[row["ligand_id"]])
            merged["matrix_label"] = row.get("label", "")
            all_rows.append(merged)
        all_values.append(values)
    if receptors is None:
        raise ValueError("no target data")
    return receptors, all_rows, np.vstack(all_values)


def label_array(rows: list[dict[str, str]]) -> np.ndarray:
    labels = []
    for row in rows:
        label = row.get("label", row.get("matrix_label", ""))
        labels.append(1 if label in {"active", "high"} else 0 if label in {"decoy", "low"} else -1)
    return np.asarray(labels, dtype=int)


def percentile_ranks(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        sorted_reference = np.sort(reference[:, column])
        result[:, column] = np.searchsorted(sorted_reference, values[:, column], side="right") / len(sorted_reference)
    return result


def bedroc(scores: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    mask = np.isfinite(scores) & (labels >= 0)
    scores = scores[mask]
    labels = labels[mask]
    total = len(scores)
    active_count = int(np.sum(labels == 1))
    if total == 0 or active_count == 0 or active_count == total:
        return math.nan
    ordered_labels = labels[np.argsort(scores, kind="stable")]
    weights = np.exp(-alpha * np.arange(1, total + 1, dtype=float) / total)
    expected = active_count * float(np.sum(weights)) / total
    observed = float(np.sum(weights[ordered_labels == 1]) / expected)
    best = float(np.sum(weights[:active_count]) / expected)
    worst = float(np.sum(weights[-active_count:]) / expected)
    return (observed - worst) / (best - worst)


def aggregate(ranks: np.ndarray, mode: str) -> np.ndarray:
    if ranks.shape[1] == 1 or mode == "minimum":
        return np.min(ranks, axis=1)
    return np.mean(np.sort(ranks, axis=1)[:, : min(2, ranks.shape[1])], axis=1)


def utility(ranks: np.ndarray, labels: np.ndarray, mode: str, alpha: float, decoy_penalty: float) -> float:
    active = labels == 1
    decoy = labels == 0
    transformed = np.exp(-alpha * aggregate(ranks, mode))
    if not np.any(active) or not np.any(decoy):
        raise ValueError("training fold lacks both classes")
    return float(np.mean(transformed[active]) - decoy_penalty * np.mean(transformed[decoy]))


def coefficients(ranks: np.ndarray, labels: np.ndarray, config: dict[str, Any], variant: str) -> tuple[np.ndarray, np.ndarray]:
    objective = config["objective"]
    alpha = float(objective["bedroc_alpha"])
    penalty = float(objective["decoy_penalty_lambda"])
    receptor_count = ranks.shape[1]
    node = np.asarray([utility(ranks[:, [i]], labels, "minimum", alpha, penalty) for i in range(receptor_count)])
    pair = np.zeros((receptor_count, receptor_count), dtype=float)
    pair_mode = "minimum" if variant == "old" else "two_support_mean"
    for i, j in itertools.combinations(range(receptor_count), 2):
        pair_value = utility(ranks[:, [i, j]], labels, pair_mode, alpha, penalty)
        baseline = max(node[i], node[j]) if variant == "old" else (node[i] + node[j]) / 2.0
        pair[i, j] = pair[j, i] = pair_value - baseline
    return node, pair


def q_value(subset: tuple[int, ...], node: np.ndarray, pair: np.ndarray) -> float:
    value = float(np.mean(node[list(subset)]))
    if len(subset) > 1:
        value += float(np.mean([pair[i, j] for i, j in itertools.combinations(subset, 2)]))
    return value


def exact_select(count: int, k: int, node: np.ndarray, pair: np.ndarray) -> tuple[int, ...]:
    return max(itertools.combinations(range(count), k), key=lambda subset: (q_value(subset, node, pair), tuple(-i for i in subset)))


def greedy_swap(count: int, k: int, node: np.ndarray, pair: np.ndarray) -> tuple[int, ...]:
    selected: tuple[int, ...] = (int(np.argmax(node)),)
    while len(selected) < k:
        candidates = [tuple(sorted((*selected, index))) for index in range(count) if index not in selected]
        selected = max(candidates, key=lambda subset: (q_value(subset, node, pair), tuple(-i for i in subset)))
    while True:
        current_value = q_value(selected, node, pair)
        proposals = []
        for old in selected:
            for new in range(count):
                if new in selected:
                    continue
                candidate = tuple(sorted((set(selected) - {old}) | {new}))
                value = q_value(candidate, node, pair)
                if value > current_value + 1e-12:
                    proposals.append((value, tuple(-i for i in candidate), candidate))
        if not proposals:
            return selected
        selected = max(proposals)[2]


def adaptive_k_selection(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: dict[str, Any],
) -> tuple[int, dict[int, float]]:
    """Choose k using only inner folds of the current outer-training split."""
    sizes = [int(value) for value in config["objective"]["ensemble_sizes"]]
    scores: dict[int, list[float]] = {k: [] for k in sizes}
    fold_count = min(
        int(config["evaluation"]["inner_folds_for_adaptive_k"]),
        len(set(groups.tolist())),
    )
    if fold_count < 2:
        return min(sizes), {k: math.nan for k in sizes}
    splitter = GroupKFold(n_splits=fold_count)
    for train_index, valid_index in splitter.split(values, groups=groups):
        train_index = np.asarray([index for index in train_index if labels[index] >= 0])
        valid_index = np.asarray([index for index in valid_index if labels[index] >= 0])
        train_ranks = percentile_ranks(values[train_index], values[train_index])
        valid_ranks = percentile_ranks(values[train_index], values[valid_index])
        node, pair = coefficients(train_ranks, labels[train_index], config, "repair")
        for k in sizes:
            subset = exact_select(values.shape[1], k, node, pair)
            score = bedroc(
                aggregate(valid_ranks[:, list(subset)], "minimum"),
                labels[valid_index],
                float(config["objective"]["bedroc_alpha"]),
            )
            if math.isfinite(score):
                scores[k].append(score)
    means = {
        k: float(np.mean(per_k)) if per_k else math.nan
        for k, per_k in scores.items()
    }
    finite = [k for k in sizes if math.isfinite(means[k])]
    if not finite:
        return min(sizes), means
    return max(finite, key=lambda k: (means[k], -k)), means


def subset_name(subset: tuple[int, ...], receptors: list[str]) -> str:
    return "+".join(receptors[index] for index in subset)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe(value: float) -> float | None:
    return value if math.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage99_qubo_objective_repair_screen.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    fold_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    target_data: dict[str, tuple[list[str], list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]] = {}
    for target, spec in config["targets"].items():
        receptors, rows, values = load_target(root, spec)
        labels = label_array(rows)
        group_field = spec["group_field"]
        groups = np.asarray([row.get(group_field, row.get("split_group_id", row["ligand_id"])) for row in rows])
        target_data[target] = (receptors, rows, values, labels, groups)
    for target, (receptors, rows, values, labels, groups) in target_data.items():
        splitter = GroupKFold(n_splits=int(config["evaluation"]["outer_folds"]))
        for fold, (train_index, test_index) in enumerate(splitter.split(values, groups=groups), start=1):
            train_index = np.asarray([index for index in train_index if labels[index] >= 0])
            test_index = np.asarray([index for index in test_index if labels[index] >= 0])
            train_values = values[train_index]
            train_labels = labels[train_index]
            test_labels = labels[test_index]
            train_ranks = percentile_ranks(train_values, train_values)
            test_ranks = percentile_ranks(train_values, values[test_index])
            chosen_k, inner_scores = adaptive_k_selection(
                train_values, train_labels, groups[train_index], config
            )
            adaptive_node, adaptive_pair = coefficients(train_ranks, train_labels, config, "repair")
            adaptive_exact = exact_select(len(receptors), chosen_k, adaptive_node, adaptive_pair)
            adaptive_local = greedy_swap(len(receptors), chosen_k, adaptive_node, adaptive_pair)
            adaptive_rows.append({
                "target_id": target,
                "fold": fold,
                "chosen_k": chosen_k,
                "inner_k1_bedroc": safe(inner_scores.get(1, math.nan)),
                "inner_k2_bedroc": safe(inner_scores.get(2, math.nan)),
                "inner_k3_bedroc": safe(inner_scores.get(3, math.nan)),
                "exact_subset": subset_name(adaptive_exact, receptors),
                "one_swap_subset": subset_name(adaptive_local, receptors),
                "outer_exact_bedroc_alpha20": safe(bedroc(aggregate(test_ranks[:, list(adaptive_exact)], "minimum"), test_labels, float(config["objective"]["bedroc_alpha"]))),
                "outer_one_swap_bedroc_alpha20": safe(bedroc(aggregate(test_ranks[:, list(adaptive_local)], "minimum"), test_labels, float(config["objective"]["bedroc_alpha"]))),
                "exact_minus_one_swap_objective": q_value(adaptive_exact, adaptive_node, adaptive_pair) - q_value(adaptive_local, adaptive_node, adaptive_pair),
                "selector_used_outer_test_labels": False,
            })
            for k in config["objective"]["ensemble_sizes"]:
                k = int(k)
                old_node, old_pair = coefficients(train_ranks, train_labels, config, "old")
                repair_node, repair_pair = coefficients(train_ranks, train_labels, config, "repair")
                selections = {
                    "single_best": (int(np.argmax(repair_node)),),
                    "mean_topk": tuple(np.argsort(-repair_node, kind="stable")[:k].tolist()),
                    "old_pair_qubo": exact_select(len(receptors), k, old_node, old_pair),
                    "repair_pair_qubo_exact": exact_select(len(receptors), k, repair_node, repair_pair),
                    "repair_pair_qubo_one_swap": greedy_swap(len(receptors), k, repair_node, repair_pair),
                }
                for method, subset in selections.items():
                    primary = bedroc(aggregate(test_ranks[:, list(subset)], "minimum"), test_labels, float(config["objective"]["bedroc_alpha"]))
                    secondary = bedroc(aggregate(test_ranks[:, list(subset)], "two_support_mean"), test_labels, float(config["objective"]["bedroc_alpha"]))
                    fold_rows.append({"target_id": target, "fold": fold, "ensemble_size": k, "method": method, "selected_receptors": subset_name(subset, receptors), "primary_bedroc_alpha20": safe(primary), "secondary_two_support_bedroc_alpha20": safe(secondary), "selector_used_test_labels": False, "selector_used_train_labels": True})
                exact = selections["repair_pair_qubo_exact"]
                local = selections["repair_pair_qubo_one_swap"]
                solver_rows.append({"target_id": target, "fold": fold, "ensemble_size": k, "exact_subset": subset_name(exact, receptors), "one_swap_subset": subset_name(local, receptors), "exact_value": q_value(exact, repair_node, repair_pair), "one_swap_value": q_value(local, repair_node, repair_pair), "exact_minus_one_swap": q_value(exact, repair_node, repair_pair) - q_value(local, repair_node, repair_pair), "exact_differs": exact != local})
    target_rows: list[dict[str, Any]] = []
    methods = config["evaluation"]["methods"]
    for target in target_data:
        for k in config["objective"]["ensemble_sizes"]:
            for method in methods:
                selected = [row for row in fold_rows if row["target_id"] == target and row["ensemble_size"] == int(k) and row["method"] == method]
                values = [row["primary_bedroc_alpha20"] for row in selected if row["primary_bedroc_alpha20"] is not None]
                target_rows.append({"target_id": target, "ensemble_size": int(k), "method": method, "mean_primary_bedroc_alpha20": safe(float(np.mean(values))), "std_primary_bedroc_alpha20": safe(float(np.std(values, ddof=1)) if len(values) > 1 else math.nan), "fold_count": len(values)})
    def mean_for(target: str, method: str, k: int) -> float:
        row = next(item for item in target_rows if item["target_id"] == target and item["method"] == method and item["ensemble_size"] == k)
        return float(row["mean_primary_bedroc_alpha20"])
    gains = [{"target_id": target, "repair_k3_minus_single": mean_for(target, "repair_pair_qubo_exact", 3) - mean_for(target, "single_best", 1), "repair_k3": mean_for(target, "repair_pair_qubo_exact", 3), "single": mean_for(target, "single_best", 1)} for target in target_data]
    mean_gain = float(np.mean([row["repair_k3_minus_single"] for row in gains]))
    worst_gain = float(np.min([row["repair_k3_minus_single"] for row in gains]))
    positive_targets = sum(row["repair_k3_minus_single"] >= 0.02 for row in gains)
    positive_folds = sum(float(row["primary_bedroc_alpha20"]) >= float(next(item for item in fold_rows if item["target_id"] == row["target_id"] and item["fold"] == row["fold"] and item["ensemble_size"] == 1 and item["method"] == "single_best")["primary_bedroc_alpha20"]) + 0.0 for row in fold_rows if row["ensemble_size"] == 3 and row["method"] == "repair_pair_qubo_exact" and row["primary_bedroc_alpha20"] is not None)
    solver_diff_count = sum(1 for row in solver_rows if row["exact_differs"])
    gate = {"positive_target_count_at_least_02": positive_targets, "mean_gain_over_single": mean_gain, "worst_target_gain": worst_gain, "positive_fold_count_nonnegative_vs_single": positive_folds, "solver_exact_differs_from_one_swap_count": solver_diff_count, "passes": positive_targets >= int(config["gate"]["minimum_positive_target_count"]) and mean_gain >= float(config["gate"]["minimum_mean_primary_gain_over_single"]) and worst_gain >= float(config["gate"]["minimum_worst_target_primary_gain"]) and positive_folds >= int(config["gate"]["minimum_positive_fold_count"])}
    outputs = config["outputs"]
    write_csv(root / outputs["fold_csv"], fold_rows)
    write_csv(root / outputs["target_csv"], target_rows)
    write_csv(root / outputs["solver_csv"], solver_rows)
    write_csv(root / outputs["adaptive_k_csv"], adaptive_rows)
    adaptive_target_rows = []
    for target in target_data:
        target_adaptive = [row for row in adaptive_rows if row["target_id"] == target]
        adaptive_mean = float(np.mean([float(row["outer_exact_bedroc_alpha20"]) for row in target_adaptive]))
        single_mean = mean_for(target, "single_best", 1)
        adaptive_target_rows.append({"target_id": target, "adaptive_k_mean_bedroc": adaptive_mean, "single_mean_bedroc": single_mean, "adaptive_gain_over_single": adaptive_mean - single_mean, "chosen_k_values": [int(row["chosen_k"]) for row in target_adaptive]})
    adaptive_mean_gain = float(np.mean([row["adaptive_gain_over_single"] for row in adaptive_target_rows]))
    adaptive_worst_gain = float(np.min([row["adaptive_gain_over_single"] for row in adaptive_target_rows]))
    adaptive_positive_targets = sum(row["adaptive_gain_over_single"] >= 0.02 for row in adaptive_target_rows)
    gate["fixed_k3_passes"] = gate.pop("passes")
    gate["adaptive_k"] = {"positive_target_count_at_least_02": adaptive_positive_targets, "mean_gain_over_single": adaptive_mean_gain, "worst_target_gain": adaptive_worst_gain}
    gate["passes"] = adaptive_positive_targets >= int(config["gate"]["minimum_positive_target_count"]) and adaptive_mean_gain >= float(config["gate"]["minimum_mean_primary_gain_over_single"]) and adaptive_worst_gain >= float(config["gate"]["minimum_worst_target_primary_gain"])
    result = {"schema_version": "1.0", "status": "stage99_qubo_objective_repair_screen_complete", "objective": config["objective"], "target_ids": sorted(target_data), "target_count": len(target_data), "gate": gate, "data_boundary": {"new_docking_jobs": 0, "quantum_hardware_jobs": 0, "historical_consumed_fresh_validation_rows_read_posthoc": len(target_data["MK14"][1]), "protected_fresh_validation_rows_read": 0, "locked_test_rows_read": 0, "test_labels_used_by_selector": False, "train_labels_used_by_selector": True}, "fixed_k3_target_gain_rows": gains, "adaptive_k_target_gain_rows": adaptive_target_rows, "interpretation": "This is an outer group-fold screen of one frozen supervised repair. MK14's already consumed fresh-validation matrix is used post-hoc; no still-protected fresh-validation or locked-test data are read. The result cannot authorize same-matrix retuning, hardware, or quantum-advantage claims."}
    (root / outputs["result_json"]).parent.mkdir(parents=True, exist_ok=True)
    (root / outputs["result_json"]).write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    audit = {"schema_version": "1.0", "status": "stage99_runner_self_audit_ok", "fold_rows": len(fold_rows), "target_rows": len(target_rows), "solver_rows": len(solver_rows), "adaptive_k_rows": len(adaptive_rows), "labels_used_only_for_outer_train_selector": True, "test_labels_used_by_selector": False, "new_docking_jobs": 0, "quantum_hardware_jobs": 0, "hashes_verified": True}
    (root / outputs["audit_json"]).write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lines = ["# Stage99 QUBO objective repair screen", "", "This screen uses existing docking matrices only. Train labels are used to fit the selector inside each outer fold; test labels are used only for BEDROC evaluation.", "", "## Frozen repair", "", "- Percentile-normalize each receptor on outer-train rows.", "- Use `exp(-20*r)` to match BEDROC alpha=20 early recognition.", "- Define singleton utility as active support minus `1.5` times decoy exposure.", "- Define pair utility from the mean of two receptor ranks, so a single extreme score is insufficient.", "- Select k in {1,2,3} by inner group validation and compare exact QUBO against greedy plus one-swap.", "", "## Gate", "", f"- Fixed-k=3 mean gain over the best single receptor: `{mean_gain:.6f}`", f"- Adaptive-k mean gain over the best single receptor: `{adaptive_mean_gain:.6f}`", f"- Adaptive-k worst-target gain: `{adaptive_worst_gain:.6f}`", f"- Adaptive-k targets with gain >= 0.02: `{adaptive_positive_targets}/{len(target_data)}`", f"- Exact repair solution differs from one-swap in `{solver_diff_count}/{len(solver_rows)}` fixed-k cells", f"- Go/No-Go: `{'GO' if gate['passes'] else 'NO-GO'}`", ""]
    (root / outputs["report_md"]).parent.mkdir(parents=True, exist_ok=True)
    (root / outputs["report_md"]).write_text("\n".join(lines), encoding="ascii")
    print(json.dumps({"status": result["status"], "gate": gate, "fixed_k3_target_gain_rows": gains, "adaptive_k_target_gain_rows": adaptive_target_rows}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
