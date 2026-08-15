from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold


META_COLUMNS = {"ligand_id", "label", "selection_role", "split_group_id", "target_id", "source", "selection_role"}


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
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"matrix has no header: {path}")
        receptors = [field for field in reader.fieldnames if field not in META_COLUMNS]
        rows = list(reader)
    values = np.asarray([[float(row[name]) for name in receptors] for row in rows], dtype=float)
    return receptors, rows, values


def load_target(root: Path, spec: dict[str, Any]) -> tuple[list[str], list[dict[str, str]], np.ndarray]:
    matrix_paths = [spec["matrix"]] if "matrix" in spec else spec["matrix_parts"]
    manifest_paths = [spec["manifest"]] if "manifest" in spec else spec["manifest_parts"]
    all_rows: list[dict[str, str]] = []
    matrix_values: list[np.ndarray] = []
    receptors: list[str] | None = None
    for matrix_path, manifest_path in zip(matrix_paths, manifest_paths):
        matrix_index = matrix_paths.index(matrix_path)
        manifest_index = manifest_paths.index(manifest_path)
        expected_matrix_hash = spec["matrix_sha256"] if isinstance(spec["matrix_sha256"], str) else spec["matrix_sha256"][matrix_index]
        expected_manifest_hash = spec["manifest_sha256"] if isinstance(spec["manifest_sha256"], str) else spec["manifest_sha256"][manifest_index]
        if expected_matrix_hash and sha256(root / matrix_path) != expected_matrix_hash:
            raise ValueError(f"matrix hash mismatch: {matrix_path}")
        if expected_manifest_hash and sha256(root / manifest_path) != expected_manifest_hash:
            raise ValueError(f"manifest hash mismatch: {manifest_path}")
        current_receptors, rows, values = read_matrix(root / matrix_path)
        manifest_rows = read_csv(root / manifest_path)
        if receptors is None:
            receptors = current_receptors
        elif receptors != current_receptors:
            raise ValueError("matrix receptor columns differ within target")
        if len(rows) != len(manifest_rows):
            raise ValueError("matrix and manifest row counts differ")
        manifest_by_id = {row["ligand_id"]: row for row in manifest_rows}
        merged_rows = []
        for row in rows:
            if row["ligand_id"] not in manifest_by_id:
                raise ValueError(f"matrix ligand missing from manifest: {row['ligand_id']}")
            merged = dict(manifest_by_id[row["ligand_id"]])
            merged["matrix_label"] = row["label"]
            merged_rows.append(merged)
        all_rows.extend(merged_rows)
        matrix_values.append(values)
    if receptors is None:
        raise ValueError("no matrix parts")
    return receptors, all_rows, np.vstack(matrix_values)


def labels_for(rows: list[dict[str, str]]) -> np.ndarray:
    values = []
    for row in rows:
        label = row.get("label", row.get("matrix_label", ""))
        values.append(1 if label in {"active", "high"} else 0 if label in {"decoy", "low"} else -1)
    return np.asarray(values, dtype=int)


def bedroc(scores: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    mask = np.isfinite(scores) & (labels >= 0)
    scores = scores[mask]
    labels = labels[mask]
    if len(scores) == 0:
        return math.nan
    order = np.argsort(scores, kind="stable")
    ordered = labels[order]
    total = len(ordered)
    active_ranks = [rank for rank, label in enumerate(ordered, start=1) if label == 1]
    active_total = len(active_ranks)
    if active_total == 0 or active_total == total:
        return math.nan

    def weight_sum(ranks: list[int]) -> float:
        return sum(math.exp(-alpha * rank / total) for rank in ranks)

    expected = active_total * weight_sum(list(range(1, total + 1))) / total
    observed = weight_sum(active_ranks) / expected
    best = weight_sum(list(range(1, active_total + 1))) / expected
    worst = weight_sum(list(range(total - active_total + 1, total + 1))) / expected
    return (observed - worst) / (best - worst)


def zscore(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values))
    return np.zeros_like(values) if scale < 1e-12 else (values - float(np.mean(values))) / scale


def pairwise_complementarity(values: np.ndarray) -> np.ndarray:
    ranked = np.asarray([rankdata(values[:, index], method="average") for index in range(values.shape[1])], dtype=float)
    correlation = np.corrcoef(ranked)
    correlation = np.nan_to_num(correlation, nan=0.0)
    return 1.0 - np.abs(correlation)


def select_mean(train_values: np.ndarray, k: int) -> tuple[int, ...]:
    means = np.mean(train_values, axis=0)
    return tuple(np.argsort(means, kind="stable")[:k].tolist())


def subset_objective(subset: tuple[int, ...], node: np.ndarray, complementarity: np.ndarray, weight: float) -> float:
    node_term = float(np.sum(node[list(subset)]))
    if len(subset) < 2:
        return node_term
    pair_term = sum(complementarity[i, j] for i, j in itertools.combinations(subset, 2))
    return node_term + weight * pair_term


def select_complementarity(train_values: np.ndarray, k: int, weight: float) -> tuple[int, ...]:
    node = -zscore(np.mean(train_values, axis=0))
    complementarity = pairwise_complementarity(train_values)
    candidates = itertools.combinations(range(train_values.shape[1]), k)
    return max(candidates, key=lambda subset: (subset_objective(subset, node, complementarity, weight), tuple(-i for i in subset)))


def select_oracle(train_values: np.ndarray, train_labels: np.ndarray, k: int, alpha: float) -> tuple[int, ...]:
    candidates = list(itertools.combinations(range(train_values.shape[1]), k))
    if not candidates:
        raise ValueError("empty oracle candidate set")
    best_value = -math.inf
    best_subset: tuple[int, ...] | None = None
    for start in range(0, len(candidates), 8192):
        chunk = candidates[start : start + 8192]
        index_array = np.asarray(chunk, dtype=int)
        candidate_scores = np.min(train_values[:, index_array], axis=2).T
        values = bedroc_batch(candidate_scores, train_labels, alpha)
        local_index = int(np.nanargmax(values))
        local_value = float(values[local_index])
        local_subset = chunk[local_index]
        if best_subset is None or (local_value, tuple(-i for i in local_subset)) > (best_value, tuple(-i for i in best_subset)):
            best_value = local_value
            best_subset = local_subset
    if best_subset is None:
        raise ValueError("oracle produced no finite candidate")
    return best_subset


def select_random(receptor_count: int, k: int, rng: random.Random) -> tuple[int, ...]:
    return tuple(sorted(rng.sample(range(receptor_count), k)))


def evaluate_subset(values: np.ndarray, labels: np.ndarray, subset: tuple[int, ...], alpha: float) -> float:
    return bedroc(np.min(values[:, subset], axis=1), labels, alpha)


def bedroc_batch(score_matrix: np.ndarray, labels: np.ndarray, alpha: float) -> np.ndarray:
    """Compute BEDROC for many candidate score vectors at once."""
    mask = labels >= 0
    scores = score_matrix[:, mask]
    eval_labels = labels[mask]
    total = scores.shape[1]
    active_total = int(np.sum(eval_labels == 1))
    if total == 0 or active_total == 0 or active_total == total:
        return np.full(score_matrix.shape[0], np.nan)
    order = np.argsort(scores, axis=1, kind="stable")
    ordered_labels = np.take(eval_labels, order)
    weights = np.exp(-alpha * np.arange(1, total + 1, dtype=float) / total)
    expected = active_total * float(np.sum(weights)) / total
    observed = np.sum(ordered_labels * weights[None, :], axis=1) / expected
    best = float(np.sum(weights[:active_total]) / expected)
    worst = float(np.sum(weights[total - active_total :]) / expected)
    return (observed - worst) / (best - worst)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: float) -> float | None:
    return value if math.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage98_cross_target_receptor_complementarity.json"))
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    fold_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    target_data: dict[str, tuple[list[str], list[dict[str, str]], np.ndarray]] = {}
    rng_base = int(config["evaluation"]["random_seed"])
    alpha = float(config["evaluation"]["bedroc_alpha"])
    weight = float(config["evaluation"]["diversity_weight"])
    methods = ["mean_score", "complementarity", "oracle_train", "random"]
    sizes = [int(value) for value in config["evaluation"]["ensemble_sizes"]]
    target_specs = config["targets"]
    if args.target is not None:
        if args.target not in target_specs:
            raise ValueError(f"unknown target: {args.target}")
        target_specs = {args.target: target_specs[args.target]}
    for target, spec in target_specs.items():
        receptors, rows, values = load_target(root, spec)
        labels = labels_for(rows)
        groups = np.asarray([row.get(spec["split_group_field"], row.get("split_group_id", row["ligand_id"])) for row in rows])
        if len(set(groups)) < int(config["evaluation"]["folds"]):
            raise ValueError(f"not enough groups for {target}")
        target_data[target] = (receptors, rows, values)
        complementarity = pairwise_complementarity(values)
        for i in range(len(receptors)):
            for j in range(i + 1, len(receptors)):
                pair_rows.append({"target_id": target, "receptor_i": receptors[i], "receptor_j": receptors[j], "spearman_abs_complementarity": complementarity[i, j], "score_mean_i": float(np.mean(values[:, i])), "score_mean_j": float(np.mean(values[:, j]))})
        splitter = GroupKFold(n_splits=int(config["evaluation"]["folds"]))
        fold_metrics: dict[tuple[str, int], list[float]] = {}
        for fold, (train_index, test_index) in enumerate(splitter.split(values, groups=groups), start=1):
            train_values = values[train_index]
            test_values = values[test_index]
            train_labels = labels[train_index]
            test_labels = labels[test_index]
            for k in sizes:
                selections: dict[str, list[tuple[int, ...]]] = {
                    "mean_score": [select_mean(train_values, k)],
                    "complementarity": [select_complementarity(train_values, k, weight)],
                    "oracle_train": [select_oracle(train_values, train_labels, k, alpha)],
                }
                random_rng = random.Random(rng_base + fold * 1000 + k * 17 + sum(ord(c) for c in target))
                selections["random"] = [select_random(values.shape[1], k, random_rng) for _ in range(int(config["evaluation"]["random_replicates"]))]
                for method, subsets in selections.items():
                    scores = [evaluate_subset(test_values, test_labels, subset, alpha) for subset in subsets]
                    finite = [score for score in scores if math.isfinite(score)]
                    selected = subsets[0]
                    fold_metrics.setdefault((method, k), []).append(float(np.mean(finite)) if finite else math.nan)
                    fold_rows.append({"target_id": target, "fold": fold, "method": method, "ensemble_size": k, "selected_receptors": "|".join(receptors[i] for i in selected), "test_bedroc_alpha_20": safe_float(float(np.mean(finite)) if finite else math.nan), "random_replicates": len(subsets) if method == "random" else 1, "selector_used_labels": method == "oracle_train"})
        for method in methods:
            for k in sizes:
                scores = [value for value in fold_metrics.get((method, k), []) if math.isfinite(value)]
                target_rows.append({"target_id": target, "method": method, "ensemble_size": k, "mean_test_bedroc_alpha_20": safe_float(float(np.mean(scores)) if scores else math.nan), "std_test_bedroc_alpha_20": safe_float(float(np.std(scores, ddof=1)) if len(scores) > 1 else math.nan), "fold_count": len(scores), "selector_used_labels": method == "oracle_train", "data_role": spec["data_role"]})

    def mean_for(target: str, method: str, k: int) -> float:
        row = next(row for row in target_rows if row["target_id"] == target and row["method"] == method and row["ensemble_size"] == k)
        return float(row["mean_test_bedroc_alpha_20"]) if row["mean_test_bedroc_alpha_20"] is not None else math.nan

    gains = []
    target_gate_rows = []
    for target in target_specs:
        baseline = mean_for(target, "complementarity", 1)
        k3 = mean_for(target, "complementarity", 3)
        gain = k3 - baseline
        gains.append(gain)
        target_gate_rows.append({"target_id": target, "complementarity_k1": baseline, "complementarity_k3": k3, "k3_gain": gain, "positive_gain": gain >= 0.02})
    positive_count = sum(row["positive_gain"] for row in target_gate_rows)
    mean_gain = float(np.mean(gains))
    worst_gain = float(np.min(gains))
    gate_passes = len(target_gate_rows) >= int(config["go_gate"]["minimum_target_count"]) and positive_count >= int(config["go_gate"]["minimum_targets_with_positive_k3_gain"]) and mean_gain >= float(config["go_gate"]["minimum_mean_k3_gain"]) and worst_gain >= float(config["go_gate"]["minimum_worst_target_k3_gain"])
    result = {"schema_version": "1.0", "status": "stage98_cross_target_receptor_complementarity_complete", "central_claim": config["central_claim"], "target_count": len(target_data), "target_ids": sorted(target_data), "gate": {"target_rows": target_gate_rows, "positive_target_count": positive_count, "mean_k3_gain": mean_gain, "worst_k3_gain": worst_gain, "passes": gate_passes}, "audit": {"selector_labels_allowed": False, "oracle_train_is_evaluation_only": True, "new_docking_jobs": 0, "quantum_hardware_jobs": 0, "synthetic_scores": 0, "fresh_validation_rows": 0}, "decision": "Proceed to manuscript and do not add new docking or quantum hardware work for this rescue route." if not gate_passes else "Prepare an independently preregistered prospective target validation before claiming generalization.", "input_hashes": {target: {"matrices": [sha256(root / path) for path in ([spec["matrix"]] if "matrix" in spec else spec["matrix_parts"])], "manifests": [sha256(root / path) for path in ([spec["manifest"]] if "manifest" in spec else spec["manifest_parts"])]} for target, spec in target_specs.items()}}
    output_cfg = config["outputs"]
    write_csv(root / output_cfg["fold_csv"], fold_rows)
    write_csv(root / output_cfg["target_csv"], target_rows)
    write_csv(root / output_cfg["pair_csv"], pair_rows)
    result_path = root / output_cfg["result_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    audit = {"schema_version": "1.0", "status": "stage98_audit_ok", "fold_rows": len(fold_rows), "target_rows": len(target_rows), "pair_rows": len(pair_rows), "labels_used_by_selector": False, "new_docking_jobs": 0, "quantum_hardware_jobs": 0}
    audit_path = root / output_cfg["audit_json"]
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report = ["# Stage98 跨蛋白受体互补性分析", "", "本分析只使用已经完成的真实 Uni-Dock 矩阵，不新增对接、不使用量子硬件。complementarity 选择器只读取对接分数和受体间秩相关性；active/decoy 或 high/low 标签只在最终 BEDROC 评价时使用。", "", "## 预注册门槛", "", f"- 蛋白数量：{len(target_data)}。", f"- BEDROC：alpha={alpha:g}。", f"- 通过条件：至少 {config['go_gate']['minimum_targets_with_positive_k3_gain']} 个蛋白的 k=3 相对 k=1 提升 >= {config['go_gate']['minimum_mean_k3_gain']:.2f}，平均提升 >= {config['go_gate']['minimum_mean_k3_gain']:.2f}，最差蛋白 >= {config['go_gate']['minimum_worst_target_k3_gain']:.2f}。", "", "## 结果", "", f"- k=3 平均提升：`{mean_gain:.6f}`。", f"- 最差蛋白提升：`{worst_gain:.6f}`。", f"- 达到 +0.02 的蛋白数：`{positive_count}/{len(target_data)}`。", f"- Go/No-Go：`{'GO' if gate_passes else 'NO-GO'}`。", "", "| Target | k=1 complementarity BEDROC | k=3 complementarity BEDROC | Gain | >=0.02 |", "|---|---:|---:|---:|---|"]
    report.extend(f"| {row['target_id']} | {row['complementarity_k1']:.6f} | {row['complementarity_k3']:.6f} | {row['k3_gain']:.6f} | {row['positive_gain']} |" for row in target_gate_rows)
    report.extend(["", "## 解释", "", "如果 Go/No-Go 为 NO-GO，则不能再通过增加蛋白、调 diversity weight 或新增 QUBO 形式来追逐同一结论；应将论文定位为受体组合效用、QUBO/CQM 表达和量子硬件边界研究。监督 oracle_train 只用于估计标签可提供的上限，不是可部署方法。", ""])
    report_path = root / output_cfg["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": result["status"], "gate_passes": gate_passes, "mean_k3_gain": mean_gain, "worst_k3_gain": worst_gain, "positive_target_count": positive_count, "fold_rows": len(fold_rows), "target_rows": len(target_rows), "pair_rows": len(pair_rows)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
