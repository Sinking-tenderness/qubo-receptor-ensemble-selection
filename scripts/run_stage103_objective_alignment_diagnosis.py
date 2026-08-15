"""Diagnose objective-versus-ranking alignment for the frozen Stage99 QUBO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_stage102a_phase_a_results as stage102a
from scripts import run_stage99_qubo_objective_repair_screen as stage99


@dataclass(frozen=True)
class TargetData:
    target_id: str
    receptors: list[str]
    values: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    outer_folds: np.ndarray | None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_parent_hashes(root: Path, config: dict[str, Any]) -> None:
    for key, expected in config["parent"].items():
        if not key.endswith("_sha256"):
            continue
        path_key = key.removesuffix("_sha256")
        actual = sha256(root / config["parent"][path_key])
        if actual != expected:
            raise ValueError(f"parent hash mismatch for {path_key}: {actual} != {expected}")


def load_targets(root: Path, config: dict[str, Any]) -> dict[str, TargetData]:
    stage99_config = read_json(root / config["parent"]["stage99_config"])
    loaded: dict[str, TargetData] = {}
    for target_id, spec in stage99_config["targets"].items():
        receptors, rows, values = stage99.load_target(root, spec)
        loaded[target_id] = TargetData(
            target_id=target_id,
            receptors=receptors,
            values=values,
            labels=stage99.label_array(rows),
            groups=np.asarray(
                [row.get(spec["group_field"], row.get("split_group_id", row["ligand_id"])) for row in rows]
            ),
            outer_folds=None,
        )
    received_root = root / config["targets"]["stage102a_received_root"]
    for target_id in config["targets"]["stage102a_targets"]:
        receptors, rows, values = stage102a.load_target(received_root, target_id)
        loaded[target_id] = TargetData(
            target_id=target_id,
            receptors=receptors,
            values=values,
            labels=stage99.label_array(rows),
            groups=np.asarray([row["split_group_id"] for row in rows]),
            outer_folds=np.asarray([int(row["outer_fold"]) for row in rows]),
        )
    return loaded


def outer_splits(data: TargetData) -> list[tuple[int, np.ndarray, np.ndarray]]:
    if data.outer_folds is not None:
        return [
            (fold, np.flatnonzero(data.outer_folds != fold), np.flatnonzero(data.outer_folds == fold))
            for fold in sorted(set(data.outer_folds.tolist()))
        ]
    splitter = GroupKFold(n_splits=5)
    return [
        (fold, train, test)
        for fold, (train, test) in enumerate(splitter.split(data.values, groups=data.groups), start=1)
    ]


def correlation(left: list[float], right: list[float]) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if math.isfinite(float(value)) else 0.0


def score_subset(
    ranks: np.ndarray,
    labels: np.ndarray,
    subset: tuple[int, ...],
    alpha: float,
    aggregation: str,
) -> float:
    return stage99.bedroc(stage99.aggregate(ranks[:, list(subset)], aggregation), labels, alpha)


def bedroc_batch(scores: np.ndarray, labels: np.ndarray, alpha: float) -> np.ndarray:
    """Compute exact Stage99 BEDROC for each score column without sampling."""
    total = len(labels)
    active_count = int(np.sum(labels == 1))
    if active_count == 0 or active_count == total:
        raise ValueError("BEDROC requires both active and decoy labels")
    ordered_labels = labels[np.argsort(scores, axis=0, kind="stable")]
    weights = np.exp(-alpha * np.arange(1, total + 1, dtype=float) / total)
    expected = active_count * float(np.sum(weights)) / total
    observed = np.sum(weights[:, None] * (ordered_labels == 1), axis=0) / expected
    best = float(np.sum(weights[:active_count]) / expected)
    worst = float(np.sum(weights[-active_count:]) / expected)
    return (observed - worst) / (best - worst)


def enumerate_subset_metrics(
    candidates: np.ndarray,
    node: np.ndarray,
    pair: np.ndarray,
    train_ranks: np.ndarray,
    train_labels: np.ndarray,
    test_ranks: np.ndarray,
    test_labels: np.ndarray,
    alpha: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exactly enumerate QUBO and BEDROC values in memory-bounded batches."""
    count, k = candidates.shape
    objective = np.empty(count, dtype=float)
    train_primary = np.empty(count, dtype=float)
    outer_primary = np.empty(count, dtype=float)
    outer_secondary = np.empty(count, dtype=float)
    pair_positions = list(itertools.combinations(range(k), 2))
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = candidates[start:stop]
        values = np.mean(node[batch], axis=1)
        if pair_positions:
            values += np.mean(
                [pair[batch[:, left], batch[:, right]] for left, right in pair_positions], axis=0
            )
        objective[start:stop] = values
        train_selected = train_ranks[:, batch]
        test_selected = test_ranks[:, batch]
        train_primary[start:stop] = bedroc_batch(np.min(train_selected, axis=2), train_labels, alpha)
        outer_primary[start:stop] = bedroc_batch(np.min(test_selected, axis=2), test_labels, alpha)
        if k == 1:
            outer_secondary[start:stop] = outer_primary[start:stop]
        else:
            outer_secondary[start:stop] = bedroc_batch(
                np.mean(np.partition(test_selected, kth=1, axis=2)[:, :, :2], axis=2),
                test_labels,
                alpha,
            )
    return objective, train_primary, outer_primary, outer_secondary


def enumerate_fold(
    data: TargetData,
    fold: int,
    train_index: np.ndarray,
    test_index: np.ndarray,
    stage99_config: dict[str, Any],
    candidate_k: list[int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    alpha = float(stage99_config["objective"]["bedroc_alpha"])
    train_values = data.values[train_index]
    train_labels = data.labels[train_index]
    test_labels = data.labels[test_index]
    train_ranks = stage99.percentile_ranks(train_values, train_values)
    test_ranks = stage99.percentile_ranks(train_values, data.values[test_index])
    node, pair = stage99.coefficients(train_ranks, train_labels, stage99_config, "repair")
    fold_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for k in candidate_k:
        candidates = np.asarray(list(itertools.combinations(range(len(data.receptors)), k)), dtype=int)
        objective, train_primary, outer_primary, outer_secondary = enumerate_subset_metrics(
            candidates,
            node,
            pair,
            train_ranks,
            train_labels,
            test_ranks,
            test_labels,
            alpha,
            batch_size,
        )
        exact_index = int(np.argmax(objective))
        outer_best_index = int(np.argmax(outer_primary))
        train_best_index = int(np.argmax(train_primary))
        fold_rows.append(
            {
                "target_id": data.target_id,
                "outer_fold": fold,
                "k": k,
                "receptor_count": len(data.receptors),
                "subset_count": len(candidates),
                "qubo_vs_train_primary_spearman": correlation(objective.tolist(), train_primary.tolist()),
                "qubo_vs_outer_primary_spearman": correlation(objective.tolist(), outer_primary.tolist()),
                "train_primary_vs_outer_primary_spearman": correlation(train_primary, outer_primary),
                "train_qubo_optimum_objective": objective[exact_index],
                "train_qubo_optimum_train_primary_bedroc20": train_primary[exact_index],
                "train_qubo_optimum_outer_primary_bedroc20": outer_primary[exact_index],
                "train_qubo_optimum_outer_secondary_bedroc20": outer_secondary[exact_index],
                "outer_primary_oracle_bedroc20": outer_primary[outer_best_index],
                "outer_primary_oracle_subset": stage99.subset_name(tuple(candidates[outer_best_index]), data.receptors),
                "outer_primary_regret_of_train_qubo_optimum": outer_primary[outer_best_index] - outer_primary[exact_index],
                "outer_primary_regret_of_train_bedroc_optimum": outer_primary[outer_best_index] - outer_primary[train_best_index],
                "train_qubo_optimum_subset": stage99.subset_name(tuple(candidates[exact_index]), data.receptors),
                "train_primary_optimum_subset": stage99.subset_name(tuple(candidates[train_best_index]), data.receptors),
                "train_qubo_matches_train_primary_optimum": bool(np.array_equal(candidates[exact_index], candidates[train_best_index])),
                "train_qubo_matches_outer_primary_oracle": bool(np.array_equal(candidates[exact_index], candidates[outer_best_index])),
                "uses_outer_labels_for_selection": False,
            }
        )
        selected_rows.append(
            {
                "target_id": data.target_id,
                "outer_fold": fold,
                "k": k,
                "selection": "train_qubo_optimum",
                "subset": stage99.subset_name(tuple(candidates[exact_index]), data.receptors),
                "qubo_objective": objective[exact_index],
                "train_primary_bedroc20": train_primary[exact_index],
                "outer_primary_bedroc20": outer_primary[exact_index],
                "outer_secondary_bedroc20": outer_secondary[exact_index],
                "uses_outer_labels_for_selection": False,
            }
        )
    k1 = next(row for row in fold_rows if row["k"] == 1)
    k2 = next(row for row in fold_rows if row["k"] == 2)
    for row in fold_rows:
        row["outer_oracle_k2_minus_k1"] = k2["outer_primary_oracle_bedroc20"] - k1["outer_primary_oracle_bedroc20"]
        row["train_qubo_k2_minus_k1"] = k2["train_qubo_optimum_outer_primary_bedroc20"] - k1["train_qubo_optimum_outer_primary_bedroc20"]
    return fold_rows, selected_rows


def summarize(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target_id in sorted({str(row["target_id"]) for row in fold_rows}):
        for k in (1, 2, 3):
            rows = [row for row in fold_rows if row["target_id"] == target_id and int(row["k"]) == k]
            result.append(
                {
                    "target_id": target_id,
                    "k": k,
                    "fold_count": len(rows),
                    "mean_qubo_vs_train_primary_spearman": float(np.mean([row["qubo_vs_train_primary_spearman"] for row in rows])),
                    "mean_qubo_vs_outer_primary_spearman": float(np.mean([row["qubo_vs_outer_primary_spearman"] for row in rows])),
                    "mean_train_primary_vs_outer_primary_spearman": float(np.mean([row["train_primary_vs_outer_primary_spearman"] for row in rows])),
                    "mean_outer_primary_regret_of_train_qubo_optimum": float(np.mean([row["outer_primary_regret_of_train_qubo_optimum"] for row in rows])),
                    "mean_selection_metric_regret_secondary_minus_primary": float(np.mean([row["train_qubo_optimum_outer_secondary_bedroc20"] - row["train_qubo_optimum_outer_primary_bedroc20"] for row in rows])),
                    "train_qubo_matches_train_primary_optimum_fold_count": int(sum(row["train_qubo_matches_train_primary_optimum"] for row in rows)),
                    "train_qubo_matches_outer_primary_oracle_fold_count": int(sum(row["train_qubo_matches_outer_primary_oracle"] for row in rows)),
                    "mean_outer_oracle_k2_minus_k1": float(np.mean([row["outer_oracle_k2_minus_k1"] for row in rows])),
                    "mean_train_qubo_selected_k2_minus_k1": float(np.mean([row["train_qubo_k2_minus_k1"] for row in rows])),
                }
            )
    return result


def render_report(result: dict[str, Any], summaries: list[dict[str, Any]]) -> str:
    rows = [row for row in summaries if int(row["k"]) == 2]
    lines = [
        "# Stage103 objective-alignment diagnosis",
        "",
        "This is a post-hoc mechanism diagnosis of the frozen Stage99 objective. It does not tune or nominate a replacement objective, initiate docking, unlock PARP1, or authorize quantum hardware.",
        "",
        "## k=2 alignment summary",
        "",
        "| Target | QUBO vs train BEDROC Spearman | QUBO vs outer BEDROC Spearman | Outer regret of QUBO subset | Oracle k=2 minus k=1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_id']} | {row['mean_qubo_vs_train_primary_spearman']:+.3f} | "
            f"{row['mean_qubo_vs_outer_primary_spearman']:+.3f} | "
            f"{row['mean_outer_primary_regret_of_train_qubo_optimum']:+.4f} | "
            f"{row['mean_outer_oracle_k2_minus_k1']:+.4f} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        result["decision"]["next_action"],
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage103_objective_alignment_diagnosis.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = read_json(root / args.config)
    validate_parent_hashes(root, config)
    stage99_config = read_json(root / config["parent"]["stage99_config"])
    targets = load_targets(root, config)
    candidate_k = [int(value) for value in config["objective_under_diagnosis"]["candidate_k_values"]]
    batch_size = int(config["diagnostics"]["enumeration_batch_size"])
    fold_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for data in targets.values():
        for fold, train_index, test_index in outer_splits(data):
            rows, selected = enumerate_fold(
                data, fold, train_index, test_index, stage99_config, candidate_k, batch_size
            )
            fold_rows.extend(rows)
            selected_rows.extend(selected)
    summaries = summarize(fold_rows)
    k2_outer = [row["mean_qubo_vs_outer_primary_spearman"] for row in summaries if row["k"] == 2]
    k2_train = [row["mean_qubo_vs_train_primary_spearman"] for row in summaries if row["k"] == 2]
    thresholds = config["diagnostics"]["correlation_interpretation_thresholds"]
    median_train = float(np.median(k2_train))
    median_outer = float(np.median(k2_outer))
    objective_fails_outer_alignment = median_outer < float(thresholds["outer_alignment_failure_if_median_spearman_below"])
    result = {
        "schema_version": "1.0",
        "status": "stage103_objective_alignment_diagnosis_complete",
        "target_ids": sorted(targets),
        "summary": {
            "k2_target_median_qubo_vs_train_primary_spearman": median_train,
            "k2_target_median_qubo_vs_outer_primary_spearman": median_outer,
            "k2_train_alignment_supported": median_train >= float(thresholds["train_alignment_supported_if_median_spearman_at_least"]),
            "k2_outer_alignment_supported": median_outer >= float(thresholds["outer_alignment_supported_if_median_spearman_at_least"]),
            "k2_outer_alignment_failure_supported": objective_fails_outer_alignment,
        },
        "decision": {
            "replacement_objective_authorized": False,
            "parp1_released": False,
            "quantum_hardware_authorized": False,
            "next_action": (
                "If train alignment is high but outer alignment is low, the dominant limitation is transfer/generalization of pair complementarity; "
                "do not tune a new objective on these same targets. If train alignment is also low, formulate one separately reviewed surrogate-alignment objective before any untouched-target study."
            ),
        },
        "data_boundary": config["data_boundary"],
        "interpretation": "All outer metrics are diagnostic only. They are not used to select any subset or tune any coefficient.",
    }
    outputs = config["outputs"]
    write_csv(root / outputs["fold_csv"], fold_rows)
    write_csv(root / outputs["subset_csv"], selected_rows)
    write_csv(root / outputs["target_csv"], summaries)
    result_path = root / outputs["result_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report_path = root / outputs["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result, summaries), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
