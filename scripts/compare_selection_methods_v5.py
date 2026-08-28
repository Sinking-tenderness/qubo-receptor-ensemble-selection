"""Compare receptor-set selection methods under the existing V5 outer folds.

The primary comparison fixes the cardinality per outer fold to the k selected
by the frozen adaptive controller. QUBO, linear top-k, and direct BEDROC
greedy therefore compete at the same set size. The single-receptor baseline
always selects one receptor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qubo_receptor_ensemble.screening import ranked_metrics_with_ids
from qubo_receptor_ensemble.solvers import build_problem, solve_problem


METRIC_KEYS = (
    "bedroc_alpha_20",
    "roc_auc",
    "pr_auc_average_precision",
    "EF1%",
    "EF5%",
    "EF10%",
)
METHODS = ("qubo", "linear", "greedy", "single")


def load_problem(path: Path) -> tuple[str, list[dict[str, object]], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    problem = payload.get("problem")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"problem has no rows: {path}")
    if not isinstance(problem, dict):
        raise ValueError(f"problem metadata is missing: {path}")
    receptor_ids = [str(value) for value in problem.get("receptor_ids", [])]
    if not receptor_ids:
        raise ValueError(f"problem has no receptor IDs: {path}")
    normalized: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"problem row is not an object: {path}")
        normalized_row = dict(row)
        normalized_row["ligand_id"] = str(row["ligand_id"])
        normalized_row["label"] = str(row["label"])
        normalized_row["outer_fold"] = int(float(row["outer_fold"]))
        for receptor_id in receptor_ids:
            normalized_row[receptor_id] = float(row[receptor_id])
        normalized.append(normalized_row)
    target_ids = {str(row.get("target_id", "")) for row in normalized}
    if len(target_ids) != 1 or "" in target_ids:
        raise ValueError(f"problem rows have inconsistent target IDs: {path}")
    if {str(row["label"]) for row in normalized} != {"active", "decoy"}:
        raise ValueError(f"problem rows must contain active and decoy labels: {path}")
    return target_ids.pop(), normalized, receptor_ids


def load_adaptive_ks(path: Path) -> dict[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"V5 decision log has no folds: {path}")
    result: dict[int, int] = {}
    for entry in folds:
        if not isinstance(entry, dict):
            raise ValueError(f"invalid V5 fold entry: {path}")
        fold = int(entry["outer_fold"])
        selected_k = int(entry["adaptive_selected_k"])
        if fold in result:
            raise ValueError(f"duplicate outer fold {fold}: {path}")
        result[fold] = selected_k
    return result


def subset_metrics(
    rows: list[dict[str, object]], subset: tuple[str, ...]
) -> dict[str, object]:
    if not subset:
        raise ValueError("subset must not be empty")
    data = {
        str(row["ligand_id"]): {
            "label": str(row["label"]),
            "score": sum(float(row[receptor_id]) for receptor_id in subset)
            / len(subset),
        }
        for row in rows
    }
    return ranked_metrics_with_ids(data, score_key="score", bedroc_alpha=20.0)


def _metric_value(rows: list[dict[str, object]], subset: tuple[str, ...]) -> float:
    return float(subset_metrics(rows, subset)["bedroc_alpha_20"])


def select_single(
    train_rows: list[dict[str, object]], receptor_ids: list[str]
) -> tuple[str, ...]:
    candidates = [
        ((receptor_id,), _metric_value(train_rows, (receptor_id,)))
        for receptor_id in receptor_ids
    ]
    return max(candidates, key=lambda item: (item[1], tuple(item[0])))[0]


def select_linear(
    train_rows: list[dict[str, object]],
    receptor_ids: list[str],
    k: int,
) -> tuple[str, ...]:
    if not 1 <= k <= len(receptor_ids):
        raise ValueError(f"invalid k={k} for {len(receptor_ids)} receptors")
    ranked = sorted(
        (
            (
                receptor_id,
                _metric_value(train_rows, (receptor_id,)),
            )
            for receptor_id in receptor_ids
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(sorted(receptor_id for receptor_id, _ in ranked[:k]))


def select_greedy(
    train_rows: list[dict[str, object]],
    receptor_ids: list[str],
    k: int,
) -> tuple[str, ...]:
    if not 1 <= k <= len(receptor_ids):
        raise ValueError(f"invalid k={k} for {len(receptor_ids)} receptors")
    selected: tuple[str, ...] = ()
    while len(selected) < k:
        candidates: list[tuple[tuple[str, ...], float]] = []
        for receptor_id in receptor_ids:
            if receptor_id in selected:
                continue
            candidate = tuple(sorted((*selected, receptor_id)))
            candidates.append((candidate, _metric_value(train_rows, candidate)))
        selected = max(candidates, key=lambda item: (item[1], item[0]))[0]
    return selected


def select_qubo(
    train_rows: list[dict[str, object]],
    receptor_ids: list[str],
    k: int,
    redundancy_weight: float = 0.25,
) -> tuple[str, ...]:
    config: dict[str, object] = {
        "type": "receptor_subset",
        "strategy": "qubo",
        "target_size": k,
        "utility_metric": "bedroc",
        "bedroc_alpha": 20.0,
        "utility_normalization": "none",
        "weights": {
            "redundancy": redundancy_weight,
            "count": 0.1,
            "size": 10.0,
        },
        "receptor_ids": list(receptor_ids),
    }
    problem = build_problem(train_rows, config)
    return tuple(solve_problem(problem, "exact").subset)


def select_methods(
    train_rows: list[dict[str, object]],
    receptor_ids: list[str],
    k: int,
    redundancy_weight: float = 0.25,
) -> dict[str, tuple[str, ...]]:
    return {
        "qubo": select_qubo(train_rows, receptor_ids, k, redundancy_weight),
        "linear": select_linear(train_rows, receptor_ids, k),
        "greedy": select_greedy(train_rows, receptor_ids, k),
        "single": select_single(train_rows, receptor_ids),
    }


def _jaccard(first: tuple[str, ...], second: tuple[str, ...]) -> float:
    left, right = set(first), set(second)
    return len(left & right) / len(left | right)


def _mean_pairwise_jaccard(subsets: list[tuple[str, ...]]) -> float:
    pairs = list(itertools.combinations(subsets, 2))
    return fmean(_jaccard(first, second) for first, second in pairs) if pairs else 1.0


def _compact_metrics(metrics: dict[str, object]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def evaluate_target(
    problem_path: Path,
    decision_log_path: Path,
    redundancy_weight: float = 0.25,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    target_id, rows, receptor_ids = load_problem(problem_path)
    adaptive_ks = load_adaptive_ks(decision_log_path)
    folds = sorted({int(row["outer_fold"]) for row in rows})
    if set(folds) != set(adaptive_ks):
        raise ValueError(f"decision log folds differ from problem rows: {target_id}")
    records: list[dict[str, object]] = []
    selected_by_method: dict[str, list[tuple[str, ...]]] = {method: [] for method in METHODS}
    for fold in folds:
        train_rows = [row for row in rows if int(row["outer_fold"]) != fold]
        test_rows = [row for row in rows if int(row["outer_fold"]) == fold]
        k = adaptive_ks[fold]
        selected = select_methods(train_rows, receptor_ids, k, redundancy_weight)
        for method in METHODS:
            subset = selected[method]
            metrics = _compact_metrics(subset_metrics(test_rows, subset))
            selected_by_method[method].append(subset)
            records.append(
                {
                    "target_id": target_id,
                    "outer_fold": fold,
                    "method": method,
                    "selected_k": len(subset),
                    "adaptive_k": k,
                    "selected_receptor_ids": "+".join(subset),
                    **{f"test_{key}": value for key, value in metrics.items()},
                }
            )
    return records, {
        "target_id": target_id,
        "problem_path": str(problem_path),
        "problem_sha256": hashlib.sha256(problem_path.read_bytes()).hexdigest(),
        "decision_log_path": str(decision_log_path),
        "outer_fold_count": len(folds),
        "ligand_count": len(rows),
        "active_count": sum(row["label"] == "active" for row in rows),
        "receptor_count": len(receptor_ids),
        "adaptive_ks": {str(fold): adaptive_ks[fold] for fold in folds},
        "selection_jaccard_mean": {
            method: _mean_pairwise_jaccard(selected_by_method[method])
            for method in METHODS
        },
    }


def summarize_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault((str(record["target_id"]), str(record["method"])), []).append(record)
    output: list[dict[str, object]] = []
    for (target_id, method), entries in sorted(grouped.items()):
        output.append(
            {
                "target_id": target_id,
                "method": method,
                "outer_fold_count": len(entries),
                "mean_selected_k": fmean(float(entry["selected_k"]) for entry in entries),
                **{
                    f"mean_test_{key}": fmean(float(entry[f"test_{key}"]) for entry in entries)
                    for key in METRIC_KEYS
                },
            }
        )
    return output


def paired_comparisons(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_target_fold_method = {
        (str(record["target_id"]), int(record["outer_fold"]), str(record["method"])): record
        for record in records
    }
    output: list[dict[str, object]] = []
    for target_id in sorted({str(record["target_id"]) for record in records}):
        folds = sorted(
            {int(record["outer_fold"]) for record in records if str(record["target_id"]) == target_id}
        )
        for comparator in ("single", "linear", "greedy"):
            for metric in METRIC_KEYS:
                deltas = [
                    float(by_target_fold_method[(target_id, fold, "qubo")][f"test_{metric}"])
                    - float(by_target_fold_method[(target_id, fold, comparator)][f"test_{metric}"])
                    for fold in folds
                ]
                output.append(
                    {
                        "target_id": target_id,
                        "comparison": f"qubo_minus_{comparator}",
                        "metric": metric,
                        "mean_delta": fmean(deltas),
                        "worst_fold_delta": min(deltas),
                        "best_fold_delta": max(deltas),
                        "positive_fold_fraction": sum(delta > 0 for delta in deltas) / len(deltas),
                        "fold_deltas": "|".join(f"{delta:.6f}" for delta in deltas),
                    }
                )
    return output


def macro_summary(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in summarize_records(records):
        grouped.setdefault(str(row["method"]), []).append(row)
    output = []
    for method, entries in sorted(grouped.items()):
        output.append(
            {
                "method": method,
                "target_count": len(entries),
                **{
                    key: fmean(float(entry[key]) for entry in entries)
                    for key in entries[0]
                    if key.startswith("mean_test_")
                },
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--redundancy-weight", type=float, default=0.25)
    parser.add_argument(
        "--target",
        action="append",
        nargs=3,
        metavar=("TARGET", "PROBLEM_JSON", "DECISION_LOG_JSON"),
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_records: list[dict[str, object]] = []
    target_metadata: list[dict[str, object]] = []
    for target_id, problem_path, decision_log_path in args.target:
        records, metadata = evaluate_target(
            Path(problem_path), Path(decision_log_path), args.redundancy_weight
        )
        if str(metadata["target_id"]).upper() != str(target_id).upper():
            raise ValueError(
                f"target label {target_id} does not match rows {metadata['target_id']}"
            )
        all_records.extend(records)
        target_metadata.append(metadata)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "folds_long.csv", all_records)
    target_summary = summarize_records(all_records)
    _write_csv(out / "summary_by_target.csv", target_summary)
    _write_csv(out / "paired_comparisons.csv", paired_comparisons(all_records))
    _write_csv(out / "macro_summary.csv", macro_summary(all_records))
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "primary_comparison": "QUBO, linear, and direct BEDROC greedy use the same outer-fold adaptive k",
                    "single_baseline": "best singleton receptor selected on outer-training rows",
                    "selection_data": "outer-training rows only",
                    "evaluation_data": "the same held-out outer fold for every method",
                    "aggregation": "mean_score",
                    "primary_metric": "BEDROC20",
                    "redundancy_weight": args.redundancy_weight,
                },
                "targets": target_metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"targets={len(target_metadata)} folds={len(all_records)} output={out}")
    for row in macro_summary(all_records):
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
