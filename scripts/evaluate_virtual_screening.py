"""Evaluate virtual-screening ranking metrics from a docking score table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


REQUIRED_COLUMNS = {"ligand_id", "label", "pose_rank", "docking_score", "status"}


def validate_columns(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("input score table has no header")
    missing = REQUIRED_COLUMNS.difference(fieldnames)
    if missing:
        raise ValueError(f"input score table is missing required columns: {sorted(missing)}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames)
        return list(reader)


def select_best_pose(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    best_by_ligand: dict[str, dict[str, object]] = {}
    failed_ligands: set[str] = set()

    for row in rows:
        ligand_id = row["ligand_id"]
        if row["status"] != "ok":
            failed_ligands.add(ligand_id)
            continue
        if row["pose_rank"] == "":
            continue
        pose_rank = int(float(row["pose_rank"]))
        if pose_rank != 1:
            continue
        docking_score = float(row["docking_score"])
        label = row["label"]
        if label not in {"active", "decoy", "inactive"}:
            raise ValueError(f"unsupported label for {ligand_id}: {label}")
        best_by_ligand[ligand_id] = {
            **row,
            "ligand_id": ligand_id,
            "binary_label": 1 if label == "active" else 0,
            "pose_rank": pose_rank,
            "docking_score": docking_score,
            "ranking_score": -docking_score,
        }

    ranked = sorted(
        best_by_ligand.values(),
        key=lambda row: (-float(row["ranking_score"]), str(row["ligand_id"])),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    if failed_ligands:
        print(f"warning: failed ligands excluded from ranking: {sorted(failed_ligands)}")
    return ranked


def roc_auc_pairwise(binary_labels: list[int], ranking_scores: list[float]) -> float:
    positives = [(score, index) for index, (label, score) in enumerate(zip(binary_labels, ranking_scores)) if label == 1]
    negatives = [(score, index) for index, (label, score) in enumerate(zip(binary_labels, ranking_scores)) if label == 0]
    if not positives or not negatives:
        return math.nan

    wins = 0.0
    total = len(positives) * len(negatives)
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    return wins / total


def average_precision(ranked: list[dict[str, object]]) -> float:
    active_total = sum(int(row["binary_label"]) for row in ranked)
    if active_total == 0:
        return math.nan

    precision_sum = 0.0
    active_seen = 0
    for index, row in enumerate(ranked, start=1):
        if int(row["binary_label"]) == 1:
            active_seen += 1
            precision_sum += active_seen / index
    return precision_sum / active_total


def bedroc(ranked: list[dict[str, object]], alpha: float) -> float:
    total = len(ranked)
    active_ranks = [index for index, row in enumerate(ranked, start=1) if int(row["binary_label"]) == 1]
    active_total = len(active_ranks)
    if total == 0 or active_total == 0 or active_total == total:
        return math.nan

    def exponential_sum(ranks: list[int]) -> float:
        return sum(math.exp(-alpha * rank / total) for rank in ranks)

    all_rank_weights = [math.exp(-alpha * rank / total) for rank in range(1, total + 1)]
    random_expected_sum = active_total * (sum(all_rank_weights) / total)
    observed_rie = exponential_sum(active_ranks) / random_expected_sum

    best_ranks = list(range(1, active_total + 1))
    worst_ranks = list(range(total - active_total + 1, total + 1))
    max_rie = exponential_sum(best_ranks) / random_expected_sum
    min_rie = exponential_sum(worst_ranks) / random_expected_sum
    if max_rie == min_rie:
        return math.nan
    return (observed_rie - min_rie) / (max_rie - min_rie)


def enrichment_factor(ranked: list[dict[str, object]], fraction: float) -> dict[str, float | int]:
    total = len(ranked)
    active_total = sum(int(row["binary_label"]) for row in ranked)
    if total == 0 or active_total == 0:
        return {
            "top_n": 0,
            "top_active": 0,
            "top_active_fraction": math.nan,
            "overall_active_fraction": math.nan,
            "ef": math.nan,
        }

    top_n = max(1, math.ceil(total * fraction))
    top_rows = ranked[:top_n]
    top_active = sum(int(row["binary_label"]) for row in top_rows)
    top_active_fraction = top_active / top_n
    overall_active_fraction = active_total / total
    return {
        "top_n": top_n,
        "top_active": top_active,
        "top_active_fraction": top_active_fraction,
        "overall_active_fraction": overall_active_fraction,
        "ef": top_active_fraction / overall_active_fraction,
    }


def scalar_metrics(
    ranked: list[dict[str, object]],
    top_fractions: list[float],
    bedroc_alpha: float,
) -> dict[str, float]:
    labels = [int(row["binary_label"]) for row in ranked]
    scores = [float(row["ranking_score"]) for row in ranked]
    values = {
        "roc_auc_pairwise": roc_auc_pairwise(labels, scores),
        "pr_auc_average_precision": average_precision(ranked),
        f"bedroc_alpha_{bedroc_alpha:g}": bedroc(ranked, bedroc_alpha),
    }
    for fraction in top_fractions:
        values[f"EF{fraction * 100:g}%"] = float(enrichment_factor(ranked, fraction)["ef"])
    return values


def percentile(values: list[float], q: float) -> float:
    finite_values = sorted(value for value in values if not math.isnan(value))
    if not finite_values:
        return math.nan
    if len(finite_values) == 1:
        return finite_values[0]
    position = (len(finite_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite_values[int(position)]
    weight = position - lower
    return finite_values[lower] * (1 - weight) + finite_values[upper] * weight


def bootstrap_confidence_intervals(
    ranked: list[dict[str, object]],
    top_fractions: list[float],
    bedroc_alpha: float,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    if iterations <= 0:
        return {}

    rng = random.Random(seed)
    metric_samples: dict[str, list[float]] = {}
    skipped = 0
    for _ in range(iterations):
        sample = [rng.choice(ranked).copy() for _ in ranked]
        sample = sorted(
            sample,
            key=lambda row: (-float(row["ranking_score"]), str(row["ligand_id"])),
        )
        labels = {int(row["binary_label"]) for row in sample}
        if labels != {0, 1}:
            skipped += 1
            continue
        values = scalar_metrics(sample, top_fractions, bedroc_alpha)
        for key, value in values.items():
            metric_samples.setdefault(key, []).append(value)

    intervals: dict[str, dict[str, float | int]] = {}
    for key, values in metric_samples.items():
        intervals[key] = {
            "mean": sum(values) / len(values) if values else math.nan,
            "ci95_low": percentile(values, 0.025),
            "ci95_high": percentile(values, 0.975),
            "n_bootstrap_used": len(values),
            "n_bootstrap_skipped": skipped,
        }
    return intervals


def build_metrics(
    ranked: list[dict[str, object]],
    top_fractions: list[float],
    bedroc_alpha: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    labels = [int(row["binary_label"]) for row in ranked]
    scores = [float(row["ranking_score"]) for row in ranked]
    label_counts = {
        "active": sum(labels),
        "non_active": len(labels) - sum(labels),
    }
    metrics: dict[str, object] = {
        "ligand_count": len(ranked),
        "label_counts": label_counts,
        "roc_auc_pairwise": roc_auc_pairwise(labels, scores),
        "pr_auc_average_precision": average_precision(ranked),
        "bedroc": {
            "alpha": bedroc_alpha,
            "value": bedroc(ranked, bedroc_alpha),
            "normalization": "finite-rank normalized RIE; 0 is worst active placement and 1 is best",
        },
        "score_direction": "ranking_score = -docking_score; higher ranking_score is better",
        "enrichment": {},
    }
    for fraction in top_fractions:
        key = f"EF{fraction * 100:g}%"
        metrics["enrichment"][key] = enrichment_factor(ranked, fraction)
    metrics["bootstrap_ci95"] = bootstrap_confidence_intervals(
        ranked=ranked,
        top_fractions=top_fractions,
        bedroc_alpha=bedroc_alpha,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    return metrics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-table", type=Path, required=True)
    parser.add_argument("--ranking-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument(
        "--top-fractions",
        nargs="+",
        type=float,
        default=[0.01, 0.05],
        help="Fractions for enrichment factor, e.g. 0.01 0.05.",
    )
    parser.add_argument("--bedroc-alpha", type=float, default=20.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_rows(args.score_table)
    ranked = select_best_pose(rows)
    metrics = build_metrics(
        ranked=ranked,
        top_fractions=args.top_fractions,
        bedroc_alpha=args.bedroc_alpha,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_csv(args.ranking_output, ranked)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"ranking_output={args.ranking_output}")
    print(f"metrics_output={args.metrics_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
