"""Compare train-only Uni-Dock GPU scores with fixed CPU Vina e32 evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from .run_unidock_gpu_equivalence import (
        file_sha256,
        output_descriptor,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        validate_inputs,
        write_csv,
        write_json,
    )
except ImportError:
    from run_unidock_gpu_equivalence import (
        file_sha256,
        output_descriptor,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        validate_inputs,
        write_csv,
        write_json,
    )


def quantile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile input is invalid")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for offset in range(cursor, end):
            ranks[order[offset]] = rank
        cursor = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs differ or are too short")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 1.0 if left == right else 0.0
    return numerator / (left_scale * right_scale)


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def top_overlap(
    ligand_ids: list[str],
    cpu_scores: list[float],
    gpu_scores: list[float],
    fraction: float,
) -> tuple[float, int]:
    count = max(1, math.ceil(len(ligand_ids) * fraction))
    cpu_order = sorted(
        zip(cpu_scores, ligand_ids), key=lambda item: (item[0], item[1])
    )
    gpu_order = sorted(
        zip(gpu_scores, ligand_ids), key=lambda item: (item[0], item[1])
    )
    cpu_top = {ligand_id for _, ligand_id in cpu_order[:count]}
    gpu_top = {ligand_id for _, ligand_id in gpu_order[:count]}
    return len(cpu_top & gpu_top) / count, count


def cpu_scores(
    root: Path,
    config: dict[str, object],
    receptor_ids: set[str],
) -> tuple[dict[tuple[str, str, str], float], list[dict[str, object]]]:
    scores: dict[tuple[str, str, str], float] = {}
    timing: list[dict[str, object]] = []
    for seed in config["inputs"]["cpu_seed_runs"]:
        seed_id = str(seed["seed_id"])
        rows = read_csv(rooted_path(root, str(seed["scores_path"])))
        for row in rows:
            if row["receptor_id"] not in receptor_ids:
                continue
            key = (seed_id, row["receptor_id"], row["ligand_id"])
            if key in scores:
                raise ValueError(f"duplicate CPU score: {key}")
            value = float(row["representative_score"])
            if not math.isfinite(value):
                raise ValueError(f"non-finite CPU score: {key}")
            scores[key] = value
        summary = read_json(rooted_path(root, str(seed["summary_path"])))
        timing.append(
            {
                "seed_id": seed_id,
                "pair_count": int(summary["observed_receptor_ligand_pairs"]),
                "elapsed_seconds": float(summary["measured_wall_runtime_seconds"]),
            }
        )
    return scores, timing


def gpu_scores(
    path: Path,
) -> tuple[
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], dict[str, str]],
]:
    scores: dict[tuple[str, str, str], float] = {}
    metadata: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in read_csv(path):
        key = (row["seed_id"], row["receptor_id"], row["ligand_id"])
        if key in scores:
            raise ValueError(f"duplicate GPU score: {key}")
        value = float(row["gpu_score"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite GPU score: {key}")
        if row["status"] != "ok":
            raise ValueError(f"non-ok GPU score: {key}")
        scores[key] = value
        metadata[key] = row
    return scores, metadata


def group_metrics(pair_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(str(row["seed_id"]), str(row["receptor_id"]))].append(row)
    output: list[dict[str, object]] = []
    for (seed_id, receptor_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: str(row["ligand_id"]))
        ligand_ids = [str(row["ligand_id"]) for row in rows]
        cpu = [float(row["cpu_vina_e32_score"]) for row in rows]
        gpu = [float(row["gpu_unidock_detail_score"]) for row in rows]
        deltas = [float(row["score_delta_gpu_minus_cpu"]) for row in rows]
        absolute = [abs(value) for value in deltas]
        overlap_5, count_5 = top_overlap(ligand_ids, cpu, gpu, 0.05)
        overlap_10, count_10 = top_overlap(ligand_ids, cpu, gpu, 0.10)
        output.append(
            {
                "seed_id": seed_id,
                "receptor_id": receptor_id,
                "pair_count": len(rows),
                "pearson": pearson(cpu, gpu),
                "spearman": spearman(cpu, gpu),
                "mean_signed_delta_kcal_per_mol": statistics.fmean(deltas),
                "median_absolute_delta_kcal_per_mol": statistics.median(
                    absolute
                ),
                "p95_absolute_delta_kcal_per_mol": quantile(absolute, 0.95),
                "maximum_absolute_delta_kcal_per_mol": max(absolute),
                "top5pct_count": count_5,
                "top5pct_overlap": overlap_5,
                "top10pct_count": count_10,
                "top10pct_overlap": overlap_10,
            }
        )
    return output


def best_receptor_agreement(pair_rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(str(row["seed_id"]), str(row["ligand_id"]))].append(row)
    agreements = 0
    seed_counts: dict[str, list[int]] = defaultdict(list)
    for (seed_id, _), rows in grouped.items():
        cpu_best = min(
            rows,
            key=lambda row: (
                float(row["cpu_vina_e32_score"]),
                row["receptor_id"],
            ),
        )["receptor_id"]
        gpu_best = min(
            rows,
            key=lambda row: (
                float(row["gpu_unidock_detail_score"]),
                row["receptor_id"],
            ),
        )["receptor_id"]
        agreed = int(cpu_best == gpu_best)
        agreements += agreed
        seed_counts[seed_id].append(agreed)
    return {
        "comparison_count": len(grouped),
        "agreement_count": agreements,
        "agreement_fraction": agreements / len(grouped),
        "by_seed": {
            seed: statistics.fmean(values)
            for seed, values in sorted(seed_counts.items())
        },
    }


def gate_check(
    observed: float,
    threshold: float,
    comparison: str,
) -> dict[str, object]:
    if comparison == "maximum":
        passed = observed <= threshold
    elif comparison == "minimum":
        passed = observed >= threshold
    else:
        raise ValueError(f"unknown gate comparison: {comparison}")
    return {
        "observed": observed,
        "threshold": threshold,
        "comparison": comparison,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    config_path = args.config.resolve()
    config = read_json(config_path)
    receptors, ligands, _ = validate_inputs(root, config)
    outputs = config["outputs"]
    pair_path = rooted_path(root, str(outputs["pairwise_comparison_csv"]))
    group_path = rooted_path(root, str(outputs["group_metrics_csv"]))
    summary_path = rooted_path(root, str(outputs["equivalence_summary_json"]))
    if not args.overwrite and any(
        path.exists() for path in (pair_path, group_path, summary_path)
    ):
        raise FileExistsError("equivalence outputs exist; pass --overwrite")

    gpu_path = rooted_path(root, str(outputs["gpu_scores_csv"]))
    gpu_summary_path = rooted_path(root, str(outputs["gpu_summary_json"]))
    if not gpu_path.is_file() or not gpu_summary_path.is_file():
        raise FileNotFoundError("complete GPU outputs are missing")
    receptor_ids = {row["conformer_id"] for row in receptors}
    cpu, cpu_timing = cpu_scores(root, config, receptor_ids)
    gpu, gpu_metadata = gpu_scores(gpu_path)
    if set(cpu) != set(gpu):
        missing_gpu = sorted(set(cpu) - set(gpu))
        extra_gpu = sorted(set(gpu) - set(cpu))
        raise ValueError(
            f"CPU/GPU pair keys differ; missing={missing_gpu[:5]}, "
            f"extra={extra_gpu[:5]}"
        )

    ligand_metadata = {row["ligand_id"]: row for row in ligands}
    pair_rows: list[dict[str, object]] = []
    for key in sorted(cpu):
        seed_id, receptor_id, ligand_id = key
        cpu_value = cpu[key]
        gpu_value = gpu[key]
        delta = gpu_value - cpu_value
        metadata = gpu_metadata[key]
        ligand = ligand_metadata[ligand_id]
        pair_rows.append(
            {
                "seed_id": seed_id,
                "base_seed": metadata["base_seed"],
                "receptor_id": receptor_id,
                "ligand_id": ligand_id,
                "label": ligand["label"],
                "selection_role": ligand["selection_role"],
                "cpu_vina_e32_score": cpu_value,
                "gpu_unidock_detail_score": gpu_value,
                "score_delta_gpu_minus_cpu": delta,
                "absolute_score_delta": abs(delta),
                "gpu_pose_count": metadata["pose_count"],
                "gpu_output_pose_sha256": metadata["output_pose_sha256"],
            }
        )
    write_csv(pair_path, pair_rows)
    groups = group_metrics(pair_rows)
    write_csv(group_path, groups)

    deltas = [float(row["score_delta_gpu_minus_cpu"]) for row in pair_rows]
    absolute = [abs(value) for value in deltas]
    cpu_values = [float(row["cpu_vina_e32_score"]) for row in pair_rows]
    gpu_values = [float(row["gpu_unidock_detail_score"]) for row in pair_rows]
    label_deltas = {
        label: [
            float(row["score_delta_gpu_minus_cpu"])
            for row in pair_rows
            if row["label"] == label
        ]
        for label in ("active", "decoy")
    }
    label_means = {
        label: statistics.fmean(values) for label, values in label_deltas.items()
    }
    label_gap = abs(label_means["active"] - label_means["decoy"])
    gpu_summary = read_json(gpu_summary_path)
    throughput_reference = config.get("throughput_reference")
    if throughput_reference is None:
        cpu_pairs = sum(int(row["pair_count"]) for row in cpu_timing)
        cpu_seconds = sum(float(row["elapsed_seconds"]) for row in cpu_timing)
        throughput_note = (
            "CPU timing is summed from the configured CPU score sources. "
            "GPU timing sums sequential Uni-Dock receptor-seed batches and "
            "excludes environment installation."
        )
    else:
        cpu_pairs = int(throughput_reference["pair_count"])
        cpu_seconds = float(throughput_reference["elapsed_seconds"])
        throughput_note = str(throughput_reference["comparison_note"])
    gpu_pairs = int(gpu_summary["gpu_pair_count"])
    gpu_seconds = float(gpu_summary["gpu_batch_elapsed_seconds_total"])
    cpu_pairs_per_second = cpu_pairs / cpu_seconds
    gpu_pairs_per_second = gpu_pairs / gpu_seconds
    speedup = gpu_pairs_per_second / cpu_pairs_per_second

    thresholds = config["equivalence_gate"]
    checks = {
        "complete_pairs": {
            "observed": len(pair_rows),
            "threshold": int(config["expected"]["total_gpu_pair_count"]),
            "comparison": "equal",
            "passed": len(pair_rows)
            == int(config["expected"]["total_gpu_pair_count"]),
        },
        "overall_median_absolute_score_delta": gate_check(
            statistics.median(absolute),
            float(
                thresholds[
                    "maximum_overall_median_absolute_score_delta_kcal_per_mol"
                ]
            ),
            "maximum",
        ),
        "overall_p95_absolute_score_delta": gate_check(
            quantile(absolute, 0.95),
            float(
                thresholds[
                    "maximum_overall_p95_absolute_score_delta_kcal_per_mol"
                ]
            ),
            "maximum",
        ),
        "minimum_group_spearman": gate_check(
            min(float(row["spearman"]) for row in groups),
            float(thresholds["minimum_group_spearman"]),
            "minimum",
        ),
        "median_group_top5pct_overlap": gate_check(
            statistics.median(float(row["top5pct_overlap"]) for row in groups),
            float(thresholds["minimum_median_group_top5pct_overlap"]),
            "minimum",
        ),
        "throughput_speedup_vs_recorded_32vcpu": gate_check(
            speedup,
            float(thresholds["minimum_throughput_speedup_vs_recorded_32vcpu"]),
            "minimum",
        ),
        "active_decoy_mean_delta_gap": gate_check(
            label_gap,
            float(
                thresholds[
                    "maximum_active_decoy_mean_delta_gap_kcal_per_mol"
                ]
            ),
            "maximum",
        ),
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    worst_pairs = sorted(
        pair_rows,
        key=lambda row: (
            -float(row["absolute_score_delta"]),
            str(row["seed_id"]),
            str(row["receptor_id"]),
            str(row["ligand_id"]),
        ),
    )[:20]
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "gpu_equivalence_gate_passed"
            if passed
            else "gpu_equivalence_gate_failed"
        ),
        "operation": "train-only engine equivalence and throughput audit; no enrichment or validation metric was calculated",
        "config": {
            "path": relative_path(root, config_path),
            "sha256": file_sha256(config_path),
        },
        "pair_count": len(pair_rows),
        "group_count": len(groups),
        "receptor_count": len(receptors),
        "ligand_count": len(ligands),
        "seed_count": len(config["inputs"]["cpu_seed_runs"]),
        "score_comparison": {
            "overall_pearson": pearson(cpu_values, gpu_values),
            "overall_spearman": spearman(cpu_values, gpu_values),
            "mean_signed_delta_kcal_per_mol": statistics.fmean(deltas),
            "median_absolute_delta_kcal_per_mol": statistics.median(absolute),
            "p95_absolute_delta_kcal_per_mol": quantile(absolute, 0.95),
            "maximum_absolute_delta_kcal_per_mol": max(absolute),
            "minimum_group_spearman": min(
                float(row["spearman"]) for row in groups
            ),
            "median_group_spearman": statistics.median(
                float(row["spearman"]) for row in groups
            ),
            "median_group_top5pct_overlap": statistics.median(
                float(row["top5pct_overlap"]) for row in groups
            ),
            "median_group_top10pct_overlap": statistics.median(
                float(row["top10pct_overlap"]) for row in groups
            ),
            "active_mean_signed_delta_kcal_per_mol": label_means["active"],
            "decoy_mean_signed_delta_kcal_per_mol": label_means["decoy"],
            "active_decoy_mean_delta_gap_kcal_per_mol": label_gap,
            "best_receptor_identity": best_receptor_agreement(pair_rows),
        },
        "throughput": {
            "recorded_cpu_reference_pair_count": cpu_pairs,
            "recorded_cpu_reference_seconds": cpu_seconds,
            "recorded_cpu_pairs_per_second": cpu_pairs_per_second,
            "gpu_pair_count": gpu_pairs,
            "gpu_batch_seconds": gpu_seconds,
            "gpu_pairs_per_second": gpu_pairs_per_second,
            "speedup_vs_recorded_32vcpu": speedup,
            "comparison_note": throughput_note,
        },
        "gate_checks": checks,
        "all_gate_checks_passed": passed,
        "largest_absolute_delta_pairs": [
            {
                "seed_id": row["seed_id"],
                "receptor_id": row["receptor_id"],
                "ligand_id": row["ligand_id"],
                "label": row["label"],
                "cpu_score": row["cpu_vina_e32_score"],
                "gpu_score": row["gpu_unidock_detail_score"],
                "absolute_delta": row["absolute_score_delta"],
            }
            for row in worst_pairs
        ],
        "outputs": {
            "pairwise_comparison_csv": output_descriptor(root, pair_path),
            "group_metrics_csv": output_descriptor(root, group_path),
        },
        "validation_rows": 0,
        "test_rows": 0,
        "enrichment_metrics_calculated": False,
        "next_action": (
            "Preregister and recompute the complete Train-696 eight-receptor evidence with this exact GPU engine before refitting and refreezing any QUBO or comparator. Fresh validation remains closed."
            if passed
            else "Do not use this GPU protocol on fresh validation. Diagnose the train-only outliers or retain the official CPU Vina protocol."
        ),
        "interpretation_note": config["decision_boundary"],
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
