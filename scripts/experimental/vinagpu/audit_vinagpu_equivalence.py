"""Audit Vina-GPU 2.1 against frozen train-only CPU Vina evidence."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from .run_vinagpu_equivalence import (
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
    from run_vinagpu_equivalence import (
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


def read_cpu_scores(
    root: Path,
    config: dict[str, object],
    receptor_ids: set[str],
) -> dict[tuple[str, str, str], float]:
    scores: dict[tuple[str, str, str], float] = {}
    for seed in config["inputs"]["cpu_seed_runs"]:
        seed_id = str(seed["seed_id"])
        for row in read_csv(rooted_path(root, str(seed["scores_path"]))):
            if row["receptor_id"] not in receptor_ids:
                continue
            key = (seed_id, row["receptor_id"], row["ligand_id"])
            if key in scores:
                raise ValueError(f"duplicate CPU score: {key}")
            value = float(row["representative_score"])
            if not math.isfinite(value):
                raise ValueError(f"non-finite CPU score: {key}")
            scores[key] = value
    return scores


def read_gpu_scores(
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
        value = float(row["gpu_vinagpu21_score"])
        if not math.isfinite(value) or row["status"] != "ok":
            raise ValueError(f"invalid GPU score: {key}")
        expected_seed = int(row["base_seed"]) + int(row["seed_offset"])
        if int(row["pair_seed"]) != expected_seed:
            raise ValueError(f"GPU pair seed differs from frozen policy: {key}")
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
        gpu = [float(row["gpu_vinagpu21_score"]) for row in rows]
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
                "median_absolute_delta_kcal_per_mol": statistics.median(absolute),
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
    by_seed: dict[str, list[int]] = defaultdict(list)
    for (seed_id, _), rows in grouped.items():
        cpu_best = min(
            rows,
            key=lambda row: (float(row["cpu_vina_e32_score"]), row["receptor_id"]),
        )["receptor_id"]
        gpu_best = min(
            rows,
            key=lambda row: (float(row["gpu_vinagpu21_score"]), row["receptor_id"]),
        )["receptor_id"]
        agreed = int(cpu_best == gpu_best)
        agreements += agreed
        by_seed[seed_id].append(agreed)
    return {
        "comparison_count": len(grouped),
        "agreement_count": agreements,
        "agreement_fraction": agreements / len(grouped),
        "by_seed": {
            seed: statistics.fmean(values) for seed, values in sorted(by_seed.items())
        },
    }


def gate_check(
    observed: float, threshold: float, comparison: str
) -> dict[str, object]:
    passed = observed <= threshold if comparison == "maximum" else observed >= threshold
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
    if not args.overwrite and any(path.exists() for path in (pair_path, group_path, summary_path)):
        raise FileExistsError("equivalence outputs exist; pass --overwrite")

    gpu_path = rooted_path(root, str(outputs["gpu_scores_csv"]))
    gpu_summary_path = rooted_path(root, str(outputs["gpu_summary_json"]))
    if not gpu_path.is_file() or not gpu_summary_path.is_file():
        raise FileNotFoundError("complete Vina-GPU outputs are missing")
    receptor_ids = {row["conformer_id"] for row in receptors}
    cpu = read_cpu_scores(root, config, receptor_ids)
    gpu, gpu_metadata = read_gpu_scores(gpu_path)
    if set(cpu) != set(gpu):
        missing = sorted(set(cpu) - set(gpu))
        extra = sorted(set(gpu) - set(cpu))
        raise ValueError(
            f"CPU/GPU pair keys differ; missing={missing[:5]}, extra={extra[:5]}"
        )

    ligand_metadata = {row["ligand_id"]: row for row in ligands}
    pair_rows: list[dict[str, object]] = []
    for key in sorted(cpu):
        seed_id, receptor_id, ligand_id = key
        cpu_value = cpu[key]
        gpu_value = gpu[key]
        metadata = gpu_metadata[key]
        ligand = ligand_metadata[ligand_id]
        pair_rows.append(
            {
                "seed_id": seed_id,
                "base_seed": metadata["base_seed"],
                "seed_offset": metadata["seed_offset"],
                "pair_seed": metadata["pair_seed"],
                "receptor_id": receptor_id,
                "ligand_id": ligand_id,
                "label": ligand["label"],
                "selection_role": ligand["selection_role"],
                "cpu_vina_e32_score": cpu_value,
                "gpu_vinagpu21_score": gpu_value,
                "score_delta_gpu_minus_cpu": gpu_value - cpu_value,
                "absolute_score_delta": abs(gpu_value - cpu_value),
                "gpu_pose_count": metadata["pose_count"],
                "gpu_output_pose_path": metadata["output_pose_path"],
                "gpu_output_pose_sha256": metadata["output_pose_sha256"],
            }
        )
    write_csv(pair_path, pair_rows)
    groups = group_metrics(pair_rows)
    write_csv(group_path, groups)

    deltas = [float(row["score_delta_gpu_minus_cpu"]) for row in pair_rows]
    absolute = [abs(value) for value in deltas]
    cpu_values = [float(row["cpu_vina_e32_score"]) for row in pair_rows]
    gpu_values = [float(row["gpu_vinagpu21_score"]) for row in pair_rows]
    label_means = {
        label: statistics.fmean(
            float(row["score_delta_gpu_minus_cpu"])
            for row in pair_rows
            if row["label"] == label
        )
        for label in ("active", "decoy")
    }
    label_gap = abs(label_means["active"] - label_means["decoy"])
    gpu_summary = read_json(gpu_summary_path)
    throughput = config["throughput_reference"]
    cpu_rate = float(throughput["pair_count"]) / float(throughput["elapsed_seconds"])
    gpu_seconds = float(gpu_summary["gpu_pair_elapsed_seconds_total"])
    gpu_rate = len(pair_rows) / gpu_seconds
    speedup = gpu_rate / cpu_rate

    thresholds = config["equivalence_gate"]
    checks = {
        "complete_pairs": {
            "observed": len(pair_rows),
            "threshold": int(config["expected"]["total_gpu_pair_count"]),
            "comparison": "equal",
            "passed": len(pair_rows) == int(config["expected"]["total_gpu_pair_count"]),
        },
        "overall_median_absolute_score_delta": gate_check(
            statistics.median(absolute),
            float(thresholds["maximum_overall_median_absolute_score_delta_kcal_per_mol"]),
            "maximum",
        ),
        "overall_p95_absolute_score_delta": gate_check(
            quantile(absolute, 0.95),
            float(thresholds["maximum_overall_p95_absolute_score_delta_kcal_per_mol"]),
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
        "active_decoy_mean_delta_gap": gate_check(
            label_gap,
            float(thresholds["maximum_active_decoy_mean_delta_gap_kcal_per_mol"]),
            "maximum",
        ),
        "throughput_speedup_vs_recorded_32vcpu": gate_check(
            speedup,
            float(thresholds["minimum_throughput_speedup_vs_recorded_32vcpu"]),
            "minimum",
        ),
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    worst = sorted(
        pair_rows,
        key=lambda row: (
            -float(row["absolute_score_delta"]),
            str(row["seed_id"]),
            str(row["receptor_id"]),
            str(row["ligand_id"]),
        ),
    )[:20]

    diagnostic_directory = rooted_path(
        root, str(outputs["diagnostic_pose_directory"])
    )
    diagnostic_directory.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in diagnostic_directory.glob("*.pdbqt"):
            path.unlink()
    diagnostic_poses: list[dict[str, object]] = []
    for row in worst:
        source = rooted_path(root, str(row["gpu_output_pose_path"]))
        name = (
            f"{row['seed_id']}__{row['receptor_id']}__"
            f"{row['ligand_id']}.pdbqt"
        )
        destination = diagnostic_directory / name
        shutil.copyfile(source, destination)
        diagnostic_poses.append(
            {
                "seed_id": row["seed_id"],
                "receptor_id": row["receptor_id"],
                "ligand_id": row["ligand_id"],
                "absolute_delta": row["absolute_score_delta"],
                **output_descriptor(root, destination),
            }
        )

    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "gpu_equivalence_gate_passed" if passed else "gpu_equivalence_gate_failed",
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
        "seed_policy": config["seed_policy"],
        "score_comparison": {
            "overall_pearson": pearson(cpu_values, gpu_values),
            "overall_spearman": spearman(cpu_values, gpu_values),
            "mean_signed_delta_kcal_per_mol": statistics.fmean(deltas),
            "median_absolute_delta_kcal_per_mol": statistics.median(absolute),
            "p95_absolute_delta_kcal_per_mol": quantile(absolute, 0.95),
            "maximum_absolute_delta_kcal_per_mol": max(absolute),
            "minimum_group_spearman": min(float(row["spearman"]) for row in groups),
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
            "recorded_cpu_reference_pair_count": int(throughput["pair_count"]),
            "recorded_cpu_reference_seconds": float(throughput["elapsed_seconds"]),
            "recorded_cpu_pairs_per_second": cpu_rate,
            "gpu_pair_count": len(pair_rows),
            "gpu_pair_process_seconds": gpu_seconds,
            "gpu_pairs_per_second": gpu_rate,
            "speedup_vs_recorded_32vcpu": speedup,
            "comparison_note": throughput["comparison_note"],
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
                "gpu_score": row["gpu_vinagpu21_score"],
                "absolute_delta": row["absolute_score_delta"],
            }
            for row in worst
        ],
        "diagnostic_poses": diagnostic_poses,
        "outputs": {
            "pairwise_comparison_csv": output_descriptor(root, pair_path),
            "group_metrics_csv": output_descriptor(root, group_path),
        },
        "validation_rows": 0,
        "test_rows": 0,
        "enrichment_metrics_calculated": False,
        "next_action": (
            "Preregister and recompute the complete Train-696 receptor evidence with this exact Vina-GPU runtime before refitting any QUBO or comparator."
            if passed
            else "Do not use this Vina-GPU protocol for larger or fresh-data runs; retain official CPU Vina or diagnose this consumed-train failure."
        ),
        "interpretation_note": config["decision_boundary"],
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
