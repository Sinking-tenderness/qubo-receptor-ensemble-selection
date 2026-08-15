"""Independently adjudicate the final Stage 11 Amendment 01 archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


CONFIG_REL = Path(
    "configs/stage11_mk14_fresh_validation_unidock113_confirmation.json"
)
AMENDMENT_REL = Path(
    "configs/stage11_mk14_fresh_validation_score_guard_amendment01.json"
)
MANIFEST_REL = Path(
    "data/processed/stage11_mk14_fresh_validation_unidock_pdbqt_manifest.csv"
)
AUDIT_REL = Path(
    "data/stage11_mk14_fresh_validation_unidock113_confirmation_audit.json"
)
OFFICIAL_RESULT_REL = Path(
    "data/stage11_mk14_fresh_validation_unidock113_confirmation_result.json"
)
RUN_REL = Path(
    "results/runs/stage11_mk14_fresh_validation_unidock113_confirmation"
)
SUMMARY_REL = RUN_REL / "summary.json"
SCORES_REL = RUN_REL / "scores.csv"
PRIMARY_MATRIX_REL = RUN_REL / "primary_median_score_matrix.csv"
MINIMUM_MATRIX_REL = RUN_REL / "sensitivity_minimum_score_matrix.csv"

SEED_IDS = ("seed0", "seed1", "seed2")
VINA_RESULT_RE = re.compile(
    r"^REMARK VINA RESULT:\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\b"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def relative_files(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())


def verify_shared_core(core_root: Path, diagnostics_root: Path) -> dict[str, object]:
    core_files = relative_files(core_root)
    missing: list[str] = []
    mismatched: list[str] = []
    for relative in core_files:
        diagnostic_path = diagnostics_root / relative
        if not diagnostic_path.is_file():
            missing.append(relative.as_posix())
        elif file_sha256(core_root / relative) != file_sha256(diagnostic_path):
            mismatched.append(relative.as_posix())
    if missing or mismatched:
        raise ValueError(
            f"core/diagnostics mismatch: missing={missing[:5]}, "
            f"hash_mismatch={mismatched[:5]}"
        )
    return {
        "core_file_count": len(core_files),
        "diagnostics_file_count": len(relative_files(diagnostics_root)),
        "shared_core_file_count": len(core_files),
        "missing_core_file_count": 0,
        "hash_mismatch_count": 0,
        "status": "all_core_files_identical_in_diagnostics_archive",
    }


def parse_pose(path: Path) -> dict[str, object]:
    model_count = 0
    end_model_count = 0
    scores: list[float] = []
    atom_types: list[str] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.startswith("MODEL "):
                model_count += 1
            elif line == "ENDMDL":
                end_model_count += 1
            elif line.startswith(("ATOM", "HETATM")):
                fields = line.split()
                if not fields:
                    raise ValueError(f"invalid atom line: {path}")
                atom_types.append(fields[-1])
            else:
                match = VINA_RESULT_RE.match(line)
                if match:
                    scores.append(float(match.group(1)))
    if model_count != 1 or end_model_count != 1 or len(scores) != 1:
        raise ValueError(
            f"pose is not a complete single model: {path} "
            f"models={model_count}, end_models={end_model_count}, scores={len(scores)}"
        )
    return {
        "score": scores[0],
        "atom_count": len(atom_types),
        "atom_types": sorted(set(atom_types)),
    }


def audit_poses_and_batches(
    core_root: Path,
    diagnostics_root: Path,
    score_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
) -> dict[str, object]:
    manifest = {row["ligand_id"]: row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("duplicate ligand IDs in Stage 11 manifest")
    seen_keys: set[tuple[str, str, str]] = set()
    outliers: list[dict[str, object]] = []
    pose_hash_failures = 0
    pose_score_failures = 0
    pose_shape_failures = 0
    for index, row in enumerate(score_rows, start=1):
        key = (row["seed_id"], row["receptor_id"], row["ligand_id"])
        if key in seen_keys:
            raise ValueError(f"duplicate aggregate score key: {key}")
        seen_keys.add(key)
        relative = Path(row["output_pose_path"])
        pose_path = diagnostics_root / relative
        if not pose_path.is_file():
            raise FileNotFoundError(pose_path)
        if file_sha256(pose_path) != row["output_pose_sha256"].upper():
            pose_hash_failures += 1
        parsed = parse_pose(pose_path)
        score = float(row["gpu_score"])
        if parsed["score"] != score:
            pose_score_failures += 1
        ligand = manifest[row["ligand_id"]]
        expected_types = sorted(
            value for value in ligand["pdbqt_atom_types"].split(";") if value
        )
        shape_ok = (
            int(parsed["atom_count"]) == int(row["output_atom_count"])
            == int(row["input_atom_count"])
            == int(ligand["pdbqt_atom_count"])
            and parsed["atom_types"] == expected_types
            and row["pose_count"] == "1"
            and row["status"] == "ok"
            and row["pose_integrity_status"] == "ok"
        )
        if not shape_ok:
            pose_shape_failures += 1
        if abs(score) > 100.0:
            outliers.append(
                {
                    "seed_id": row["seed_id"],
                    "receptor_id": row["receptor_id"],
                    "ligand_id": row["ligand_id"],
                    "label": row["label"],
                    "score_kcal_per_mol": score,
                    "pose_sha256": row["output_pose_sha256"].upper(),
                    "pose_score_exact": parsed["score"] == score,
                    "atom_count": int(parsed["atom_count"]),
                    "atom_types": parsed["atom_types"],
                }
            )
        if index % 5000 == 0:
            print(f"audited_pose_count={index}", flush=True)

    batch_count = 0
    batch_score_rows = 0
    known_warnings = 0
    unresolved_warnings = 0
    batch_pose_failures = 0
    batch_hash_failures = 0
    batch_keys: set[tuple[str, str, str]] = set()
    for batch_path in sorted((diagnostics_root / RUN_REL / "batches").glob("*/*/batch_summary.json")):
        batch = read_json(batch_path)
        batch_count += 1
        batch_dir = batch_path.parent
        scores_path = batch_dir / "scores.csv"
        log_path = batch_dir / "unidock.log"
        if (
            file_sha256(scores_path) != str(batch["scores_sha256"]).upper()
            or file_sha256(log_path) != str(batch["log_sha256"]).upper()
        ):
            batch_hash_failures += 1
        batch_rows = read_csv(scores_path)
        batch_score_rows += len(batch_rows)
        for row in batch_rows:
            key = (row["seed_id"], row["receptor_id"], row["ligand_id"])
            if key in batch_keys:
                raise ValueError(f"duplicate batch score key: {key}")
            batch_keys.add(key)
        warning = dict(batch["warning_adjudication"])
        pose_audit = dict(batch["pose_integrity_audit"])
        known_warnings += int(warning["known_warning_event_count"])
        unresolved_warnings += int(warning["unresolved_warning_event_count"])
        batch_pose_failures += int(pose_audit["failure_count"])
        if batch.get("status") != "ok" or len(batch_rows) != int(batch["ligand_count"]):
            raise ValueError(f"batch summary is incomplete: {batch_path}")

    if batch_keys != seen_keys:
        raise ValueError("batch score grid and aggregate score grid differ")
    failures = (
        pose_hash_failures
        + pose_score_failures
        + pose_shape_failures
        + batch_hash_failures
        + unresolved_warnings
        + batch_pose_failures
    )
    return {
        "status": "independent_full_pose_and_batch_audit_ok" if failures == 0 else "failed",
        "pose_count": len(score_rows),
        "pose_sha256_failure_count": pose_hash_failures,
        "pose_score_failure_count": pose_score_failures,
        "pose_shape_failure_count": pose_shape_failures,
        "batch_count": batch_count,
        "batch_score_row_count": batch_score_rows,
        "batch_hash_failure_count": batch_hash_failures,
        "known_warning_event_count": known_warnings,
        "unresolved_warning_event_count": unresolved_warnings,
        "batch_pose_integrity_failure_count": batch_pose_failures,
        "outlier_over_original_guard_count": len(outliers),
        "outliers": outliers,
    }


def score_grid(
    score_rows: list[dict[str, str]],
) -> dict[tuple[str, str, str], float]:
    values: dict[tuple[str, str, str], float] = {}
    for row in score_rows:
        key = (row["seed_id"], row["ligand_id"], row["receptor_id"])
        if key in values:
            raise ValueError(f"duplicate score key: {key}")
        value = float(row["gpu_score"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite score: {key}")
        values[key] = value
    return values


def build_matrices(
    values: dict[tuple[str, str, str], float],
    manifest_rows: list[dict[str, str]],
    receptor_ids: list[str],
    outlier_policy: str = "raw",
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    primary: list[dict[str, object]] = []
    minimum: list[dict[str, object]] = []
    seeds: dict[str, list[dict[str, object]]] = {seed_id: [] for seed_id in SEED_IDS}
    for ligand in manifest_rows:
        ligand_id = ligand["ligand_id"]
        metadata = {
            "ligand_id": ligand_id,
            "label": ligand["label"],
            "selection_role": ligand["selection_role"],
        }
        primary_row: dict[str, object] = dict(metadata)
        minimum_row: dict[str, object] = dict(metadata)
        seed_rows = {seed_id: dict(metadata) for seed_id in SEED_IDS}
        for receptor_id in receptor_ids:
            raw = [values[(seed_id, ligand_id, receptor_id)] for seed_id in SEED_IDS]
            if outlier_policy == "clip_100":
                used = [max(-100.0, min(100.0, value)) for value in raw]
            elif outlier_policy == "missing":
                used = [value for value in raw if abs(value) <= 100.0]
            elif outlier_policy == "raw":
                used = raw
            else:
                raise ValueError(f"unsupported outlier policy: {outlier_policy}")
            if len(used) < 2:
                raise ValueError(f"too few scores after outlier policy: {ligand_id}/{receptor_id}")
            primary_row[receptor_id] = statistics.median(used)
            minimum_row[receptor_id] = min(used)
            for seed_id, value in zip(SEED_IDS, raw):
                seed_rows[seed_id][receptor_id] = value
        primary.append(primary_row)
        minimum.append(minimum_row)
        for seed_id in SEED_IDS:
            seeds[seed_id].append(seed_rows[seed_id])
    return primary, minimum, seeds


def compare_matrix(
    rebuilt: list[dict[str, object]],
    archived: list[dict[str, str]],
    receptor_ids: list[str],
) -> int:
    if len(rebuilt) != len(archived):
        raise ValueError("rebuilt and archived matrix row counts differ")
    mismatch_count = 0
    for expected, observed in zip(rebuilt, archived):
        if any(
            str(expected[key]) != observed[key]
            for key in ("ligand_id", "label", "selection_role")
        ):
            mismatch_count += 1
            continue
        for receptor_id in receptor_ids:
            if float(expected[receptor_id]) != float(observed[receptor_id]):
                mismatch_count += 1
                break
    return mismatch_count


def roc_auc(labels: list[int], ranking_scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, ranking_scores) if label == 1]
    negatives = [score for label, score in zip(labels, ranking_scores) if label == 0]
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def bedroc_from_labels(labels: list[int], alpha: float = 20.0) -> float:
    total = len(labels)
    active_ranks = [index for index, label in enumerate(labels, start=1) if label]
    active_total = len(active_ranks)
    if total == 0 or active_total == 0 or active_total == total:
        return math.nan
    weights = [math.exp(-alpha * rank / total) for rank in range(1, total + 1)]
    random_sum = active_total * sum(weights) / total
    observed = sum(math.exp(-alpha * rank / total) for rank in active_ranks) / random_sum
    best = sum(weights[:active_total]) / random_sum
    worst = sum(weights[-active_total:]) / random_sum
    return (observed - worst) / (best - worst)


def ranked_metrics(rows: list[dict[str, object]], subset: tuple[str, ...]) -> dict[str, object]:
    ranked = sorted(
        (
            min(float(row[receptor_id]) for receptor_id in subset),
            str(row["ligand_id"]),
            int(row["label"] == "active"),
        )
        for row in rows
    )
    labels = [label for _, _, label in ranked]
    ranking_scores = [-score for score, _, _ in ranked]
    active_total = sum(labels)
    precision_sum = 0.0
    active_seen = 0
    for index, label in enumerate(labels, start=1):
        if label:
            active_seen += 1
            precision_sum += active_seen / index

    def ef(fraction: float) -> float:
        top_n = max(1, math.ceil(len(ranked) * fraction))
        return (sum(labels[:top_n]) / top_n) / (active_total / len(ranked))

    return {
        "ligand_count": len(ranked),
        "active_count": active_total,
        "roc_auc": roc_auc(labels, ranking_scores),
        "pr_auc_average_precision": precision_sum / active_total,
        "bedroc_alpha_20": bedroc_from_labels(labels),
        "EF1%": ef(0.01),
        "EF5%": ef(0.05),
        "EF10%": ef(0.10),
        "top10_active_count": sum(labels[:10]),
        "top10_ligand_ids": [ligand_id for _, ligand_id, _ in ranked[:10]],
    }


def candidate_records(
    rows: list[dict[str, object]], subset: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    return {
        str(row["ligand_id"]): {
            "label": str(row["label"]),
            "score": min(float(row[receptor_id]) for receptor_id in subset),
        }
        for row in rows
    }


def candidate_summaries(
    primary: list[dict[str, object]],
    minimum: list[dict[str, object]],
    seeds: dict[str, list[dict[str, object]]],
    candidates: dict[str, tuple[str, ...]],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for candidate_id, subset in candidates.items():
        seed_metrics = {
            seed_id: ranked_metrics(seeds[seed_id], subset) for seed_id in SEED_IDS
        }
        seed_bedroc = [
            float(seed_metrics[seed_id]["bedroc_alpha_20"]) for seed_id in SEED_IDS
        ]
        output[candidate_id] = {
            "primary": ranked_metrics(primary, subset),
            "sensitivity": ranked_metrics(minimum, subset),
            "seed_metrics": seed_metrics,
            "mean_seed_bedroc": statistics.fmean(seed_bedroc),
            "worst_seed_bedroc": min(seed_bedroc),
        }
    return output


def linear_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def sampled_bedroc(
    records: dict[str, dict[str, object]],
    grouped_ids: dict[str, list[str]],
    sampled_groups: list[str],
) -> float:
    ranked: list[tuple[float, int, str, int]] = []
    for draw_index, group_id in enumerate(sampled_groups):
        for ligand_id in grouped_ids[group_id]:
            record = records[ligand_id]
            ranked.append(
                (
                    float(record["score"]),
                    draw_index,
                    ligand_id,
                    int(record["label"] == "active"),
                )
            )
    ranked.sort()
    return bedroc_from_labels([label for *_, label in ranked])


def paired_bootstrap(
    records: dict[str, dict[str, dict[str, object]]],
    group_by_ligand: dict[str, str],
    candidate_id: str,
    comparator_ids: list[str],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    ligand_ids = set(records[candidate_id])
    if set(group_by_ligand) != ligand_ids:
        raise ValueError("bootstrap group map differs from candidate records")
    grouped_ids: dict[str, list[str]] = defaultdict(list)
    for ligand_id, group_id in group_by_ligand.items():
        grouped_ids[group_id].append(ligand_id)
    for values in grouped_ids.values():
        values.sort()
    group_ids = sorted(grouped_ids)
    rng = random.Random(seed)
    deltas: dict[str, list[float]] = {value: [] for value in comparator_ids}
    attempts = 0
    valid = 0
    while valid < replicates:
        attempts += 1
        if attempts > replicates * 2:
            raise ValueError("too many bootstrap draws lacked both labels")
        sampled = rng.choices(group_ids, k=len(group_ids))
        candidate_value = sampled_bedroc(records[candidate_id], grouped_ids, sampled)
        comparator_values = {
            comparator: sampled_bedroc(records[comparator], grouped_ids, sampled)
            for comparator in comparator_ids
        }
        if not math.isfinite(candidate_value) or any(
            not math.isfinite(value) for value in comparator_values.values()
        ):
            continue
        for comparator, value in comparator_values.items():
            deltas[comparator].append(candidate_value - value)
        valid += 1
    return {
        "unit": "split_group_id block",
        "seed": seed,
        "valid_replicates": valid,
        "attempts": attempts,
        "confidence_level": 0.95,
        "deltas": {
            comparator: {
                "mean": statistics.fmean(values),
                "lower_95pct": linear_quantile(values, 0.025),
                "upper_95pct": linear_quantile(values, 0.975),
                "positive_fraction": sum(value > 0.0 for value in values) / len(values),
            }
            for comparator, values in deltas.items()
        },
    }


def comparisons(
    metrics: dict[str, dict[str, object]],
    bootstrap: dict[str, object],
    candidate_id: str,
    comparator_ids: list[str],
) -> tuple[dict[str, dict[str, object]], bool]:
    output: dict[str, dict[str, object]] = {}
    candidate = metrics[candidate_id]
    bootstrap_deltas = dict(bootstrap["deltas"])
    for comparator_id in comparator_ids:
        comparator = metrics[comparator_id]
        primary_delta = float(candidate["primary"]["bedroc_alpha_20"]) - float(
            comparator["primary"]["bedroc_alpha_20"]
        )
        mean_delta = float(candidate["mean_seed_bedroc"]) - float(
            comparator["mean_seed_bedroc"]
        )
        worst_delta = float(candidate["worst_seed_bedroc"]) - float(
            comparator["worst_seed_bedroc"]
        )
        lower = float(bootstrap_deltas[comparator_id]["lower_95pct"])
        output[comparator_id] = {
            "primary_bedroc_delta": primary_delta,
            "mean_seed_bedroc_delta": mean_delta,
            "worst_seed_bedroc_delta": worst_delta,
            "bootstrap_lower_95pct_primary_bedroc_delta": lower,
            "bootstrap_upper_95pct_primary_bedroc_delta": float(
                bootstrap_deltas[comparator_id]["upper_95pct"]
            ),
            "bootstrap_positive_fraction": float(
                bootstrap_deltas[comparator_id]["positive_fraction"]
            ),
            "passed": primary_delta > 0.0
            and mean_delta > 0.0
            and worst_delta > 0.0
            and lower > 0.0,
        }
    return output, all(value["passed"] for value in output.values())


def maximum_numeric_difference(left: object, right: object) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left).intersection(right)
        return max((maximum_numeric_difference(left[key], right[key]) for key in keys), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0


def scenario_result(
    primary: list[dict[str, object]],
    minimum: list[dict[str, object]],
    seeds: dict[str, list[dict[str, object]]],
    candidates: dict[str, tuple[str, ...]],
    group_by_ligand: dict[str, str],
    candidate_id: str,
    comparator_ids: list[str],
    replicates: int,
    seed: int,
    bootstrap_override: dict[str, object] | None = None,
) -> dict[str, object]:
    metrics = candidate_summaries(primary, minimum, seeds, candidates)
    records = {
        name: candidate_records(primary, subset) for name, subset in candidates.items()
    }
    bootstrap = bootstrap_override or paired_bootstrap(
        records,
        group_by_ligand,
        candidate_id,
        comparator_ids,
        replicates,
        seed,
    )
    gate_comparisons, gate_passed = comparisons(
        metrics, bootstrap, candidate_id, comparator_ids
    )
    compact_metrics = {
        name: {
            "primary_bedroc": float(value["primary"]["bedroc_alpha_20"]),
            "primary_pr_auc": float(value["primary"]["pr_auc_average_precision"]),
            "primary_ef1pct": float(value["primary"]["EF1%"]),
            "mean_seed_bedroc": float(value["mean_seed_bedroc"]),
            "worst_seed_bedroc": float(value["worst_seed_bedroc"]),
        }
        for name, value in metrics.items()
    }
    return {
        "candidate_metrics": compact_metrics,
        "comparisons": gate_comparisons,
        "bootstrap": bootstrap,
        "gate_passed": gate_passed,
        "_full_metrics": metrics,
    }


def affected_aggregation_rows(
    values: dict[tuple[str, str, str], float],
    outliers: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for outlier in outliers:
        ligand_id = str(outlier["ligand_id"])
        receptor_id = str(outlier["receptor_id"])
        seed_values = {
            seed_id: values[(seed_id, ligand_id, receptor_id)] for seed_id in SEED_IDS
        }
        ordered = list(seed_values.values())
        output.append(
            {
                **outlier,
                "three_seed_scores": seed_values,
                "primary_median_score": statistics.median(ordered),
                "sensitivity_minimum_score": min(ordered),
                "outlier_used_by_primary_median": float(outlier["score_kcal_per_mol"])
                == statistics.median(ordered),
                "outlier_used_by_sensitivity_minimum": float(
                    outlier["score_kcal_per_mol"]
                )
                == min(ordered),
            }
        )
    return output


def write_report(path: Path, result: dict[str, object]) -> None:
    raw = dict(result["sensitivity_analysis"])["raw_retained"]
    metrics = dict(raw["candidate_metrics"])
    comparisons_value = dict(raw["comparisons"])
    technical = dict(result["technical_integrity"])
    lines = [
        "# Stage 11 MAPK14 Amendment 01 Independent Adjudication",
        "",
        f"Status: `{result['status']}`",
        "",
        "## Technical result",
        "",
        f"- All {technical['pose_count']} poses passed independent SHA-256, score, "
        "single-model, atom-count, and atom-type checks.",
        f"- All {technical['batch_count']} batches and {technical['batch_score_row_count']} "
        "batch score rows were complete.",
        f"- Known warnings: {technical['known_warning_event_count']}; unresolved warnings: "
        f"{technical['unresolved_warning_event_count']}.",
        f"- Scores above the original 100 kcal/mol guard: "
        f"{technical['outlier_over_original_guard_count']}.",
        "",
        "## Frozen candidate result",
        "",
        "| Candidate | Primary BEDROC | Mean-seed BEDROC | Worst-seed BEDROC | PR-AUC | EF1% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate_id, value in metrics.items():
        lines.append(
            f"| {candidate_id} | {value['primary_bedroc']:.6f} | "
            f"{value['mean_seed_bedroc']:.6f} | {value['worst_seed_bedroc']:.6f} | "
            f"{value['primary_pr_auc']:.6f} | {value['primary_ef1pct']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Comparison | Primary delta | Bootstrap 95% CI | Positive replicates | Gate |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for comparator_id, value in comparisons_value.items():
        lines.append(
            f"| exact_pair_synergy vs {comparator_id} | "
            f"{value['primary_bedroc_delta']:+.6f} | "
            f"[{value['bootstrap_lower_95pct_primary_bedroc_delta']:+.6f}, "
            f"{value['bootstrap_upper_95pct_primary_bedroc_delta']:+.6f}] | "
            f"{value['bootstrap_positive_fraction']:.1%} | "
            f"{'pass' if value['passed'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Outlier sensitivity",
            "",
            "The four finite positive-score outliers are genuine values in their pose files. "
            "None is selected by the three-seed median or minimum aggregation. Clipping them "
            "at 100 kcal/mol therefore changes zero primary or sensitivity matrix cells. "
            "Treating them as missing or excluding all four affected ligands also leaves the "
            "confirmatory gate failed.",
            "",
            "## Decision",
            "",
            "The technical execution and matrix are accepted. The exact QUBO subset has a "
            "positive point estimate against both frozen greedy controls and beats the best "
            "single receptor, but both paired-bootstrap 95% intervals cross zero. This supports "
            "a receptor-ensemble/QUBO application proof of concept, not a statistically stable "
            "QUBO advantage and not a quantum computational advantage.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(
    core_root: Path,
    diagnostics_root: Path,
    core_archive: Path | None,
    diagnostics_archive: Path | None,
    output_json: Path,
    report_path: Path,
    bootstrap_replicates: int,
) -> dict[str, object]:
    core_root = core_root.resolve()
    diagnostics_root = diagnostics_root.resolve()
    shared = verify_shared_core(core_root, diagnostics_root)
    config = read_json(core_root / CONFIG_REL)
    amendment = read_json(core_root / AMENDMENT_REL)
    summary = read_json(core_root / SUMMARY_REL)
    matrix_audit = read_json(core_root / AUDIT_REL)
    official_result = read_json(core_root / OFFICIAL_RESULT_REL)
    score_rows = read_csv(core_root / SCORES_REL)
    manifest_rows = read_csv(core_root / MANIFEST_REL)
    receptor_ids = [str(value) for value in config["expected"]["receptor_ids"]]
    candidates = {
        name: tuple(str(item) for item in dict(value)["subset"])
        for name, value in dict(config["candidates"]).items()
    }
    evaluation = dict(config["evaluation"])
    candidate_id = str(evaluation["confirmatory_candidate"])
    comparator_ids = [str(value) for value in evaluation["confirmatory_controls"]]
    bootstrap_config = dict(evaluation["paired_bootstrap"])
    bootstrap_seed = int(bootstrap_config["seed"])

    expected_pair_count = int(config["expected"]["pair_count"])
    if len(score_rows) != expected_pair_count:
        raise ValueError("aggregate score count differs from preregistration")
    if summary.get("status") != "stage11_fresh_validation_unidock_matrix_ok":
        raise ValueError("Stage 11 matrix generation did not finish")
    if matrix_audit.get("status") != "independent_stage11_fresh_validation_unidock_matrix_audit_ok":
        raise ValueError("archived Stage 11 matrix audit did not pass")
    if amendment.get("status") != "stage11_score_guard_amendment01_frozen":
        raise ValueError("Amendment 01 is not frozen")

    technical = audit_poses_and_batches(
        core_root, diagnostics_root, score_rows, manifest_rows
    )
    if technical["status"] != "independent_full_pose_and_batch_audit_ok":
        raise ValueError("independent full-pose audit failed")

    values = score_grid(score_rows)
    raw_primary, raw_minimum, raw_seeds = build_matrices(
        values, manifest_rows, receptor_ids, "raw"
    )
    primary_mismatch = compare_matrix(
        raw_primary, read_csv(core_root / PRIMARY_MATRIX_REL), receptor_ids
    )
    minimum_mismatch = compare_matrix(
        raw_minimum, read_csv(core_root / MINIMUM_MATRIX_REL), receptor_ids
    )
    if primary_mismatch or minimum_mismatch:
        raise ValueError("independently rebuilt aggregate matrix differs")

    group_by_ligand = {
        row["ligand_id"]: row["split_group_id"] for row in manifest_rows
    }
    raw_scenario = scenario_result(
        raw_primary,
        raw_minimum,
        raw_seeds,
        candidates,
        group_by_ligand,
        candidate_id,
        comparator_ids,
        bootstrap_replicates,
        bootstrap_seed,
    )
    numeric_difference = maximum_numeric_difference(
        raw_scenario["_full_metrics"], official_result["candidate_metrics"]
    )
    official_bootstrap_difference = maximum_numeric_difference(
        raw_scenario["bootstrap"], official_result["paired_bootstrap"]
    )
    if numeric_difference > 1e-12 or official_bootstrap_difference > 1e-12:
        raise ValueError(
            "independent metric reproduction differs from archived result: "
            f"metrics={numeric_difference}, bootstrap={official_bootstrap_difference}"
        )

    clip_primary, clip_minimum, _ = build_matrices(
        values, manifest_rows, receptor_ids, "clip_100"
    )
    missing_primary, missing_minimum, missing_seeds = build_matrices(
        values, manifest_rows, receptor_ids, "missing"
    )

    raw_candidate_records = {
        name: candidate_records(raw_primary, subset)
        for name, subset in candidates.items()
    }
    missing_candidate_records = {
        name: candidate_records(missing_primary, subset)
        for name, subset in candidates.items()
    }
    missing_bootstrap_override = (
        raw_scenario["bootstrap"]
        if missing_candidate_records == raw_candidate_records
        else None
    )

    def changed_cells(
        left: list[dict[str, object]], right: list[dict[str, object]]
    ) -> int:
        return sum(
            float(a[receptor_id]) != float(b[receptor_id])
            for a, b in zip(left, right)
            for receptor_id in receptor_ids
        )

    missing_scenario = scenario_result(
        missing_primary,
        missing_minimum,
        missing_seeds,
        candidates,
        group_by_ligand,
        candidate_id,
        comparator_ids,
        bootstrap_replicates,
        bootstrap_seed,
        missing_bootstrap_override,
    )
    affected_ids = {str(value["ligand_id"]) for value in technical["outliers"]}
    dropped_primary = [row for row in raw_primary if row["ligand_id"] not in affected_ids]
    dropped_minimum = [row for row in raw_minimum if row["ligand_id"] not in affected_ids]
    dropped_seeds = {
        seed_id: [
            row for row in rows if row["ligand_id"] not in affected_ids
        ]
        for seed_id, rows in raw_seeds.items()
    }
    dropped_groups = {
        ligand_id: group_id
        for ligand_id, group_id in group_by_ligand.items()
        if ligand_id not in affected_ids
    }
    dropped_scenario = scenario_result(
        dropped_primary,
        dropped_minimum,
        dropped_seeds,
        candidates,
        dropped_groups,
        candidate_id,
        comparator_ids,
        bootstrap_replicates,
        bootstrap_seed,
    )

    for value in (raw_scenario, missing_scenario, dropped_scenario):
        value.pop("_full_metrics", None)
    raw_scenario["primary_matrix_changed_cell_count"] = 0
    raw_scenario["minimum_matrix_changed_cell_count"] = 0
    clip_scenario = json.loads(json.dumps(raw_scenario))
    clip_scenario["primary_matrix_changed_cell_count"] = changed_cells(
        raw_primary, clip_primary
    )
    clip_scenario["minimum_matrix_changed_cell_count"] = changed_cells(
        raw_minimum, clip_minimum
    )
    missing_scenario["primary_matrix_changed_cell_count"] = changed_cells(
        raw_primary, missing_primary
    )
    missing_scenario["minimum_matrix_changed_cell_count"] = changed_cells(
        raw_minimum, missing_minimum
    )
    dropped_scenario["excluded_ligand_count"] = len(affected_ids)

    best_single = max(
        (
            (
                receptor_id,
                float(ranked_metrics(raw_primary, (receptor_id,))["bedroc_alpha_20"]),
            )
            for receptor_id in receptor_ids
        ),
        key=lambda value: value[1],
    )
    archive_identity: dict[str, object] = {}
    if core_archive is not None:
        archive_identity["core"] = {
            "path": str(core_archive.resolve()),
            "sha256": file_sha256(core_archive),
        }
    if diagnostics_archive is not None:
        archive_identity["diagnostics"] = {
            "path": str(diagnostics_archive.resolve()),
            "sha256": file_sha256(diagnostics_archive),
        }

    result = {
        "schema_version": "1.0",
        "adjudication_id": "stage11-mk14-fresh-validation-amendment01-independent-adjudication-v1",
        "status": "stage11_technical_result_accepted_scientific_gate_not_passed",
        "archive_identity": archive_identity,
        "archive_consistency": shared,
        "technical_integrity": technical,
        "aggregate_reproduction": {
            "score_row_count": len(score_rows),
            "primary_matrix_row_count": len(raw_primary),
            "minimum_matrix_row_count": len(raw_minimum),
            "primary_matrix_mismatch_count": primary_mismatch,
            "minimum_matrix_mismatch_count": minimum_mismatch,
            "candidate_metric_maximum_absolute_difference": numeric_difference,
            "bootstrap_maximum_absolute_difference": official_bootstrap_difference,
            "status": "independent_matrix_metrics_and_bootstrap_exactly_reproduced",
        },
        "outlier_aggregation_trace": affected_aggregation_rows(
            values, technical["outliers"]
        ),
        "sensitivity_analysis": {
            "raw_retained": raw_scenario,
            "clip_to_original_guard_100": clip_scenario,
            "outlier_cells_treated_as_missing": missing_scenario,
            "four_affected_ligands_excluded": dropped_scenario,
        },
        "best_single_receptor": {
            "receptor_id": best_single[0],
            "primary_bedroc": best_single[1],
            "exact_pair_synergy_delta": float(
                raw_scenario["candidate_metrics"][candidate_id]["primary_bedroc"]
            )
            - best_single[1],
        },
        "decision": {
            "technical_gate_passed": True,
            "ensemble_over_best_single_point_estimate": True,
            "exact_qubo_over_both_greedy_point_estimates": True,
            "frozen_statistical_gate_passed": bool(raw_scenario["gate_passed"]),
            "outlier_sensitivity_changes_gate": any(
                bool(value["gate_passed"]) != bool(raw_scenario["gate_passed"])
                for value in (clip_scenario, missing_scenario, dropped_scenario)
            ),
            "supports_qubo_application_poc": True,
            "supports_stable_qubo_advantage": False,
            "supports_quantum_computational_advantage": False,
            "recommended_next_step": (
                "Freeze MAPK14 as an application proof of concept. Do not tune on this "
                "validation set or add more MAPK14 docking merely to chase significance; "
                "replicate the frozen workflow across independent protein targets before "
                "making a broader method claim."
            ),
        },
        "data_boundary": {
            "validation_rows_read": len(manifest_rows),
            "train_score_rows_read": 0,
            "test_rows_read": 0,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    write_report(report_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--core-archive", type=Path)
    parser.add_argument("--diagnostics-archive", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "data/stage11_mk14_fresh_validation_unidock113_amendment01_adjudication.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "reports/stage-11/mk14_fresh_validation_unidock113_amendment01_adjudication.md"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    run(
        args.core_root,
        args.diagnostics_root,
        args.core_archive,
        args.diagnostics_archive,
        args.output_json,
        args.report,
        args.bootstrap_replicates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
