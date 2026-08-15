"""Evaluate the Stage 07c four-seed and warning-replay confirmation."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

try:
    from scripts.compare_receptor_screening import ranked_metrics_with_ids
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.compare_receptor_screening import ranked_metrics_with_ids

try:
    from .run_stage07c_unidock_warning_adjudication import (
        NEW_SEED_ID,
        PROFILE_ID,
        REPLAY_RECEPTOR_ID,
        validate_config,
        verify_implementation,
    )
    from .run_unidock_gpu_equivalence import (
        file_sha256,
        output_descriptor,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        write_csv,
        write_json,
    )
except ImportError:
    from run_stage07c_unidock_warning_adjudication import (
        NEW_SEED_ID,
        PROFILE_ID,
        REPLAY_RECEPTOR_ID,
        validate_config,
        verify_implementation,
    )
    from run_unidock_gpu_equivalence import (
        file_sha256,
        output_descriptor,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        write_csv,
        write_json,
    )


def finite_spearman(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Spearman inputs differ or are too short")
    value = float(spearmanr(first, second).statistic)
    if not math.isfinite(value):
        raise ValueError("Spearman correlation is not finite")
    return value


def top_fraction_overlap(
    ligand_ids: list[str],
    first: dict[str, float],
    second: dict[str, float],
    fraction: float,
) -> float:
    count = max(1, math.ceil(len(ligand_ids) * fraction))
    first_top = set(
        sorted(ligand_ids, key=lambda value: (first[value], value))[:count]
    )
    second_top = set(
        sorted(ligand_ids, key=lambda value: (second[value], value))[:count]
    )
    return len(first_top & second_top) / count


def combined_groups(
    prior_rows: list[dict[str, str]], current_rows: list[dict[str, str]]
) -> dict[tuple[str, str], dict[str, dict[str, object]]]:
    groups: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    selected = list(prior_rows) + [
        row for row in current_rows if row["run_role"] == "new_seed"
    ]
    for row in selected:
        if row["status"] != "ok" or row["pose_integrity_status"] != "ok":
            raise ValueError("Stage 07c combined evidence contains a failed row")
        key = (row["seed_id"], row["receptor_id"])
        ligand_id = row["ligand_id"]
        if ligand_id in groups[key]:
            raise ValueError(f"duplicate Stage 07c score: {key}/{ligand_id}")
        score = float(row["gpu_score"])
        if not math.isfinite(score):
            raise ValueError(f"non-finite Stage 07c score: {key}/{ligand_id}")
        groups[key][ligand_id] = {
            "label": row["label"],
            "score": score,
        }
    return groups


def group_metric_rows(
    groups: dict[tuple[str, str], dict[str, dict[str, object]]]
) -> tuple[list[dict[str, object]], dict[tuple[str, str], dict[str, object]]]:
    rows: list[dict[str, object]] = []
    metrics: dict[tuple[str, str], dict[str, object]] = {}
    for key, data in sorted(groups.items()):
        value = ranked_metrics_with_ids(data)
        metrics[key] = value
        rows.append(
            {
                "profile_id": PROFILE_ID,
                "seed_id": key[0],
                "receptor_id": key[1],
                **{
                    metric: result
                    for metric, result in value.items()
                    if metric != "top10_ligand_ids"
                },
            }
        )
    return rows, metrics


def seed_stability_rows(
    groups: dict[tuple[str, str], dict[str, dict[str, object]]],
    metrics: dict[tuple[str, str], dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seed_ids = sorted({key[0] for key in groups})
    receptor_ids = sorted({key[1] for key in groups})
    for receptor_id in receptor_ids:
        for first_seed, second_seed in itertools.combinations(seed_ids, 2):
            first_key = (first_seed, receptor_id)
            second_key = (second_seed, receptor_id)
            first = groups[first_key]
            second = groups[second_key]
            if set(first) != set(second):
                raise ValueError("Stage 07c seed coverage differs")
            ligand_ids = sorted(first)
            first_scores = {
                ligand_id: float(first[ligand_id]["score"])
                for ligand_id in ligand_ids
            }
            second_scores = {
                ligand_id: float(second[ligand_id]["score"])
                for ligand_id in ligand_ids
            }
            rows.append(
                {
                    "profile_id": PROFILE_ID,
                    "receptor_id": receptor_id,
                    "first_seed_id": first_seed,
                    "second_seed_id": second_seed,
                    "includes_new_seed": NEW_SEED_ID
                    in {first_seed, second_seed},
                    "spearman": finite_spearman(
                        [first_scores[value] for value in ligand_ids],
                        [second_scores[value] for value in ligand_ids],
                    ),
                    "top5pct_overlap": top_fraction_overlap(
                        ligand_ids,
                        first_scores,
                        second_scores,
                        0.05,
                    ),
                    "absolute_bedroc_delta": abs(
                        float(metrics[first_key]["bedroc_alpha_20"])
                        - float(metrics[second_key]["bedroc_alpha_20"])
                    ),
                }
            )
    return rows


def replay_comparison_rows(
    current_rows: list[dict[str, str]], reference_rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    replay = {
        row["ligand_id"]: row
        for row in current_rows
        if row["run_role"] == "warning_replay"
    }
    reference = {row["ligand_id"]: row for row in reference_rows}
    if set(replay) != set(reference):
        raise ValueError("Stage 07c replay ligand coverage differs")
    rows: list[dict[str, object]] = []
    for ligand_id in sorted(reference):
        observed = replay[ligand_id]
        expected = reference[ligand_id]
        observed_score = float(observed["gpu_score"])
        expected_score = float(expected["gpu_score"])
        rows.append(
            {
                "ligand_id": ligand_id,
                "receptor_id": REPLAY_RECEPTOR_ID,
                "expected_score": expected_score,
                "observed_score": observed_score,
                "absolute_score_delta": abs(observed_score - expected_score),
                "score_exact_match": observed_score == expected_score,
                "expected_pose_sha256": expected["output_pose_sha256"],
                "observed_pose_sha256": observed["output_pose_sha256"],
                "pose_sha256_exact_match": observed["output_pose_sha256"]
                == expected["output_pose_sha256"],
                "pose_integrity_status": observed["pose_integrity_status"],
            }
        )
    return rows


def write_report(
    path: Path,
    checks: dict[str, bool],
    metrics: dict[str, object],
    selected: str | None,
    config: dict[str, object],
) -> None:
    lines = [
        "# Stage 07c MAPK14 Uni-Dock Warning Adjudication",
        "",
        "## Boundary",
        "",
        "This confirmation uses consumed Train-160 rows only. Fresh validation",
        "and the locked test remain unread. Historical CPU Vina evidence remains",
        "separate from the Uni-Dock evidence stream.",
        "",
        "## Four-seed stability",
        "",
        f"- Minimum Spearman: {float(metrics['minimum_seed_pair_spearman']):.4f}",
        f"- Median Top 5% overlap: {float(metrics['median_seed_pair_top5pct_overlap']):.4f}",
        f"- Maximum BEDROC delta: {float(metrics['maximum_seed_pair_bedroc_delta']):.4f}",
        "",
        "## Warning replay",
        "",
        f"- Exact score matches: {metrics['replay_exact_score_count']}/160",
        f"- Exact pose hash matches: {metrics['replay_exact_pose_hash_count']}/160",
        f"- Known warning events: {metrics['known_warning_event_count']}",
        f"- Unresolved warning events: {metrics['unresolved_warning_event_count']}",
        f"- Pose integrity failures: {metrics['pose_integrity_failure_count']}",
        "",
        "## Gate",
        "",
    ]
    for key, passed in checks.items():
        lines.append(f"- {key}: {str(passed).lower()}")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Selected profile: `{selected if selected is not None else 'none'}`.",
            "",
            "A pass freezes only the Uni-Dock development protocol for a",
            "separately preregistered training matrix. It does not authorize",
            "validation/test access or establish QUBO or quantum advantage.",
            "",
            f"Authorization: `{config['experiment_id']}`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    validate_config(config)
    verify_implementation(config, "evaluator", Path(__file__))
    outputs = dict(config["outputs"])
    result_path = rooted_path(root, str(outputs["evaluation_result_json"]))
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {result_path}")

    scores_path = rooted_path(root, str(outputs["scores_csv"]))
    batches_path = rooted_path(root, str(outputs["batch_runs_csv"]))
    run_summary_path = rooted_path(root, str(outputs["run_summary_json"]))
    prior_path = rooted_path(root, str(config["inputs"]["prior_scores"]["path"]))
    replay_reference_path = rooted_path(
        root, str(config["inputs"]["replay_reference"]["path"])
    )
    current_rows = read_csv(scores_path)
    batch_rows = read_csv(batches_path)
    prior_rows = read_csv(prior_path)
    replay_reference_rows = read_csv(replay_reference_path)
    run_summary = read_json(run_summary_path)
    config_sha256 = file_sha256(config_path)
    if run_summary["config"]["sha256"] != config_sha256:
        raise ValueError("Stage 07c run config hash differs")
    if run_summary["outputs"]["scores_csv"]["sha256"] != file_sha256(
        scores_path
    ):
        raise ValueError("Stage 07c score hash differs")
    if run_summary["outputs"]["batch_runs_csv"]["sha256"] != file_sha256(
        batches_path
    ):
        raise ValueError("Stage 07c batch hash differs")
    if int(run_summary["fresh_validation_rows_read"]) != 0:
        raise ValueError("fresh validation rows were read")
    if int(run_summary["test_rows_read"]) != 0:
        raise ValueError("test rows were read")
    if len(current_rows) != int(config["expected"]["total_gpu_pair_count"]):
        raise ValueError("Stage 07c current score count differs")
    if len(batch_rows) != int(config["expected"]["batch_count"]):
        raise ValueError("Stage 07c batch count differs")

    groups = combined_groups(prior_rows, current_rows)
    if len(groups) != 16 or any(len(group) != 160 for group in groups.values()):
        raise ValueError("Stage 07c combined four-seed coverage differs")
    metric_rows, metrics_by_group = group_metric_rows(groups)
    stability_rows = seed_stability_rows(groups, metrics_by_group)
    replay_rows = replay_comparison_rows(current_rows, replay_reference_rows)
    gate = dict(config["profile_gate"])
    minimum_rho = min(float(row["spearman"]) for row in stability_rows)
    median_overlap = statistics.median(
        float(row["top5pct_overlap"]) for row in stability_rows
    )
    maximum_bedroc_delta = max(
        float(row["absolute_bedroc_delta"]) for row in stability_rows
    )
    exact_score_count = sum(bool(row["score_exact_match"]) for row in replay_rows)
    exact_pose_count = sum(
        bool(row["pose_sha256_exact_match"]) for row in replay_rows
    )
    known_warnings = sum(
        int(row["known_warning_event_count"]) for row in batch_rows
    )
    unresolved_warnings = sum(
        int(row["unresolved_warning_event_count"]) for row in batch_rows
    )
    pose_failures = sum(
        int(row["pose_integrity_failure_count"]) for row in batch_rows
    )
    checks = {
        "complete_new_seed": sum(
            row["run_role"] == "new_seed" for row in current_rows
        )
        == int(config["expected"]["new_seed_pair_count"]),
        "complete_warning_replay": len(replay_rows)
        == int(config["expected"]["warning_replay_pair_count"]),
        "zero_unresolved_warning_events": unresolved_warnings == 0,
        "zero_pose_integrity_failures": pose_failures == 0,
        "exact_replay_scores": exact_score_count == len(replay_rows),
        "exact_replay_pose_hashes": exact_pose_count == len(replay_rows),
        "minimum_seed_pair_spearman": minimum_rho
        >= float(gate["minimum_seed_pair_spearman"]),
        "maximum_seed_pair_bedroc_delta": maximum_bedroc_delta
        <= float(gate["maximum_seed_pair_bedroc_delta"]),
    }
    passed = all(checks.values())
    selected = PROFILE_ID if passed else None
    summary_metrics = {
        "minimum_seed_pair_spearman": minimum_rho,
        "median_seed_pair_top5pct_overlap": median_overlap,
        "maximum_seed_pair_bedroc_delta": maximum_bedroc_delta,
        "replay_exact_score_count": exact_score_count,
        "replay_exact_pose_hash_count": exact_pose_count,
        "known_warning_event_count": known_warnings,
        "unresolved_warning_event_count": unresolved_warnings,
        "pose_integrity_failure_count": pose_failures,
    }

    metrics_path = rooted_path(root, str(outputs["group_metrics_csv"]))
    stability_path = rooted_path(root, str(outputs["seed_stability_csv"]))
    replay_path = rooted_path(root, str(outputs["replay_comparison_csv"]))
    report_path = rooted_path(root, str(outputs["evaluation_report_md"]))
    write_csv(metrics_path, metric_rows)
    write_csv(stability_path, stability_rows)
    write_csv(replay_path, replay_rows)
    write_report(report_path, checks, summary_metrics, selected, config)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "unidock_profile_frozen_train_only"
            if selected is not None
            else "unidock_profile_not_frozen_train_only"
        ),
        "selected_profile_id": selected,
        "config": {
            "path": relative_path(root, config_path),
            "sha256": config_sha256,
        },
        "data_boundary": {
            "source": "consumed Train-160",
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "gate_checks": checks,
        "metrics": summary_metrics,
        "run_summary": output_descriptor(root, run_summary_path),
        "outputs": {
            "group_metrics_csv": output_descriptor(root, metrics_path),
            "seed_stability_csv": output_descriptor(root, stability_path),
            "replay_comparison_csv": output_descriptor(root, replay_path),
            "evaluation_report_md": output_descriptor(root, report_path),
        },
        "interpretation_note": config["decision_boundary"],
    }
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
