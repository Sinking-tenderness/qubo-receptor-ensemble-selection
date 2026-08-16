"""Evaluate Stage 07b Uni-Dock profiles on consumed Train-160 rows."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

try:
    from .evaluate_stage07_unidock_sensitivity import (
        finite_spearman,
        group_metric_rows,
        rows_by_group,
        top_fraction_overlap,
    )
    from .run_stage07b_unidock_enhanced_confirmation import (
        CANDIDATE_PROFILES,
        PROFILE_ORDER,
        validate_config,
        verify_implementation,
    )
    from .run_unidock_gpu_equivalence import (
        file_sha256,
        output_descriptor,
        read_csv,
        read_json,
        rooted_path,
        write_csv,
        write_json,
    )
except ImportError:
    from evaluate_stage07_unidock_sensitivity import (
        finite_spearman,
        group_metric_rows,
        rows_by_group,
        top_fraction_overlap,
    )
    from run_stage07b_unidock_enhanced_confirmation import (
        CANDIDATE_PROFILES,
        PROFILE_ORDER,
        validate_config,
        verify_implementation,
    )
    from run_unidock_gpu_equivalence import (
        file_sha256,
        output_descriptor,
        read_csv,
        read_json,
        rooted_path,
        write_csv,
        write_json,
    )


def profile_comparison_rows(
    groups: dict[tuple[str, str, str], dict[str, dict[str, object]]],
    metrics: dict[tuple[str, str, str], dict[str, object]],
    reference_profile_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    reference_keys = [
        key for key in groups if key[0] == reference_profile_id
    ]
    for _, seed_id, receptor_id in sorted(reference_keys):
        reference_key = (reference_profile_id, seed_id, receptor_id)
        reference = groups[reference_key]
        ligand_ids = sorted(reference)
        for profile_id in PROFILE_ORDER:
            key = (profile_id, seed_id, receptor_id)
            candidate = groups.get(key)
            if candidate is None or set(candidate) != set(reference):
                raise ValueError(f"profile/reference coverage differs: {key}")
            if any(
                candidate[ligand_id]["label"]
                != reference[ligand_id]["label"]
                for ligand_id in ligand_ids
            ):
                raise ValueError(f"profile/reference labels differ: {key}")
            candidate_scores = {
                ligand_id: float(candidate[ligand_id]["score"])
                for ligand_id in ligand_ids
            }
            reference_scores = {
                ligand_id: float(reference[ligand_id]["score"])
                for ligand_id in ligand_ids
            }
            deltas = [
                candidate_scores[ligand_id] - reference_scores[ligand_id]
                for ligand_id in ligand_ids
            ]
            rows.append(
                {
                    "profile_id": profile_id,
                    "reference_profile_id": reference_profile_id,
                    "seed_id": seed_id,
                    "receptor_id": receptor_id,
                    "ligand_count": len(ligand_ids),
                    "spearman_vs_reference": finite_spearman(
                        [candidate_scores[value] for value in ligand_ids],
                        [reference_scores[value] for value in ligand_ids],
                    ),
                    "top5pct_overlap_vs_reference": top_fraction_overlap(
                        ligand_ids,
                        candidate_scores,
                        reference_scores,
                        0.05,
                    ),
                    "median_absolute_score_delta_vs_reference": (
                        statistics.median(abs(value) for value in deltas)
                    ),
                    "p95_absolute_score_delta_vs_reference": sorted(
                        abs(value) for value in deltas
                    )[math.ceil(0.95 * len(deltas)) - 1],
                    "bedroc_alpha_20": metrics[key]["bedroc_alpha_20"],
                    "reference_bedroc_alpha_20": metrics[reference_key][
                        "bedroc_alpha_20"
                    ],
                    "bedroc_delta_vs_reference": float(
                        metrics[key]["bedroc_alpha_20"]
                    )
                    - float(metrics[reference_key]["bedroc_alpha_20"]),
                }
            )
    return rows


def seed_stability_rows(
    groups: dict[tuple[str, str, str], dict[str, dict[str, object]]],
    metrics: dict[tuple[str, str, str], dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    receptor_ids = sorted({key[2] for key in groups})
    seed_ids = sorted({key[1] for key in groups})
    for profile_id in PROFILE_ORDER:
        for receptor_id in receptor_ids:
            for first_seed, second_seed in itertools.combinations(seed_ids, 2):
                first_key = (profile_id, first_seed, receptor_id)
                second_key = (profile_id, second_seed, receptor_id)
                first = groups[first_key]
                second = groups[second_key]
                if set(first) != set(second):
                    raise ValueError("seed score coverage differs")
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
                        "profile_id": profile_id,
                        "receptor_id": receptor_id,
                        "first_seed_id": first_seed,
                        "second_seed_id": second_seed,
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


def profile_summary_rows(
    config: dict[str, object],
    comparison_rows: list[dict[str, object]],
    stability_rows: list[dict[str, object]],
    batch_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    gate = dict(config["profile_gate"])
    reference_id = str(gate["reference_profile_id"])
    reference_elapsed = sum(
        float(row["elapsed_seconds"])
        for row in batch_rows
        if row["profile_id"] == reference_id
    )
    summaries: list[dict[str, object]] = []
    for profile_id in PROFILE_ORDER:
        comparisons = [
            row for row in comparison_rows if row["profile_id"] == profile_id
        ]
        stability = [
            row for row in stability_rows if row["profile_id"] == profile_id
        ]
        batches = [row for row in batch_rows if row["profile_id"] == profile_id]
        elapsed = sum(float(row["elapsed_seconds"]) for row in batches)
        checks = {
            "complete_batches": len(batches)
            == int(config["expected"]["batch_count_per_profile"]),
            "zero_engine_warnings": sum(
                int(row["engine_warning_count"]) for row in batches
            )
            == 0,
            "zero_pose_integrity_failures": sum(
                int(row["pose_integrity_failure_count"]) for row in batches
            )
            == 0,
            "minimum_group_spearman_vs_reference": min(
                float(row["spearman_vs_reference"])
                for row in comparisons
            )
            >= float(gate["minimum_group_spearman_vs_reference"]),
            "median_group_top5pct_overlap_vs_reference": statistics.median(
                float(row["top5pct_overlap_vs_reference"])
                for row in comparisons
            )
            >= float(
                gate["minimum_median_group_top5pct_overlap_vs_reference"]
            ),
            "maximum_absolute_group_bedroc_delta_vs_reference": max(
                abs(float(row["bedroc_delta_vs_reference"]))
                for row in comparisons
            )
            <= float(gate["maximum_absolute_group_bedroc_delta_vs_reference"]),
            "minimum_seed_pair_spearman": min(
                float(row["spearman"]) for row in stability
            )
            >= float(gate["minimum_seed_pair_spearman"]),
            "maximum_seed_pair_bedroc_delta": max(
                float(row["absolute_bedroc_delta"]) for row in stability
            )
            <= float(gate["maximum_seed_pair_bedroc_delta"]),
        }
        scientific_pass = all(checks.values())
        candidate = profile_id in CANDIDATE_PROFILES
        summaries.append(
            {
                "profile_id": profile_id,
                "candidate_profile": candidate,
                "exhaustiveness": config["profiles"][profile_id][
                    "exhaustiveness"
                ],
                "max_step": config["profiles"][profile_id]["max_step"],
                "batch_count": len(batches),
                "elapsed_seconds": elapsed,
                "pairs_per_second": int(
                    config["expected"]["pair_count_per_profile"]
                )
                / elapsed,
                "speedup_vs_reference": reference_elapsed / elapsed,
                "minimum_group_spearman_vs_reference": min(
                    float(row["spearman_vs_reference"])
                    for row in comparisons
                ),
                "median_group_spearman_vs_reference": statistics.median(
                    float(row["spearman_vs_reference"])
                    for row in comparisons
                ),
                "median_group_top5pct_overlap_vs_reference": statistics.median(
                    float(row["top5pct_overlap_vs_reference"])
                    for row in comparisons
                ),
                "maximum_absolute_group_bedroc_delta_vs_reference": max(
                    abs(float(row["bedroc_delta_vs_reference"]))
                    for row in comparisons
                ),
                "minimum_seed_pair_spearman": min(
                    float(row["spearman"]) for row in stability
                ),
                "median_seed_pair_top5pct_overlap": statistics.median(
                    float(row["top5pct_overlap"]) for row in stability
                ),
                "maximum_seed_pair_bedroc_delta": max(
                    float(row["absolute_bedroc_delta"]) for row in stability
                ),
                "engine_warning_count": sum(
                    int(row["engine_warning_count"]) for row in batches
                ),
                "pose_integrity_failure_count": sum(
                    int(row["pose_integrity_failure_count"]) for row in batches
                ),
                "gate_checks": checks,
                "scientific_gate_checks_passed": scientific_pass,
                "selection_eligible": candidate and scientific_pass,
            }
        )
    return summaries


def select_profile(summaries: list[dict[str, object]]) -> str | None:
    candidate_rank = {
        profile_id: index
        for index, profile_id in enumerate(CANDIDATE_PROFILES)
    }
    passing = [row for row in summaries if row["selection_eligible"]]
    if not passing:
        return None
    passing.sort(
        key=lambda row: (
            -float(row["pairs_per_second"]),
            candidate_rank[str(row["profile_id"])],
        )
    )
    return str(passing[0]["profile_id"])


def write_report(
    path: Path,
    summaries: list[dict[str, object]],
    selected: str | None,
    config: dict[str, object],
) -> None:
    lines = [
        "# Stage 07b MAPK14 Uni-Dock Enhanced Confirmation",
        "",
        "## Boundary",
        "",
        "This confirmation uses consumed Train-160 rows only. Fresh validation",
        "and the locked test remain unread. Historical CPU Vina scores remain",
        "separate from the Uni-Dock evidence stream.",
        "",
        "## Profile gate",
        "",
        "| Profile | Exhaustiveness | Max step | Pairs/s | Min rho vs enhanced | Median Top5% | Max BEDROC delta | Min seed rho | Max seed BEDROC delta | Warnings | Pose failures | Eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['profile_id']} | {row['exhaustiveness']} | "
            f"{row['max_step']} | {float(row['pairs_per_second']):.3f} | "
            f"{float(row['minimum_group_spearman_vs_reference']):.3f} | "
            f"{float(row['median_group_top5pct_overlap_vs_reference']):.3f} | "
            f"{float(row['maximum_absolute_group_bedroc_delta_vs_reference']):.4f} | "
            f"{float(row['minimum_seed_pair_spearman']):.3f} | "
            f"{float(row['maximum_seed_pair_bedroc_delta']):.4f} | "
            f"{row['engine_warning_count']} | "
            f"{row['pose_integrity_failure_count']} | "
            f"{str(bool(row['selection_eligible'])).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Selected profile: `{selected if selected is not None else 'none'}`.",
            "",
            "The fastest candidate passing every preregistered check is selected.",
            "The detail recheck is diagnostic and cannot be selected.",
            "Selection freezes only the Uni-Dock development protocol.",
            "It does not authorize validation/test access or establish QUBO or",
            "quantum advantage.",
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
    verify_implementation(
        config,
        "metric_helper",
        Path(__file__).with_name("evaluate_stage07_unidock_sensitivity.py"),
    )
    outputs = dict(config["outputs"])
    result_path = rooted_path(root, str(outputs["evaluation_result_json"]))
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {result_path}")

    scores_path = rooted_path(root, str(outputs["scores_csv"]))
    batches_path = rooted_path(root, str(outputs["batch_runs_csv"]))
    run_summary_path = rooted_path(root, str(outputs["run_summary_json"]))
    scores = read_csv(scores_path)
    batches = read_csv(batches_path)
    run_summary = read_json(run_summary_path)
    config_sha256 = file_sha256(config_path)
    if run_summary["config"]["sha256"] != config_sha256:
        raise ValueError("run summary config hash differs")
    if run_summary["outputs"]["scores_csv"]["sha256"] != file_sha256(
        scores_path
    ):
        raise ValueError("run summary score hash differs")
    if run_summary["outputs"]["batch_runs_csv"]["sha256"] != file_sha256(
        batches_path
    ):
        raise ValueError("run summary batch hash differs")
    if int(run_summary["fresh_validation_rows_read"]) != 0:
        raise ValueError("fresh validation rows were read")
    if int(run_summary["test_rows_read"]) != 0:
        raise ValueError("test rows were read")
    if len(scores) != int(config["expected"]["total_pair_count"]):
        raise ValueError("Stage 07b score row count differs")
    if len(batches) != int(config["expected"]["total_batch_count"]):
        raise ValueError("Stage 07b batch row count differs")
    if any(row.get("pose_integrity_status") != "ok" for row in scores):
        if sum(
            int(row["pose_integrity_failure_count"]) for row in batches
        ) == 0:
            raise ValueError("pose integrity rows and batch audit disagree")

    groups = rows_by_group(scores)
    expected_group_count = (
        len(PROFILE_ORDER)
        * int(config["expected"]["seed_count"])
        * int(config["expected"]["receptor_count"])
    )
    if len(groups) != expected_group_count:
        raise ValueError("Stage 07b metric group count differs")
    if any(
        len(group) != int(config["expected"]["ligand_count"])
        for group in groups.values()
    ):
        raise ValueError("Stage 07b metric group coverage differs")

    metric_rows, metrics = group_metric_rows(groups)
    reference_id = str(config["profile_gate"]["reference_profile_id"])
    comparison_rows = profile_comparison_rows(groups, metrics, reference_id)
    stability_rows = seed_stability_rows(groups, metrics)
    summary_rows = profile_summary_rows(
        config, comparison_rows, stability_rows, batches
    )
    selected = select_profile(summary_rows)

    group_metrics_path = rooted_path(root, str(outputs["group_metrics_csv"]))
    comparison_path = rooted_path(
        root, str(outputs["profile_comparisons_csv"])
    )
    stability_path = rooted_path(root, str(outputs["seed_stability_csv"]))
    profile_summary_path = rooted_path(
        root, str(outputs["profile_summary_csv"])
    )
    report_path = rooted_path(root, str(outputs["evaluation_report_md"]))
    write_csv(group_metrics_path, metric_rows)
    write_csv(comparison_path, comparison_rows)
    write_csv(stability_path, stability_rows)
    flat_summaries = []
    for row in summary_rows:
        flat = {
            key: value for key, value in row.items() if key != "gate_checks"
        }
        flat.update(
            {
                f"check_{key}": value
                for key, value in row["gate_checks"].items()
            }
        )
        flat_summaries.append(flat)
    write_csv(profile_summary_path, flat_summaries)
    write_report(report_path, summary_rows, selected, config)

    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "unidock_profile_selected_train_only"
            if selected is not None
            else "no_unidock_profile_passed_train_only"
        ),
        "selected_profile_id": selected,
        "candidate_profile_ids": list(CANDIDATE_PROFILES),
        "reference_profile_id": reference_id,
        "selection_rule": config["profile_gate"]["selection_rule"],
        "config": {
            "path": str(config_path.relative_to(root)).replace("\\", "/"),
            "sha256": config_sha256,
        },
        "data_boundary": {
            "source": "consumed Train-160",
            "evidence_class": config["data_boundary"]["evidence_class"],
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "profile_summaries": summary_rows,
        "run_summary": output_descriptor(root, run_summary_path),
        "outputs": {
            "group_metrics_csv": output_descriptor(root, group_metrics_path),
            "profile_comparisons_csv": output_descriptor(
                root, comparison_path
            ),
            "seed_stability_csv": output_descriptor(root, stability_path),
            "profile_summary_csv": output_descriptor(
                root, profile_summary_path
            ),
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
