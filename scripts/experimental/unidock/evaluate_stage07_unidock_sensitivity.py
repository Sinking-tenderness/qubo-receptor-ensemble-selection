"""Evaluate Stage 07 Uni-Dock profiles on consumed Train-160 rows only."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

try:
    from scripts.compare_receptor_screening import ranked_metrics_with_ids
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.compare_receptor_screening import ranked_metrics_with_ids

try:
    from .run_stage07_unidock_sensitivity import (
        PROFILE_ORDER,
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
    from run_stage07_unidock_sensitivity import (
        PROFILE_ORDER,
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


def rows_by_group(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str], dict[str, dict[str, object]]]:
    groups: dict[
        tuple[str, str, str], dict[str, dict[str, object]]
    ] = defaultdict(dict)
    for row in rows:
        key = (row["profile_id"], row["seed_id"], row["receptor_id"])
        ligand_id = row["ligand_id"]
        if ligand_id in groups[key]:
            raise ValueError(f"duplicate sensitivity score: {key}/{ligand_id}")
        if row["status"] != "ok":
            raise ValueError(f"failed sensitivity score: {key}/{ligand_id}")
        score = float(row["gpu_score"])
        if not math.isfinite(score):
            raise ValueError(f"non-finite sensitivity score: {key}/{ligand_id}")
        groups[key][ligand_id] = {
            "label": row["label"],
            "score": score,
        }
    return groups


def group_metric_rows(
    groups: dict[tuple[str, str, str], dict[str, dict[str, object]]]
) -> tuple[list[dict[str, object]], dict[tuple[str, str, str], dict[str, object]]]:
    metrics_by_group: dict[tuple[str, str, str], dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for key, data in sorted(groups.items()):
        metrics = ranked_metrics_with_ids(data)
        metrics_by_group[key] = metrics
        rows.append(
            {
                "profile_id": key[0],
                "seed_id": key[1],
                "receptor_id": key[2],
                **{
                    metric: value
                    for metric, value in metrics.items()
                    if metric != "top10_ligand_ids"
                },
            }
        )
    return rows, metrics_by_group


def profile_comparisons(
    groups: dict[tuple[str, str, str], dict[str, dict[str, object]]],
    metrics: dict[tuple[str, str, str], dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    detail_keys = [key for key in groups if key[0] == "detail"]
    for _, seed_id, receptor_id in sorted(detail_keys):
        reference_key = ("detail", seed_id, receptor_id)
        reference = groups[reference_key]
        ligand_ids = sorted(reference)
        for profile_id in PROFILE_ORDER:
            key = (profile_id, seed_id, receptor_id)
            candidate = groups.get(key)
            if candidate is None or set(candidate) != set(reference):
                raise ValueError(f"profile/detail coverage differs: {key}")
            if any(
                candidate[ligand_id]["label"]
                != reference[ligand_id]["label"]
                for ligand_id in ligand_ids
            ):
                raise ValueError(f"profile/detail labels differ: {key}")
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
                    "reference_profile_id": "detail",
                    "seed_id": seed_id,
                    "receptor_id": receptor_id,
                    "ligand_count": len(ligand_ids),
                    "spearman_vs_detail": finite_spearman(
                        [candidate_scores[value] for value in ligand_ids],
                        [reference_scores[value] for value in ligand_ids],
                    ),
                    "top5pct_overlap_vs_detail": top_fraction_overlap(
                        ligand_ids,
                        candidate_scores,
                        reference_scores,
                        0.05,
                    ),
                    "median_absolute_score_delta_vs_detail": statistics.median(
                        abs(value) for value in deltas
                    ),
                    "p95_absolute_score_delta_vs_detail": sorted(
                        abs(value) for value in deltas
                    )[math.ceil(0.95 * len(deltas)) - 1],
                    "bedroc_alpha_20": metrics[key]["bedroc_alpha_20"],
                    "detail_bedroc_alpha_20": metrics[reference_key][
                        "bedroc_alpha_20"
                    ],
                    "bedroc_delta_vs_detail": float(
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
    detail_elapsed = sum(
        float(row["elapsed_seconds"])
        for row in batch_rows
        if row["profile_id"] == "detail"
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
            "minimum_group_spearman_vs_detail": min(
                float(row["spearman_vs_detail"]) for row in comparisons
            )
            >= float(gate["minimum_group_spearman_vs_detail"]),
            "median_group_top5pct_overlap_vs_detail": statistics.median(
                float(row["top5pct_overlap_vs_detail"])
                for row in comparisons
            )
            >= float(gate["minimum_median_group_top5pct_overlap_vs_detail"]),
            "maximum_absolute_group_bedroc_delta_vs_detail": max(
                abs(float(row["bedroc_delta_vs_detail"]))
                for row in comparisons
            )
            <= float(gate["maximum_absolute_group_bedroc_delta_vs_detail"]),
            "minimum_seed_pair_spearman": min(
                float(row["spearman"]) for row in stability
            )
            >= float(gate["minimum_seed_pair_spearman"]),
            "maximum_seed_pair_bedroc_delta": max(
                float(row["absolute_bedroc_delta"]) for row in stability
            )
            <= float(gate["maximum_seed_pair_bedroc_delta"]),
        }
        summaries.append(
            {
                "profile_id": profile_id,
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
                "speedup_vs_detail": detail_elapsed / elapsed,
                "minimum_group_spearman_vs_detail": min(
                    float(row["spearman_vs_detail"])
                    for row in comparisons
                ),
                "median_group_spearman_vs_detail": statistics.median(
                    float(row["spearman_vs_detail"])
                    for row in comparisons
                ),
                "median_group_top5pct_overlap_vs_detail": statistics.median(
                    float(row["top5pct_overlap_vs_detail"])
                    for row in comparisons
                ),
                "maximum_absolute_group_bedroc_delta_vs_detail": max(
                    abs(float(row["bedroc_delta_vs_detail"]))
                    for row in comparisons
                ),
                "minimum_seed_pair_spearman": min(
                    float(row["spearman"]) for row in stability
                ),
                "maximum_seed_pair_bedroc_delta": max(
                    float(row["absolute_bedroc_delta"]) for row in stability
                ),
                "engine_warning_count": sum(
                    int(row["engine_warning_count"]) for row in batches
                ),
                "gate_checks": checks,
                "all_gate_checks_passed": all(checks.values()),
            }
        )
    return summaries


def write_report(
    path: Path,
    summaries: list[dict[str, object]],
    selected: str | None,
    config: dict[str, object],
) -> None:
    lines = [
        "# Stage 07 MAPK14 Uni-Dock Search Sensitivity",
        "",
        "## Boundary",
        "",
        "This profile selection uses consumed Train-160 rows only. Fresh validation",
        "and the locked test remain unread. Uni-Dock is treated as a new docking",
        "engine; no CPU Vina score-equivalence claim is made.",
        "",
        "## Profile gate",
        "",
        "| Profile | Exhaustiveness | Max step | Pairs/s | Min rho vs detail | Median Top5% | Max BEDROC delta | Min seed rho | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['profile_id']} | {row['exhaustiveness']} | "
            f"{row['max_step']} | {float(row['pairs_per_second']):.3f} | "
            f"{float(row['minimum_group_spearman_vs_detail']):.3f} | "
            f"{float(row['median_group_top5pct_overlap_vs_detail']):.3f} | "
            f"{float(row['maximum_absolute_group_bedroc_delta_vs_detail']):.4f} | "
            f"{float(row['minimum_seed_pair_spearman']):.3f} | "
            f"{str(bool(row['all_gate_checks_passed'])).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Selected profile: `{selected if selected is not None else 'none'}`.",
            "",
            "The first passing profile in fast, balance, detail order is selected.",
            "Selection freezes only the Uni-Dock development protocol. It does not",
            "authorize validation/test access or establish QUBO or quantum advantage.",
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
    scores_path = rooted_path(root, str(outputs["scores_csv"]))
    batch_path = rooted_path(root, str(outputs["batch_runs_csv"]))
    run_summary_path = rooted_path(root, str(outputs["run_summary_json"]))
    for path in (scores_path, batch_path, run_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_summary = read_json(run_summary_path)
    if run_summary.get("status") != "ok":
        raise ValueError("Uni-Dock sensitivity run is not complete")
    if run_summary["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("run/config SHA-256 differs")
    if run_summary["outputs"]["scores_csv"]["sha256"] != file_sha256(
        scores_path
    ):
        raise ValueError("run score-table SHA-256 differs")
    if run_summary["outputs"]["batch_runs_csv"]["sha256"] != file_sha256(
        batch_path
    ):
        raise ValueError("run batch-table SHA-256 differs")

    result_path = rooted_path(root, str(outputs["evaluation_result_json"]))
    report_path = rooted_path(root, str(outputs["evaluation_report_md"]))
    metric_path = rooted_path(root, str(outputs["group_metrics_csv"]))
    comparison_path = rooted_path(root, str(outputs["profile_comparisons_csv"]))
    stability_path = rooted_path(root, str(outputs["seed_stability_csv"]))
    profile_path = rooted_path(root, str(outputs["profile_summary_csv"]))
    generated = (
        result_path,
        report_path,
        metric_path,
        comparison_path,
        stability_path,
        profile_path,
    )
    if any(path.exists() for path in generated) and not overwrite:
        raise FileExistsError("sensitivity evaluation outputs exist; use --overwrite")

    scores = read_csv(scores_path)
    batches = read_csv(batch_path)
    if len(scores) != int(config["expected"]["total_pair_count"]):
        raise ValueError("sensitivity score count differs")
    groups = rows_by_group(scores)
    expected_groups = (
        len(PROFILE_ORDER)
        * int(config["expected"]["seed_count"])
        * int(config["expected"]["receptor_count"])
    )
    if len(groups) != expected_groups:
        raise ValueError("sensitivity group count differs")
    if any(
        len(rows) != int(config["expected"]["ligand_count"])
        for rows in groups.values()
    ):
        raise ValueError("a sensitivity group is incomplete")

    metric_rows, metrics = group_metric_rows(groups)
    comparison_rows = profile_comparisons(groups, metrics)
    stability_rows = seed_stability_rows(groups, metrics)
    summary_rows = profile_summary_rows(
        config, comparison_rows, stability_rows, batches
    )
    selected = next(
        (
            str(row["profile_id"])
            for row in summary_rows
            if bool(row["all_gate_checks_passed"])
        ),
        None,
    )
    write_csv(metric_path, metric_rows)
    write_csv(comparison_path, comparison_rows)
    write_csv(stability_path, stability_rows)
    write_csv(
        profile_path,
        [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key != "gate_checks"
                },
                **{
                    f"check_{key}": value
                    for key, value in dict(row["gate_checks"]).items()
                },
            }
            for row in summary_rows
        ],
    )
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
        "selection_order": list(PROFILE_ORDER),
        "profile_summaries": summary_rows,
        "data_boundary": {
            "source": "consumed Train-160",
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "evidence_class": "development hyperparameter sensitivity",
        },
        "config": {
            "path": relative_path(root, config_path),
            "sha256": file_sha256(config_path),
        },
        "run_summary": output_descriptor(root, run_summary_path),
        "outputs": {
            "group_metrics_csv": output_descriptor(root, metric_path),
            "profile_comparisons_csv": output_descriptor(
                root, comparison_path
            ),
            "seed_stability_csv": output_descriptor(root, stability_path),
            "profile_summary_csv": output_descriptor(root, profile_path),
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
