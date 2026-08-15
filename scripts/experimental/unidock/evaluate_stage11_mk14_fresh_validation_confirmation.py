"""Evaluate only the MAPK14 receptor subsets frozen before Stage 11 docking."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from .run_stage11_mk14_fresh_validation_confirmation import validate_inputs
    from .run_unidock_gpu_equivalence import (
        file_sha256,
        output_descriptor,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        verified_path,
        write_json,
    )
    from scripts.evaluate_virtual_screening import bedroc
    from scripts.select_receptor_baselines import metrics_for_subset
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.evaluate_virtual_screening import bedroc
    from scripts.select_receptor_baselines import metrics_for_subset
    from scripts.experimental.unidock.run_stage11_mk14_fresh_validation_confirmation import (
        validate_inputs,
    )
    from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
        file_sha256,
        output_descriptor,
        read_csv,
        read_json,
        relative_path,
        rooted_path,
        verified_path,
        write_json,
    )


SEED_IDS = ("seed0", "seed1", "seed2")
METRIC_KEYS = (
    "ligand_count",
    "active_count",
    "roc_auc",
    "pr_auc_average_precision",
    "bedroc_alpha_20",
    "EF1%",
    "EF5%",
    "EF10%",
    "top10_active_count",
)


def verify_implementation(
    root: Path, config: dict[str, object], key: str, expected: Path
) -> None:
    descriptor = dict(config["implementation"])[key]
    path = rooted_path(root, str(descriptor["path"]))
    if path.resolve() != expected.resolve():
        raise ValueError(f"Stage 11 evaluation implementation path differs: {key}")
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage 11 evaluation implementation hash differs: {key}")


def compact_metrics(value: dict[str, object]) -> dict[str, object]:
    return {key: value[key] for key in METRIC_KEYS}


def seed_matrices(
    score_rows: list[dict[str, str]],
    ligands: list[dict[str, str]],
    receptor_ids: list[str],
) -> dict[str, list[dict[str, str]]]:
    values: dict[tuple[str, str, str], str] = {}
    for row in score_rows:
        key = (row["seed_id"], row["ligand_id"], row["receptor_id"])
        if key in values:
            raise ValueError(f"Stage 11 evaluation duplicate score key: {key}")
        values[key] = row["gpu_score"]
    expected = len(SEED_IDS) * len(ligands) * len(receptor_ids)
    if len(values) != expected:
        raise ValueError("Stage 11 evaluation score grid is incomplete")
    output: dict[str, list[dict[str, str]]] = {}
    for seed_id in SEED_IDS:
        matrix = []
        for ligand in ligands:
            ligand_id = ligand["ligand_id"]
            matrix.append(
                {
                    "ligand_id": ligand_id,
                    "label": ligand["label"],
                    "selection_role": ligand["selection_role"],
                    **{
                        receptor_id: values[(seed_id, ligand_id, receptor_id)]
                        for receptor_id in receptor_ids
                    },
                }
            )
        output[seed_id] = matrix
    return output


def subset_score_records(
    rows: list[dict[str, str]], subset: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    return {
        row["ligand_id"]: {
            "label": row["label"],
            "score": min(float(row[receptor_id]) for receptor_id in subset),
        }
        for row in rows
    }


def quantile(values: list[float], probability: float) -> float:
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
    ranked_values: list[tuple[float, int, str, int]] = []
    for draw_index, group_id in enumerate(sampled_groups):
        for ligand_id in grouped_ids[group_id]:
            record = records[ligand_id]
            ranked_values.append(
                (
                    float(record["score"]),
                    draw_index,
                    ligand_id,
                    int(record["label"] == "active"),
                )
            )
    ranked_values.sort(key=lambda value: (value[0], value[1], value[2]))
    return float(
        bedroc(
            [{"binary_label": binary_label} for *_, binary_label in ranked_values],
            20.0,
        )
    )


def paired_bootstrap(
    records_by_candidate: dict[str, dict[str, dict[str, object]]],
    group_by_ligand: dict[str, str],
    candidate_id: str,
    comparator_ids: list[str],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    ligand_ids = set(records_by_candidate[candidate_id])
    if any(set(records_by_candidate[key]) != ligand_ids for key in comparator_ids):
        raise ValueError("Stage 11 bootstrap candidate ligand grids differ")
    if set(group_by_ligand) != ligand_ids:
        raise ValueError("Stage 11 bootstrap group map differs")
    grouped_ids: dict[str, list[str]] = defaultdict(list)
    for ligand_id, group_id in group_by_ligand.items():
        grouped_ids[group_id].append(ligand_id)
    for values in grouped_ids.values():
        values.sort()
    group_ids = sorted(grouped_ids)
    rng = random.Random(seed)
    deltas = {comparator: [] for comparator in comparator_ids}
    valid = 0
    attempts = 0
    while valid < replicates:
        attempts += 1
        if attempts > replicates * 2:
            raise ValueError("too many Stage 11 bootstrap samples lacked both labels")
        sampled = rng.choices(group_ids, k=len(group_ids))
        candidate_value = sampled_bedroc(
            records_by_candidate[candidate_id], grouped_ids, sampled
        )
        if not math.isfinite(candidate_value):
            continue
        comparator_values = {
            comparator: sampled_bedroc(
                records_by_candidate[comparator], grouped_ids, sampled
            )
            for comparator in comparator_ids
        }
        if any(not math.isfinite(value) for value in comparator_values.values()):
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
                "lower_95pct": quantile(values, 0.025),
                "upper_95pct": quantile(values, 0.975),
            }
            for comparator, values in deltas.items()
        },
    }


def robust_summary(
    metrics: dict[str, dict[str, object]]
) -> dict[str, object]:
    seed_bedroc = [float(metrics[seed]["bedroc_alpha_20"]) for seed in SEED_IDS]
    return {
        "primary": metrics["primary"],
        "sensitivity": metrics["sensitivity"],
        "seed_metrics": {seed: metrics[seed] for seed in SEED_IDS},
        "mean_seed_bedroc": statistics.fmean(seed_bedroc),
        "worst_seed_bedroc": min(seed_bedroc),
    }


def write_report(path: Path, result: dict[str, object]) -> None:
    lines = [
        "# Stage 11 MAPK14 Fresh-Validation Confirmation",
        "",
        f"Status: `{result['status']}`",
        "",
        "| Candidate | Primary BEDROC | Mean-seed BEDROC | Worst-seed BEDROC |",
        "|---|---:|---:|---:|",
    ]
    for candidate_id, value in dict(result["candidate_metrics"]).items():
        summary = dict(value)
        primary = dict(summary["primary"])
        lines.append(
            f"| {candidate_id} | {float(primary['bedroc_alpha_20']):.6f} | "
            f"{float(summary['mean_seed_bedroc']):.6f} | "
            f"{float(summary['worst_seed_bedroc']):.6f} |"
        )
    lines.extend(
        [
            "",
            "The gate compares only candidates frozen before Stage 11 scores existed.",
            "A pass supports a QUBO-formulated global-search application, not quantum advantage.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    verify_implementation(root, config, "evaluator", Path(__file__))
    verify_implementation(
        root,
        config,
        "runner",
        Path(__file__).with_name("run_stage11_mk14_fresh_validation_confirmation.py"),
    )
    verify_implementation(
        root, config, "metric_helper", root / "scripts/select_receptor_baselines.py"
    )
    verify_implementation(
        root, config, "bedroc_helper", root / "scripts/evaluate_virtual_screening.py"
    )
    outputs = dict(config["outputs"])
    audit_path = rooted_path(root, str(outputs["audit_json"]))
    audit = read_json(audit_path)
    if audit.get("status") != "independent_stage11_fresh_validation_unidock_matrix_audit_ok":
        raise ValueError("Stage 11 independent matrix audit did not pass")
    if str(audit["config"]["sha256"]).upper() != file_sha256(config_path):
        raise ValueError("Stage 11 audit config identity differs")

    receptors, ligands, _ = validate_inputs(root, config_path, config)
    receptor_ids = [row["conformer_id"] for row in receptors]
    summary = read_json(rooted_path(root, str(outputs["summary_json"])))
    source_outputs = dict(summary["outputs"])
    primary_rows = read_csv(
        verified_path(root, dict(source_outputs["median_matrix_csv"]))
    )
    sensitivity_rows = read_csv(
        verified_path(root, dict(source_outputs["minimum_matrix_csv"]))
    )
    score_rows = read_csv(verified_path(root, dict(source_outputs["scores_csv"])))
    matrices = {
        "primary": primary_rows,
        "sensitivity": sensitivity_rows,
        **seed_matrices(score_rows, ligands, receptor_ids),
    }

    candidates = {
        candidate_id: tuple(str(value) for value in dict(value)["subset"])
        for candidate_id, value in dict(config["candidates"]).items()
    }
    for candidate_id, subset in candidates.items():
        if len(subset) != 3 or not set(subset).issubset(receptor_ids):
            raise ValueError(f"Stage 11 candidate differs: {candidate_id}")
    candidate_metrics: dict[str, dict[str, object]] = {}
    records: dict[str, dict[str, dict[str, object]]] = {}
    for candidate_id, subset in candidates.items():
        metrics = {
            matrix_id: compact_metrics(
                metrics_for_subset(rows, subset, "min_score")
            )
            for matrix_id, rows in matrices.items()
        }
        candidate_metrics[candidate_id] = robust_summary(metrics)
        records[candidate_id] = subset_score_records(primary_rows, subset)

    single_receptor_metrics = {
        receptor_id: compact_metrics(
            metrics_for_subset(primary_rows, (receptor_id,), "min_score")
        )
        for receptor_id in receptor_ids
    }
    evaluation = dict(config["evaluation"])
    primary_id = str(evaluation["confirmatory_candidate"])
    comparator_ids = [str(value) for value in evaluation["confirmatory_controls"]]
    bootstrap_config = dict(evaluation["paired_bootstrap"])
    bootstrap = paired_bootstrap(
        records,
        {row["ligand_id"]: row["split_group_id"] for row in ligands},
        primary_id,
        comparator_ids,
        int(bootstrap_config["replicates"]),
        int(bootstrap_config["seed"]),
    )

    candidate_value = candidate_metrics[primary_id]
    comparisons: dict[str, dict[str, object]] = {}
    for comparator in comparator_ids:
        comparator_value = candidate_metrics[comparator]
        primary_delta = float(candidate_value["primary"]["bedroc_alpha_20"]) - float(
            comparator_value["primary"]["bedroc_alpha_20"]
        )
        mean_delta = float(candidate_value["mean_seed_bedroc"]) - float(
            comparator_value["mean_seed_bedroc"]
        )
        worst_delta = float(candidate_value["worst_seed_bedroc"]) - float(
            comparator_value["worst_seed_bedroc"]
        )
        lower = float(bootstrap["deltas"][comparator]["lower_95pct"])
        comparisons[comparator] = {
            "primary_bedroc_delta": primary_delta,
            "mean_seed_bedroc_delta": mean_delta,
            "worst_seed_bedroc_delta": worst_delta,
            "bootstrap_lower_95pct_primary_bedroc_delta": lower,
            "passed": primary_delta > 0.0
            and mean_delta > 0.0
            and worst_delta > 0.0
            and lower > 0.0,
        }
    gate_passed = all(value["passed"] for value in comparisons.values())
    result_path = rooted_path(root, str(outputs["evaluation_json"]))
    report_path = rooted_path(root, str(outputs["evaluation_report"]))
    if not overwrite and (result_path.exists() or report_path.exists()):
        raise FileExistsError("Stage 11 evaluation outputs exist; pass --overwrite")
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": (
            "stage11_frozen_candidate_gate_passed"
            if gate_passed
            else "stage11_frozen_candidate_gate_not_passed"
        ),
        "config": {
            "path": relative_path(root, config_path),
            "sha256": file_sha256(config_path),
        },
        "source_audit": output_descriptor(root, audit_path),
        "candidate_metrics": candidate_metrics,
        "single_receptor_primary_metrics": single_receptor_metrics,
        "confirmatory_comparisons": comparisons,
        "paired_bootstrap": bootstrap,
        "gate_passed": gate_passed,
        "data_boundary": {
            "validation_rows_read": len(ligands),
            "train_score_rows_read": 0,
            "test_rows_read": 0,
        },
        "interpretation_note": (
            "A pass supports external generalization of one frozen, exploratory "
            "QUBO-formulated global-search candidate. Because the candidate also "
            "matched a 560-subset classical exhaustive optimum during selection, "
            "this result cannot establish QUBO-specific or quantum advantage."
        ),
    }
    write_json(result_path, result)
    write_report(report_path, result)
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
