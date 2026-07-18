"""Audit label-blind e32 diagnostics for e16 matrix admission rescue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


VINA_RESULT_PATTERN = re.compile(
    r"^REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", re.MULTILINE
)
LOG_MODE_ONE_PATTERN = re.compile(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", re.MULTILINE)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def single_score(pattern: re.Pattern[str], text: str, source: Path) -> float:
    matches = [float(match.group(1)) for match in pattern.finditer(text)]
    if len(matches) != 1:
        raise ValueError(f"expected one top score in {source}, got {len(matches)}")
    return matches[0]


def evaluate_rescue(
    e16_scores: list[float], e32_scores: list[float], threshold: float
) -> dict[str, object]:
    if len(e16_scores) < 2 or len(e16_scores) != len(e32_scores):
        raise ValueError("paired e16/e32 scores require equal multi-seed lengths")
    if not all(math.isfinite(value) for value in [*e16_scores, *e32_scores]):
        raise ValueError("paired scores contain a non-finite value")
    e16_median = statistics.median(e16_scores)
    e16_minimum = min(e16_scores)
    e32_median = statistics.median(e32_scores)
    e32_minimum = min(e32_scores)
    e32_range = max(e32_scores) - min(e32_scores)
    median_delta = abs(e16_median - e32_median)
    minimum_delta = abs(e16_minimum - e32_minimum)
    checks = {
        "e32_seed_range_within_threshold": e32_range <= threshold,
        "e16_e32_median_delta_within_threshold": median_delta <= threshold,
        "e16_e32_minimum_delta_within_threshold": minimum_delta <= threshold,
    }
    return {
        "e16_scores": e16_scores,
        "e16_seed_range_kcal_per_mol": max(e16_scores) - min(e16_scores),
        "e16_median_score": e16_median,
        "e16_minimum_score": e16_minimum,
        "e32_scores": e32_scores,
        "e32_seed_range_kcal_per_mol": e32_range,
        "e32_median_score": e32_median,
        "e32_minimum_score": e32_minimum,
        "absolute_e16_e32_median_delta_kcal_per_mol": median_delta,
        "absolute_e16_e32_minimum_delta_kcal_per_mol": minimum_delta,
        "acceptance_threshold_kcal_per_mol": threshold,
        "checks": checks,
        "rescue_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="ascii"))
    aggregated_path = Path(config["inputs"]["aggregated_seed_scores_csv"])
    aggregate_summary_path = Path(config["inputs"]["aggregate_summary_json"])
    diagnostic_dir = Path(config["inputs"]["e32_diagnostic_directory"])
    e32_config_path = diagnostic_dir / "e32_cpu2.txt"
    for key, path in (
        ("aggregated_seed_scores_csv", aggregated_path),
        ("aggregate_summary_json", aggregate_summary_path),
        ("e32_config", e32_config_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = config["input_sha256"][key]
        if file_sha256(path) != str(expected).upper():
            raise ValueError(f"input hash differs: {key}")
    aggregate_summary = json.loads(aggregate_summary_path.read_text(encoding="ascii"))
    if aggregate_summary.get("status") != "ok":
        raise ValueError("source aggregation did not pass")
    if int(aggregate_summary.get("locked_test_manifest_rows", -1)) != 0:
        raise ValueError("source aggregation contains locked test rows")
    e32_config_text = e32_config_path.read_text(encoding="ascii")
    if "exhaustiveness = 32" not in e32_config_text:
        raise ValueError("diagnostic configuration is not e32")

    threshold_source = config["threshold_source"]
    threshold_source_path = Path(threshold_source["path"])
    if not threshold_source_path.is_file() or file_sha256(
        threshold_source_path
    ) != str(threshold_source["sha256"]).upper():
        raise ValueError("threshold source is missing or its hash differs")
    threshold_source_json = json.loads(
        threshold_source_path.read_text(encoding="ascii")
    )
    source_threshold = float(
        threshold_source_json["acceptance_thresholds"][
            "maximum_highest_protocol_seed_range_kcal_per_mol"
        ]
    )

    aggregate_rows = read_csv(aggregated_path)
    aggregate_by_pair = {
        (row["ligand_id"], row["receptor_id"]): row for row in aggregate_rows
    }
    if len(aggregate_by_pair) != len(aggregate_rows):
        raise ValueError("aggregated score table contains duplicate pairs")
    seed_records = config["seed_records"]
    seed_fields = [str(record["seed_id"]) for record in seed_records]
    base_seeds = [int(record["base_seed"]) for record in seed_records]
    threshold = float(config["acceptance"]["maximum_score_delta_kcal_per_mol"])
    if not math.isclose(
        threshold,
        float(threshold_source["value_kcal_per_mol"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        threshold, source_threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("rescue threshold differs from its frozen source")
    case_results: list[dict[str, object]] = []

    for case in config["cases"]:
        ligand_id = str(case["ligand_id"])
        receptor_id = str(case["receptor_id"])
        key = (ligand_id, receptor_id)
        if key not in aggregate_by_pair:
            raise ValueError(f"diagnostic pair is absent from aggregation: {key}")
        aggregate_row = aggregate_by_pair[key]
        e16_scores = [
            float(aggregate_row[f"{seed_id}_representative_score"])
            for seed_id in seed_fields
        ]
        e32_scores: list[float] = []
        evidence: list[dict[str, object]] = []
        for base_seed in base_seeds:
            stem = f"{ligand_id}__{receptor_id}__seed{base_seed}"
            pose_path = diagnostic_dir / "poses" / f"{stem}.pdbqt"
            log_path = diagnostic_dir / "logs" / f"{stem}.log"
            for path in (pose_path, log_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            pose_text = pose_path.read_text(encoding="ascii", errors="replace")
            log_text = log_path.read_text(encoding="ascii", errors="replace")
            if "AutoDock Vina v1.2.7" not in log_text or "Exhaustiveness: 32" not in log_text:
                raise ValueError(f"diagnostic log protocol differs: {log_path}")
            pose_score = single_score(VINA_RESULT_PATTERN, pose_text, pose_path)
            log_score = single_score(LOG_MODE_ONE_PATTERN, log_text, log_path)
            # Vina's console table uses fewer displayed digits than the PDBQT remark.
            if not math.isclose(pose_score, log_score, rel_tol=0.0, abs_tol=0.005):
                raise ValueError(f"pose/log score differs: {stem}")
            e32_scores.append(pose_score)
            evidence.append(
                {
                    "base_seed": base_seed,
                    "pose_path": pose_path.as_posix(),
                    "pose_sha256": file_sha256(pose_path),
                    "log_path": log_path.as_posix(),
                    "log_sha256": file_sha256(log_path),
                    "score_kcal_per_mol": pose_score,
                }
            )
        result = evaluate_rescue(e16_scores, e32_scores, threshold)
        case_results.append(
            {
                "case_id": str(case["case_id"]),
                "ligand_id": ligand_id,
                "receptor_id": receptor_id,
                **result,
                "e32_evidence": evidence,
            }
        )

    all_passed = all(bool(row["rescue_passed"]) for row in case_results)
    output_path = Path(config["outputs"]["summary_json"])
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; use --overwrite: {output_path}")
    output = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "matrix_admission_rescued" if all_passed else "matrix_admission_rescue_failed",
        "config": {"path": args.config.as_posix(), "sha256": file_sha256(args.config)},
        "threshold_source": config["threshold_source"],
        "source_archives": config["source_archives"],
        "case_count": len(case_results),
        "case_results": case_results,
        "all_cases_passed": all_passed,
        "original_raw_range_gate_passed": False,
        "original_matrix_cells_replaced": 0,
        "primary_matrix_authorized": all_passed,
        "sensitivity_matrix_authorized": all_passed,
        "enrichment_metrics_calculated": False,
        "test_evaluated": False,
        "interpretation_note": config["interpretation_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(output, indent=2, ensure_ascii=True))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
