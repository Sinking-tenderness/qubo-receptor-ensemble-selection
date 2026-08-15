"""Adjudicate the completed Stage 14d FA10 redocking failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tarfile
from pathlib import Path

from scripts.adjudicate_stage13f_egfr_cognate_redocking_failure import (
    archive_bytes,
    archive_csv,
    archive_json,
    file_sha256,
    read_json,
    truth,
    verify_archive,
)


SUMMARY_MEMBER = "data/stage14d_fa10_cognate_redocking_summary.json"
GATE_MEMBER = "data/processed/stage14d_fa10_receptor_gate_results.csv"
PAIR_MEMBER = "data/processed/stage14d_fa10_cognate_redocking_results.csv"


def failure_diagnostics(
    pair_rows: list[dict[str, str]], failed_ids: list[str], threshold: float
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for receptor_id in failed_ids:
        selected = [row for row in pair_rows if row["conformer_id"] == receptor_id]
        rmsds = [float(row["top_ranked_rmsd_angstrom"]) for row in selected]
        affinities = [float(row["top_ranked_affinity_kcal_per_mol"]) for row in selected]
        near_threshold = min(rmsds) > threshold and max(rmsds) <= threshold + 0.25
        output.append(
            {
                "conformer_id": receptor_id,
                "seed_count": len(selected),
                "successful_seed_count": sum(
                    truth(row["top_ranked_pose_success"]) for row in selected
                ),
                "minimum_rmsd_angstrom": min(rmsds),
                "median_rmsd_angstrom": statistics.median(rmsds),
                "maximum_rmsd_angstrom": max(rmsds),
                "rmsd_span_angstrom": max(rmsds) - min(rmsds),
                "mean_affinity_kcal_per_mol": statistics.mean(affinities),
                "failure_class": (
                    "three-seed stable near-threshold pose mismatch"
                    if near_threshold
                    else "three-seed stable alternative pose"
                ),
                "unique_cause_established": False,
            }
        )
    return output


def render_report(result: dict[str, object]) -> str:
    lines = [
        "# Stage 14e FA10 Cognate-Redocking Failure Adjudication",
        "",
        "## Decision",
        "",
        "Stage 14d completed all 48 planned Uni-Dock jobs with zero unresolved warnings",
        "and zero pose-integrity failures, but only 13 of 16 receptors passed the frozen",
        "three-seed RMSD gate. The FA10 confirmatory technical gate is therefore closed.",
        "",
        "| Receptor | Median RMSD (A) | Range (A) | Seeds passing | Failure class |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in result["failed_receptor_diagnostics"]:
        lines.append(
            "| {conformer_id} | {median_rmsd_angstrom:.3f} | "
            "{minimum_rmsd_angstrom:.3f}-{maximum_rmsd_angstrom:.3f} | "
            "{successful_seed_count}/3 | {failure_class} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The failures are reproducible across seeds and are not runtime or pose-integrity",
            "failures. FA10_1F0S is consistently close to, but above, the preregistered 2.0 A",
            "cutoff; this does not authorize rounding or threshold relaxation. FA10_1LPG and",
            "FA10_2J2U reproducibly favor distant alternative poses. The run does not identify",
            "a unique molecular cause for any failure.",
            "",
            "The single frozen reserve could raise the cohort to at most 14 passing receptors,",
            "below the required 16. Lowering the cutoff, adding post-result receptors, or",
            "screening a 13-receptor subset cannot rescue the confirmatory FA10 endpoint.",
            "",
            "## Next Step",
            "",
            "Do not run FA10 Train-696. Preserve this valid negative technical result and proceed",
            "to HIVPR, the final target frozen in the multi-target master preregistration. Any",
            "later FA10 13-receptor experiment must be labeled post-hoc exploratory.",
            "",
        ]
    )
    return "\n".join(lines)


def run_adjudication(
    config_path: Path,
    core_archive_path: Path,
    diagnostic_archive_path: Path,
    root: Path,
) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    archives = dict(config["source_result_archives"])
    verify_archive(core_archive_path, dict(archives["core"]), "core")
    verify_archive(diagnostic_archive_path, dict(archives["diagnostics"]), "diagnostics")

    master_descriptor = dict(config["master_preregistration"])
    master_path = root / str(master_descriptor["path"])
    if file_sha256(master_path) != str(master_descriptor["sha256"]).upper():
        raise ValueError("master preregistration SHA-256 differs")
    master = read_json(master_path)
    target_order = [row["target_id"] for row in master["fixed_target_order"]]
    if target_order != ["EGFR", "FA10", "HIVPR"]:
        raise ValueError("master target order differs")
    if not truth(dict(master["target_selection_policy"])["no_result_driven_replacement"]):
        raise ValueError("master no-result-driven-replacement policy differs")

    source_config = dict(config["source_config"])
    with tarfile.open(core_archive_path, "r:gz") as core:
        source_bytes = archive_bytes(core, str(source_config["path"]))
        if hashlib.sha256(source_bytes).hexdigest().upper() != str(
            source_config["sha256"]
        ).upper():
            raise ValueError("source config SHA-256 differs inside core archive")
        summary = archive_json(core, SUMMARY_MEMBER)
        gate_rows = archive_csv(core, GATE_MEMBER)
        pair_rows = archive_csv(core, PAIR_MEMBER)
        core_records = {
            member: archive_bytes(core, member)
            for member in (SUMMARY_MEMBER, GATE_MEMBER, PAIR_MEMBER)
        }

    expected = dict(config["expected"])
    with tarfile.open(diagnostic_archive_path, "r:gz") as diagnostics:
        for member, content in core_records.items():
            if archive_bytes(diagnostics, member) != content:
                raise ValueError(f"core and diagnostic records differ: {member}")
        names = diagnostics.getnames()
        observed_counts = {
            "diagnostic_batch_summary_count": sum(
                name.endswith("/batch_summary.json") for name in names
            ),
            "diagnostic_rmsd_summary_count": sum(
                name.endswith("/rmsd/summary.json") for name in names
            ),
            "diagnostic_log_count": sum(name.endswith("/unidock.log") for name in names),
            "diagnostic_pose_count": sum(name.endswith("_out.pdbqt") for name in names),
        }
    for key, observed in observed_counts.items():
        if observed != int(expected[key]):
            raise ValueError(f"diagnostic archive count differs: {key}")

    if summary["status"] != "stage14d_fa10_cognate_redocking_gate_failed":
        raise ValueError("Stage 14d summary status differs")
    if len(gate_rows) != int(expected["receptor_count"]):
        raise ValueError("Stage 14d receptor gate row count differs")
    if len(pair_rows) != int(expected["pair_count"]):
        raise ValueError("Stage 14d pair row count differs")
    pair_keys = {(row["conformer_id"], row["seed_id"]) for row in pair_rows}
    if len(pair_keys) != len(pair_rows):
        raise ValueError("Stage 14d pair keys are duplicated")
    if any(
        int(summary[key]) != 0
        for key in ("unresolved_warning_event_count", "pose_integrity_failure_count")
    ):
        raise ValueError("Stage 14d technical integrity differs")
    if any(int(value) != 0 for value in dict(summary["data_boundary"]).values()):
        raise ValueError("Stage 14d crossed a protected data boundary")

    passing = [row["conformer_id"] for row in gate_rows if truth(row["gate_pass"])]
    failed = [row["conformer_id"] for row in gate_rows if not truth(row["gate_pass"])]
    if len(passing) != int(expected["expected_passing_receptor_count"]):
        raise ValueError("Stage 14d passing receptor count differs")
    if failed != [str(value) for value in expected["expected_failed_receptor_ids"]]:
        raise ValueError(f"Stage 14d failed receptor set differs: {failed}")

    maximum_recoverable = len(passing) + int(expected["available_reserve_receptor_count"])
    required = int(expected["required_passing_receptor_count"])
    if maximum_recoverable >= required:
        raise ValueError("frozen reserve capacity unexpectedly permits gate recovery")

    result = {
        "schema_version": "1.0",
        "adjudication_id": config["adjudication_id"],
        "status": "stage14e_fa10_confirmatory_technical_gate_closed",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_result_archives": {
            "core": {
                "basename": core_archive_path.name,
                "sha256": file_sha256(core_archive_path),
            },
            "diagnostics": {
                "basename": diagnostic_archive_path.name,
                "sha256": file_sha256(diagnostic_archive_path),
            },
        },
        "completed_pair_count": len(pair_rows),
        "engine_failure_count": 0,
        "unresolved_warning_event_count": 0,
        "pose_integrity_failure_count": 0,
        "passing_receptor_count": len(passing),
        "failed_receptor_count": len(failed),
        "passing_receptor_ids": passing,
        "failed_receptor_ids": failed,
        "failed_receptor_diagnostics": failure_diagnostics(
            pair_rows, failed, float(expected["maximum_rmsd_angstrom"])
        ),
        "recovery_capacity": {
            "passing_selected_receptor_count": len(passing),
            "frozen_reserve_receptor_count": int(expected["available_reserve_receptor_count"]),
            "maximum_recoverable_receptor_count": maximum_recoverable,
            "required_receptor_count": required,
            "remaining_shortfall": required - maximum_recoverable,
            "confirmatory_recovery_possible": False,
        },
        "data_boundary": {
            "ligand_labels_read": 0,
            "benchmark_docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "decision": {
            "run_fa10_train696": False,
            "redock_single_reserve_for_confirmatory_recovery": False,
            "next_frozen_target": "HIVPR",
            "fa10_13_receptor_future_status": "posthoc_exploratory_only",
        },
        "decision_boundary": config["decision_boundary"],
    }
    outputs = dict(config["outputs"])
    result_path = root / str(outputs["result_json"])
    report_path = root / str(outputs["report_md"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    report_path.write_text(render_report(result), encoding="ascii")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--core-archive", type=Path, required=True)
    parser.add_argument("--diagnostic-archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run_adjudication(
        args.config,
        args.core_archive,
        args.diagnostic_archive,
        args.root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
