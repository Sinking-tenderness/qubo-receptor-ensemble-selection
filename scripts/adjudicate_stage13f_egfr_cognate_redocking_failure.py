"""Adjudicate the completed Stage 13f EGFR redocking failure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import tarfile
from pathlib import Path


SUMMARY_MEMBER = "data/stage13f_egfr_cognate_redocking_summary.json"
GATE_MEMBER = "data/processed/stage13f_egfr_receptor_gate_results.csv"
PAIR_MEMBER = "data/processed/stage13f_egfr_cognate_redocking_results.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def archive_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    handle = archive.extractfile(archive.getmember(name))
    if handle is None:
        raise ValueError(f"archive member cannot be read: {name}")
    return handle.read()


def archive_json(archive: tarfile.TarFile, name: str) -> dict[str, object]:
    value = json.loads(archive_bytes(archive, name).decode("ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"archive JSON is not an object: {name}")
    return value


def archive_csv(archive: tarfile.TarFile, name: str) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(archive_bytes(archive, name).decode("utf-8"))))
    if not rows:
        raise ValueError(f"archive CSV is empty: {name}")
    return rows


def truth(value: object) -> bool:
    return value is True or str(value).lower() == "true"


def verify_archive(path: Path, descriptor: dict[str, object], label: str) -> None:
    if path.name != str(descriptor["basename"]):
        raise ValueError(f"{label} archive basename differs")
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"{label} archive SHA-256 differs")


def failure_diagnostics(
    pair_rows: list[dict[str, str]],
    failed_ids: list[str],
    citation_ids: set[str],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for receptor_id in failed_ids:
        selected = [row for row in pair_rows if row["conformer_id"] == receptor_id]
        rmsds = [float(row["top_ranked_rmsd_angstrom"]) for row in selected]
        affinities = [float(row["top_ranked_affinity_kcal_per_mol"]) for row in selected]
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
                "citation_mentions_covalent_or_irreversible_mechanism": (
                    receptor_id in citation_ids
                ),
                "interpretation": (
                    "three-seed stable alternative pose; citation wording suggests that "
                    "the frozen noncovalent engine may omit relevant chemistry"
                    if receptor_id in citation_ids
                    else "three-seed stable alternative pose under the frozen noncovalent protocol"
                ),
            }
        )
    return output


def render_report(result: dict[str, object]) -> str:
    lines = [
        "# Stage 13g EGFR Cognate-Redocking Failure Adjudication",
        "",
        "## Decision",
        "",
        "Stage 13f completed all 48 planned Uni-Dock jobs with zero unresolved warnings",
        "and zero pose-integrity failures, but only 12 of 16 receptors passed the frozen",
        "three-seed RMSD gate. The EGFR confirmatory technical gate is therefore closed.",
        "",
        "| Receptor | Median RMSD (A) | Range (A) | Seeds passing | Citation diagnostic |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in result["failed_receptor_diagnostics"]:
        lines.append(
            "| {conformer_id} | {median_rmsd_angstrom:.3f} | "
            "{minimum_rmsd_angstrom:.3f}-{maximum_rmsd_angstrom:.3f} | "
            "{successful_seed_count}/3 | {citation} |".format(
                **row,
                citation=(
                    "covalent/irreversible wording"
                    if row["citation_mentions_covalent_or_irreversible_mechanism"]
                    else "none"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The failures are reproducible across seeds and are not runtime failures. Citation",
            "wording is a mechanism hypothesis only; no explicit protein-ligand covalent bond",
            "was present in the admitted coordinates. The result does not identify a unique cause.",
            "",
            "The single frozen reserve could raise the cohort to at most 13 passing receptors,",
            "below the required 16. Lowering the RMSD cutoff, adding post-result receptors, or",
            "screening a 12-receptor subset cannot rescue the confirmatory EGFR endpoint.",
            "",
            "## Next Step",
            "",
            "Do not run EGFR Train-696. Preserve this valid negative technical result and proceed",
            "to FA10, the next target frozen in the multi-target master preregistration. Any later",
            "EGFR 12-receptor experiment must be labeled post-hoc exploratory.",
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
    if target_order[:2] != ["EGFR", "FA10"]:
        raise ValueError("master target order differs")
    policy = dict(master["target_selection_policy"])
    if not truth(policy["no_result_driven_replacement"]):
        raise ValueError("master no-result-driven-replacement policy differs")

    source_config = dict(config["source_config"])
    with tarfile.open(core_archive_path, "r:gz") as core:
        if hashlib.sha256(archive_bytes(core, str(source_config["path"]))).hexdigest().upper() != str(
            source_config["sha256"]
        ).upper():
            raise ValueError("source config SHA-256 differs inside core archive")
        summary = archive_json(core, SUMMARY_MEMBER)
        gate_rows = archive_csv(core, GATE_MEMBER)
        pair_rows = archive_csv(core, PAIR_MEMBER)

    expected = dict(config["expected"])
    with tarfile.open(diagnostic_archive_path, "r:gz") as diagnostics:
        if archive_json(diagnostics, SUMMARY_MEMBER) != summary:
            raise ValueError("core and diagnostic summaries differ")
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

    if summary["status"] != "stage13f_egfr_cognate_redocking_gate_failed":
        raise ValueError("Stage 13f summary status differs")
    if len(gate_rows) != int(expected["receptor_count"]):
        raise ValueError("Stage 13f receptor gate row count differs")
    if len(pair_rows) != int(expected["pair_count"]):
        raise ValueError("Stage 13f pair row count differs")
    pair_keys = {(row["conformer_id"], row["seed_id"]) for row in pair_rows}
    if len(pair_keys) != len(pair_rows):
        raise ValueError("Stage 13f pair keys are duplicated")
    if any(
        int(summary[key]) != 0
        for key in (
            "unresolved_warning_event_count",
            "pose_integrity_failure_count",
        )
    ):
        raise ValueError("Stage 13f technical integrity differs")
    if any(int(value) != 0 for value in dict(summary["data_boundary"]).values()):
        raise ValueError("Stage 13f crossed a protected data boundary")

    passing = [row["conformer_id"] for row in gate_rows if truth(row["gate_pass"])]
    failed = [row["conformer_id"] for row in gate_rows if not truth(row["gate_pass"])]
    if len(passing) != int(expected["expected_passing_receptor_count"]):
        raise ValueError("Stage 13f passing receptor count differs")
    if failed != [str(value) for value in expected["expected_failed_receptor_ids"]]:
        raise ValueError(f"Stage 13f failed receptor set differs: {failed}")

    citation_rows = dict(dict(summary["input_audit"])["citation_intent_diagnostic"])[
        "records"
    ]
    citation_ids = {str(row["conformer_id"]) for row in citation_rows}
    maximum_recoverable = len(passing) + int(expected["available_reserve_receptor_count"])
    required = int(expected["required_passing_receptor_count"])
    if maximum_recoverable >= required:
        raise ValueError("frozen reserve capacity unexpectedly permits gate recovery")

    result = {
        "schema_version": "1.0",
        "adjudication_id": config["adjudication_id"],
        "status": "stage13g_egfr_confirmatory_technical_gate_closed",
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
            pair_rows, failed, citation_ids
        ),
        "recovery_capacity": {
            "passing_selected_receptor_count": len(passing),
            "frozen_reserve_receptor_count": int(
                expected["available_reserve_receptor_count"]
            ),
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
            "run_egfr_train696": False,
            "redock_single_reserve_for_confirmatory_recovery": False,
            "next_frozen_target": "FA10",
            "egfr_12_receptor_future_status": "posthoc_exploratory_only",
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
