"""Adjudicate the Stage 08b replacement failed-diagnostics archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

try:
    from .adjudicate_stage08_mk14_expanded16_redocking_failure import (
        file_sha256,
        read_json,
        summarize_gate,
    )
except ImportError:
    from adjudicate_stage08_mk14_expanded16_redocking_failure import (
        file_sha256,
        read_json,
        summarize_gate,
    )


def archive_json(archive: tarfile.TarFile, name: str) -> dict[str, object]:
    handle = archive.extractfile(archive.getmember(name))
    if handle is None:
        raise ValueError(f"archive member cannot be read: {name}")
    value = json.loads(handle.read().decode("ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"archive JSON is not an object: {name}")
    return value


def run_adjudication(config_path: Path, archive_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    source = dict(config["source_archive"])
    if archive_path.name != str(source["basename"]):
        raise ValueError("Stage 08b archive basename differs")
    if file_sha256(archive_path) != str(source["sha256"]).upper():
        raise ValueError("Stage 08b archive hash differs")
    expected = dict(config["expected"])
    receptor_ids = [str(value) for value in expected["receptor_ids"]]
    seed_ids = [str(value) for value in expected["seed_ids"]]
    prefix = "results/runs/stage08b_mk14_expanded16_replacement_redocking/redocking/batches"
    rows: list[dict[str, object]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        source_config_path = str(config["source_config"]["path"])
        handle = archive.extractfile(archive.getmember(source_config_path))
        if handle is None:
            raise ValueError("Stage 08b source config cannot be read")
        if hashlib.sha256(handle.read()).hexdigest().upper() != str(
            config["source_config"]["sha256"]
        ).upper():
            raise ValueError("Stage 08b source config hash differs")
        for seed_id in seed_ids:
            for receptor_id in receptor_ids:
                base = f"{prefix}/{seed_id}/{receptor_id}"
                batch = archive_json(archive, f"{base}/batch_summary.json")
                rmsd = archive_json(archive, f"{base}/rmsd/summary.json")
                warning = dict(batch["warning_adjudication"])
                pose = dict(batch["pose_integrity"])
                if batch.get("status") != "ok" or rmsd.get("status") != "ok":
                    raise ValueError(f"Stage 08b technical batch failed: {seed_id}/{receptor_id}")
                rows.append(
                    {
                        "seed_id": seed_id,
                        "receptor_id": receptor_id,
                        "affinity_kcal_per_mol": float(
                            rmsd["top_ranked_affinity_kcal_per_mol"]
                        ),
                        "rmsd_angstrom": float(rmsd["top_ranked_rmsd_angstrom"]),
                        "known_warning_event_count": int(
                            warning["known_warning_event_count"]
                        ),
                        "unresolved_warning_event_count": int(
                            warning["unresolved_warning_event_count"]
                        ),
                        "pose_integrity_failure_count": int(pose["failure_count"]),
                    }
                )
    if len(rows) != int(expected["pair_count"]):
        raise ValueError("Stage 08b pair count differs")
    if len({(row["seed_id"], row["receptor_id"]) for row in rows}) != len(rows):
        raise ValueError("Stage 08b result keys are duplicated")
    unresolved = sum(int(row["unresolved_warning_event_count"]) for row in rows)
    pose_failures = sum(int(row["pose_integrity_failure_count"]) for row in rows)
    if unresolved or pose_failures:
        raise ValueError("Stage 08b technical integrity gate failed")
    threshold = float(expected["maximum_median_rmsd_angstrom"])
    gate_rows = summarize_gate(
        rows,
        receptor_ids,
        threshold,
        int(expected["minimum_successful_seeds_per_receptor"]),
    )
    admitted = [str(row["conformer_id"]) for row in gate_rows if row["gate_pass"]]
    failed = [str(row["conformer_id"]) for row in gate_rows if not row["gate_pass"]]
    if admitted != [str(value) for value in expected["expected_admitted_receptor_ids"]]:
        raise ValueError("Stage 08b admitted receptor set differs")
    if failed != [str(value) for value in expected["expected_failed_receptor_ids"]]:
        raise ValueError("Stage 08b failed receptor set differs")
    result = {
        "schema_version": "1.0",
        "adjudication_id": config["adjudication_id"],
        "status": "stage08b_replacement_gate_failed_one_receptor",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_archive": {
            "basename": archive_path.name,
            "sha256": file_sha256(archive_path),
        },
        "completed_pair_count": len(rows),
        "engine_failure_count": 0,
        "unresolved_warning_event_count": unresolved,
        "pose_integrity_failure_count": pose_failures,
        "receptor_gate_results": gate_rows,
        "admitted_receptor_ids": admitted,
        "failed_receptor_ids": failed,
        "data_boundary": {
            "benchmark_ligand_labels_read": 0,
            "benchmark_docking_scores_read": 0,
            "previous_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "decision": "Admit 3ITZ, permanently exclude 2BAK, propagate exclusion to any zero-distance structural duplicate using the same co-crystal ligand, and preregister one more structural replacement.",
        "decision_boundary": config["decision_boundary"],
    }
    output_path = Path(str(config["output_json"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    run_adjudication(args.config, args.archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
