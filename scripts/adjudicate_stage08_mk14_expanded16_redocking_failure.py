"""Adjudicate a Stage 08 failed-diagnostics archive without rerunning docking."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tarfile
from pathlib import Path


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


def archive_json(archive: tarfile.TarFile, name: str) -> dict[str, object]:
    member = archive.getmember(name)
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"archive member cannot be read: {name}")
    value = json.loads(handle.read().decode("ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"archive JSON is not an object: {name}")
    return value


def summarize_gate(
    rows: list[dict[str, object]],
    receptor_ids: list[str],
    threshold: float,
    minimum_successes: int,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for receptor_id in receptor_ids:
        receptor_rows = [row for row in rows if row["receptor_id"] == receptor_id]
        if len(receptor_rows) != 3:
            raise ValueError(f"redocking evidence is incomplete: {receptor_id}")
        rmsds = [float(row["rmsd_angstrom"]) for row in receptor_rows]
        successful = sum(value <= threshold for value in rmsds)
        median_rmsd = statistics.median(rmsds)
        result.append(
            {
                "conformer_id": receptor_id,
                "successful_seed_count": successful,
                "median_rmsd_angstrom": median_rmsd,
                "maximum_rmsd_angstrom": max(rmsds),
                "gate_pass": successful >= minimum_successes
                and median_rmsd <= threshold,
            }
        )
    return result


def run_adjudication(
    config_path: Path, archive_path: Path
) -> dict[str, object]:
    config = read_json(config_path)
    source = dict(config["source_archive"])
    if archive_path.name != str(source["basename"]):
        raise ValueError("failed-diagnostics archive basename differs")
    if file_sha256(archive_path) != str(source["sha256"]).upper():
        raise ValueError("failed-diagnostics archive SHA-256 differs")

    expected = dict(config["expected"])
    receptor_ids = [str(value) for value in expected["receptor_ids"]]
    seed_ids = [str(value) for value in expected["seed_ids"]]
    prefix = "results/runs/stage08_mk14_expanded16_unidock_redocking/redocking/batches"
    rows: list[dict[str, object]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        source_config_path = str(config["source_config"]["path"])
        config_member = archive.getmember(source_config_path)
        config_handle = archive.extractfile(config_member)
        if config_handle is None:
            raise ValueError("source config cannot be read from diagnostics archive")
        config_payload = config_handle.read()
        if hashlib.sha256(config_payload).hexdigest().upper() != str(
            config["source_config"]["sha256"]
        ).upper():
            raise ValueError("source config hash differs inside diagnostics archive")

        for seed_id in seed_ids:
            for receptor_id in receptor_ids:
                base = f"{prefix}/{seed_id}/{receptor_id}"
                batch = archive_json(archive, f"{base}/batch_summary.json")
                rmsd = archive_json(archive, f"{base}/rmsd/summary.json")
                warning = dict(batch["warning_adjudication"])
                pose = dict(batch["pose_integrity"])
                if batch.get("status") != "ok" or rmsd.get("status") != "ok":
                    raise ValueError(f"failed technical batch: {seed_id}/{receptor_id}")
                rows.append(
                    {
                        "seed_id": seed_id,
                        "receptor_id": receptor_id,
                        "affinity_kcal_per_mol": float(
                            rmsd["top_ranked_affinity_kcal_per_mol"]
                        ),
                        "rmsd_angstrom": float(rmsd["top_ranked_rmsd_angstrom"]),
                        "pose_success": bool(rmsd["top_ranked_pose_success"]),
                        "known_warning_event_count": int(
                            warning["known_warning_event_count"]
                        ),
                        "unresolved_warning_event_count": int(
                            warning["unresolved_warning_event_count"]
                        ),
                        "pose_integrity_failure_count": int(pose["failure_count"]),
                        "batch_summary_sha256": hashlib.sha256(
                            json.dumps(batch, sort_keys=True, separators=(",", ":")).encode(
                                "ascii"
                            )
                        ).hexdigest().upper(),
                    }
                )

    if len(rows) != int(expected["pair_count"]):
        raise ValueError("redocking evidence pair count differs")
    keys = {(row["seed_id"], row["receptor_id"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("redocking evidence contains duplicate keys")
    unresolved = sum(int(row["unresolved_warning_event_count"]) for row in rows)
    pose_failures = sum(int(row["pose_integrity_failure_count"]) for row in rows)
    if unresolved or pose_failures:
        raise ValueError("redocking evidence contains a technical integrity failure")

    threshold = float(expected["maximum_median_rmsd_angstrom"])
    gate_rows = summarize_gate(
        rows,
        receptor_ids,
        threshold,
        int(expected["minimum_successful_seeds_per_receptor"]),
    )
    failed = [str(row["conformer_id"]) for row in gate_rows if not row["gate_pass"]]
    if failed != [str(value) for value in expected["expected_failed_receptor_ids"]]:
        raise ValueError(f"failed receptor set differs: {failed}")

    result = {
        "schema_version": "1.0",
        "adjudication_id": config["adjudication_id"],
        "status": "stage08_redocking_gate_failed_two_receptors",
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_archive": {
            "basename": archive_path.name,
            "sha256": file_sha256(archive_path),
        },
        "completed_pair_count": len(rows),
        "expected_pair_count": int(expected["pair_count"]),
        "engine_failure_count": 0,
        "unresolved_warning_event_count": unresolved,
        "pose_integrity_failure_count": pose_failures,
        "receptor_gate_results": gate_rows,
        "admitted_receptor_ids": [
            str(row["conformer_id"]) for row in gate_rows if row["gate_pass"]
        ],
        "failed_receptor_ids": failed,
        "data_boundary": {
            "benchmark_ligand_labels_read": 0,
            "benchmark_docking_scores_read": 0,
            "previous_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "decision": "Do not relax the preregistered RMSD gate and do not rerun the same deterministic checkpoints. Exclude the two failed receptors and preregister a structural-distance replacement cascade.",
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
