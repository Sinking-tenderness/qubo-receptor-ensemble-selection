"""Extract frozen Stage 07b evidence for the Stage 07c confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath


CORE_SHA256 = "597CFB0C8B927188B9E6525793F91BFA4A352C7B52860D153B73D5557712EAFF"
DIAGNOSTICS_SHA256 = (
    "775D26D5E5D060FCA757039D2A003A7E888A1E8624E40562D8A37D02F7A104AD"
)
CORE_SCORES_MEMBER = (
    "results/runs/stage07b_mk14_unidock113_train160_enhanced_confirmation/"
    "scores.csv"
)
REPLAY_SCORES_MEMBER = (
    "results/runs/stage07b_mk14_unidock113_train160_enhanced_confirmation/"
    "enhanced/batches/seed2/MK14_3MPT_aligned/scores.csv"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name}")


def read_member(archive_path: Path, member_name: str) -> bytes:
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            safe_member(member.name)
        member = archive.getmember(member_name)
        if not member.isfile():
            raise ValueError(f"archive member is not a file: {member_name}")
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"cannot read archive member: {member_name}")
        return handle.read()


def csv_rows(value: bytes) -> list[dict[str, str]]:
    text = value.decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("evidence CSV is empty")
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty evidence CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--enhanced-output", type=Path, required=True)
    parser.add_argument("--replay-output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    args = parser.parse_args()

    observed_core_hash = file_sha256(args.core)
    observed_diagnostics_hash = file_sha256(args.diagnostics)
    if observed_core_hash != CORE_SHA256:
        raise ValueError("Stage 07b core archive SHA-256 differs")
    if observed_diagnostics_hash != DIAGNOSTICS_SHA256:
        raise ValueError("Stage 07b diagnostics archive SHA-256 differs")

    core_scores_bytes = read_member(args.core, CORE_SCORES_MEMBER)
    core_rows = csv_rows(core_scores_bytes)
    enhanced_rows = [row for row in core_rows if row["profile_id"] == "enhanced"]
    enhanced_rows.sort(
        key=lambda row: (
            row["seed_id"], row["receptor_id"], row["ligand_id"]
        )
    )
    if len(enhanced_rows) != 1920:
        raise ValueError("Stage 07b enhanced evidence row count differs")
    if {row["seed_id"] for row in enhanced_rows} != {"seed0", "seed1", "seed2"}:
        raise ValueError("Stage 07b enhanced evidence seeds differ")
    if any(row["pose_integrity_status"] != "ok" for row in enhanced_rows):
        raise ValueError("Stage 07b enhanced evidence contains a failed pose")

    replay_bytes = read_member(args.diagnostics, REPLAY_SCORES_MEMBER)
    replay_rows = csv_rows(replay_bytes)
    replay_rows.sort(key=lambda row: row["ligand_id"])
    if len(replay_rows) != 160:
        raise ValueError("Stage 07b warning replay reference count differs")
    if {row["seed_id"] for row in replay_rows} != {"seed2"}:
        raise ValueError("Stage 07b warning replay seed differs")
    if {row["receptor_id"] for row in replay_rows} != {"MK14_3MPT_aligned"}:
        raise ValueError("Stage 07b warning replay receptor differs")
    if any(row["pose_integrity_status"] != "ok" for row in replay_rows):
        raise ValueError("Stage 07b warning replay contains a failed pose")

    write_csv(args.enhanced_output, enhanced_rows)
    write_csv(args.replay_output, replay_rows)
    provenance = {
        "schema_version": "1.0",
        "status": "ok",
        "operation": "freeze Stage 07b enhanced evidence for Stage 07c",
        "sources": {
            "core_archive": {
                "filename": args.core.name,
                "sha256": observed_core_hash,
                "member": CORE_SCORES_MEMBER,
                "member_sha256": bytes_sha256(core_scores_bytes),
            },
            "pose_diagnostics_archive": {
                "filename": args.diagnostics.name,
                "sha256": observed_diagnostics_hash,
                "member": REPLAY_SCORES_MEMBER,
                "member_sha256": bytes_sha256(replay_bytes),
            },
        },
        "outputs": {
            "enhanced_three_seed_scores": {
                "path": args.enhanced_output.as_posix(),
                "rows": len(enhanced_rows),
                "sha256": file_sha256(args.enhanced_output),
            },
            "warning_replay_reference": {
                "path": args.replay_output.as_posix(),
                "rows": len(replay_rows),
                "sha256": file_sha256(args.replay_output),
            },
        },
        "fresh_validation_rows_read": 0,
        "test_rows_read": 0,
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
