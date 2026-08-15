"""Freeze the 144 PPARD development ligands not used by Pilot-96."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run(root: Path, output: Path, summary_output: Path) -> dict[str, Any]:
    root = root.resolve()
    train_path = root / "data/processed/stage56_ppard_train240_ligand_manifest.csv"
    pilot_path = root / "data/processed/stage56_ppard_pilot96_ligand_manifest.csv"
    stage60_path = root / "data/stage60_ppard_transferred_qubo_freeze_result.json"
    stage60_audit_path = root / "data/stage60_ppard_transferred_qubo_freeze_audit.json"
    stage60 = read_json(stage60_path)
    stage60_audit = read_json(stage60_audit_path)
    if stage60["status"] != "stage60_ppard_transferred_qubo_and_k_rule_frozen":
        raise ValueError("Stage60 freeze did not complete")
    if not stage60["decision"]["remaining_development_manifest_freeze_authorized"]:
        raise ValueError("Stage60 did not authorize the remaining manifest")
    if stage60_audit["status"] != "stage60_ppard_transferred_qubo_independent_audit_ok":
        raise ValueError("Stage60 independent audit did not pass")
    train = read_csv(train_path)
    pilot = read_csv(pilot_path)
    remaining = [row for row in train if row["pilot_selected"] == "False"]
    if len(train) != 240 or len(pilot) != 96 or len(remaining) != 144:
        raise ValueError("PPARD development partition dimensions differ")
    if Counter(row["label"] for row in remaining) != {
        "active": 72,
        "decoy": 72,
    }:
        raise ValueError("PPARD Remaining-144 label balance differs")
    if {row["pilot_outer_fold"] for row in remaining} != {""} or {
        row["pilot_role"] for row in remaining
    } != {""}:
        raise ValueError("PPARD Remaining-144 contains a pilot role")
    pilot_ids = {row["ligand_id"] for row in pilot}
    remaining_ids = {row["ligand_id"] for row in remaining}
    if pilot_ids & remaining_ids or pilot_ids | remaining_ids != {
        row["ligand_id"] for row in train
    }:
        raise ValueError("PPARD Pilot-96 and Remaining-144 do not partition Train-240")
    output = output if output.is_absolute() else root / output
    summary_output = (
        summary_output if summary_output.is_absolute() else root / summary_output
    )
    write_csv(output, remaining)
    summary = {
        "schema_version": "1.0",
        "freeze_id": "stage61a-ppard-remaining144-development-manifest-v1",
        "status": "stage61a_ppard_remaining144_manifest_frozen",
        "source_train_manifest": {
            "path": train_path.relative_to(root).as_posix(),
            "sha256": sha256(train_path),
        },
        "excluded_pilot_manifest": {
            "path": pilot_path.relative_to(root).as_posix(),
            "sha256": sha256(pilot_path),
        },
        "stage60_result": {
            "path": stage60_path.relative_to(root).as_posix(),
            "sha256": sha256(stage60_path),
        },
        "stage60_audit": {
            "path": stage60_audit_path.relative_to(root).as_posix(),
            "sha256": sha256(stage60_audit_path),
        },
        "ligand_count": len(remaining),
        "label_counts": dict(
            sorted(Counter(row["label"] for row in remaining).items())
        ),
        "pilot_overlap_count": 0,
        "train_partition_complete": True,
        "output": {
            "path": output.relative_to(root).as_posix(),
            "sha256": sha256(output),
            "size_bytes": output.stat().st_size,
        },
        "data_boundary": {
            "development_identity_rows_read": 240,
            "docking_score_rows_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
        },
        "next_step": "prepare only these 144 ligands for Uni-Dock without launching docking",
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/stage61a_ppard_remaining144_ligand_manifest.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/stage61a_ppard_remaining144_manifest_freeze.json"),
    )
    args = parser.parse_args()
    run(args.root, args.output, args.summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
