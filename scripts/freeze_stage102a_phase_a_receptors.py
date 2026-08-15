"""Freeze the EGFR and FA10 redocking-qualified receptor manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCES = {
    "EGFR": {
        "adjudication": "data/stage13g_egfr_cognate_redocking_failure_adjudication.json",
        "manifest": "data/processed/stage13e_egfr_prepared_receptor_manifest.csv",
    },
    "FA10": {
        "adjudication": "data/stage14e_fa10_cognate_redocking_failure_adjudication.json",
        "manifest": "data/processed/stage14c_fa10_prepared_receptor_manifest.csv",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "stage102a_phase_a_receptors_frozen",
        "targets": {},
        "benchmark_docking_scores_read": 0,
    }
    for target, spec in SOURCES.items():
        adjudication_path = root / spec["adjudication"]
        source_manifest_path = root / spec["manifest"]
        adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
        source_rows = read_csv(source_manifest_path)
        passing_ids = list(adjudication["passing_receptor_ids"])
        by_id = {row["conformer_id"]: row for row in source_rows}
        if any(receptor_id not in by_id for receptor_id in passing_ids):
            raise ValueError(f"{target} passing receptor is missing from its source manifest")
        rows = [
            {
                **by_id[receptor_id],
                "stage102a_gate_pass": True,
                "stage102a_source_adjudication": spec["adjudication"],
            }
            for receptor_id in passing_ids
        ]
        for row in rows:
            receptor_path = root / row["receptor_pdbqt"]
            if not receptor_path.is_file() or sha256(receptor_path) != row["receptor_pdbqt_sha256"]:
                raise ValueError(f"{target} receptor identity differs: {row['conformer_id']}")
        output = root / f"data/processed/stage102a_{target.lower()}_passing_receptor_manifest.csv"
        write_csv(output, rows)
        summary["targets"][target] = {
            "receptor_count": len(rows),
            "receptor_ids": passing_ids,
            "manifest": {"path": output.relative_to(root).as_posix(), "sha256": sha256(output)},
            "source_adjudication": {"path": spec["adjudication"], "sha256": sha256(adjudication_path)},
        }
    if summary["targets"]["EGFR"]["receptor_count"] != 12 or summary["targets"]["FA10"]["receptor_count"] != 13:
        raise ValueError("Stage102A passing receptor counts differ")
    output = root / "data/stage102a_phase_a_receptor_freeze_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
