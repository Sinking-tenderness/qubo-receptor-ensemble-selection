"""Build a lossless, line-addressable ligand manifest from DUD-E ISM files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from .make_dude_subset import DudeRecord, parse_ism
except ImportError:
    from make_dude_subset import DudeRecord, parse_ism


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def manifest_row(
    record: DudeRecord, label: str, target_id: str, source: str
) -> dict[str, object]:
    return {
        "ligand_id": f"{target_id}_{label}_L{record.source_line_number:06d}",
        "smiles": record.smiles,
        "label": label,
        "source": source,
        "target_id": target_id,
        "source_molecule_id": record.source_molecule_id,
        "source_extra_id": record.source_extra_id,
        "source_line_number": record.source_line_number,
    }


def build_rows(
    active_records: list[DudeRecord],
    decoy_records: list[DudeRecord],
    target_id: str,
    source: str,
) -> list[dict[str, object]]:
    return [
        *(manifest_row(record, "active", target_id, source) for record in active_records),
        *(manifest_row(record, "decoy", target_id, source) for record in decoy_records),
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty DUD-E manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def duplicate_id_audit(rows: list[dict[str, object]]) -> dict[str, object]:
    counts = Counter((str(row["label"]), str(row["source_molecule_id"])) for row in rows)
    duplicates = {key: count for key, count in counts.items() if count > 1}
    return {
        "unique_label_source_id_count": len(counts),
        "duplicate_label_source_id_count": len(duplicates),
        "rows_with_duplicate_label_source_id": sum(duplicates.values()),
        "maximum_label_source_id_multiplicity": max(duplicates.values(), default=1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actives", type=Path, required=True)
    parser.add_argument("--decoys", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--source", default="DUD-E")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.actives, args.decoys):
        if not path.is_file():
            raise FileNotFoundError(path)

    active_records = parse_ism(args.actives)
    decoy_records = parse_ism(args.decoys)
    rows = build_rows(active_records, decoy_records, args.target_id, args.source)
    if len({str(row["ligand_id"]) for row in rows}) != len(rows):
        raise RuntimeError("line-addressable ligand IDs are not unique")
    write_csv(args.output, rows)
    summary = {
        "schema_version": "1.0",
        "status": "ok",
        "operation": "lossless DUD-E ISM to line-addressable ligand manifest",
        "target_id": args.target_id,
        "source": args.source,
        "inputs": {
            "actives": {
                "path": args.actives.as_posix(),
                "sha256": file_sha256(args.actives),
                "row_count": len(active_records),
            },
            "decoys": {
                "path": args.decoys.as_posix(),
                "sha256": file_sha256(args.decoys),
                "row_count": len(decoy_records),
            },
        },
        "output": {
            "path": args.output.as_posix(),
            "sha256": file_sha256(args.output),
            "row_count": len(rows),
        },
        "label_counts": {"active": len(active_records), "decoy": len(decoy_records)},
        "duplicate_source_id_audit": duplicate_id_audit(rows),
        "deduplication_applied": False,
        "interpretation_note": (
            "Every non-empty source row is retained. Ligand IDs encode target, label, "
            "and one-based source line number."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
