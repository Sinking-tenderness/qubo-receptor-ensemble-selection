"""Merge disjoint ligand CSV manifests with hash and role audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


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


def merge_rows(groups: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    rows = [row for group in groups for row in group]
    identifiers = [row["ligand_id"] for row in rows]
    duplicates = sorted(identifier for identifier, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        raise ValueError(f"ligand IDs occur in multiple inputs: {duplicates}")
    return sorted(
        rows,
        key=lambda row: (
            row.get("selection_role", ""),
            row.get("label", ""),
            row["ligand_id"],
        ),
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    for path in args.input:
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = merge_rows([read_csv(path) for path in args.input])
    write_csv(args.output, rows)
    role_label_counts = Counter(
        (row.get("selection_role", "unspecified"), row["label"]) for row in rows
    )
    summary = {
        "schema_version": "1.0",
        "status": "ok",
        "operation": "merge disjoint ligand manifests",
        "inputs": [
            {"path": path.as_posix(), "sha256": file_sha256(path)} for path in args.input
        ],
        "row_count": len(rows),
        "unique_ligand_count": len({row["ligand_id"] for row in rows}),
        "role_label_counts": {
            f"{role}:{label}": count
            for (role, label), count in sorted(role_label_counts.items())
        },
        "output": {"path": args.output.as_posix(), "sha256": file_sha256(args.output)},
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
