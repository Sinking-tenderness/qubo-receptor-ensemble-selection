"""Select a deterministic, group-diverse pilot from one locked ligand split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


LABELS = ("active", "decoy")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ligand_id", "label", "split", "split_group_id", "scaffold_smiles"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"split table is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("split table is empty")
    return rows


def sample_rows(
    rows: list[dict[str, str]],
    source_split: str,
    count_per_label: int,
    seed: int,
    selection_role: str = "execution_smoke_only",
) -> list[dict[str, str]]:
    if count_per_label <= 0:
        raise ValueError("count_per_label must be positive")
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    used_groups: set[str] = set()
    for label in LABELS:
        candidates = [
            row.copy()
            for row in rows
            if row["split"] == source_split and row["label"] == label
        ]
        candidates.sort(key=lambda row: row["ligand_id"])
        rng.shuffle(candidates)
        label_selected: list[dict[str, str]] = []
        for row in candidates:
            if row["split_group_id"] in used_groups:
                continue
            row["selection_role"] = selection_role
            label_selected.append(row)
            used_groups.add(row["split_group_id"])
            if len(label_selected) == count_per_label:
                break
        if len(label_selected) != count_per_label:
            raise ValueError(
                f"could select only {len(label_selected)} group-diverse {label} rows"
            )
        selected.extend(label_selected)
    return sorted(selected, key=lambda row: (row["label"], row["ligand_id"]))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--count-per-label", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selection-role", default="execution_smoke_only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    selected = sample_rows(
        read_rows(args.input),
        args.source_split,
        args.count_per_label,
        args.seed,
        args.selection_role,
    )
    write_csv(args.output, selected)
    summary = {
        "schema_version": "1.0",
        "status": "ok",
        "operation": "deterministic group-diverse split pilot selection",
        "input": {"path": args.input.as_posix(), "sha256": file_sha256(args.input)},
        "source_split": args.source_split,
        "selection_role": args.selection_role,
        "seed": args.seed,
        "count_per_label": args.count_per_label,
        "row_count": len(selected),
        "label_counts": {
            label: sum(row["label"] == label for row in selected) for label in LABELS
        },
        "unique_split_group_count": len({row["split_group_id"] for row in selected}),
        "unique_scaffold_count": len({row["scaffold_smiles"] for row in selected}),
        "selected_ligand_ids": [row["ligand_id"] for row in selected],
        "output": {"path": args.output.as_posix(), "sha256": file_sha256(args.output)},
        "interpretation_boundary": (
            "The selected rows retain their locked source-split role. This selector "
            "does not release or inspect the locked test partition."
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
