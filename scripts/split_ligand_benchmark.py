"""Create a deterministic stratified train/validation/test ligand split."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ligand_id", "label", "canonical_smiles"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"input is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("input contains no ligand rows")
    if any(row["label"] not in {"active", "decoy"} for row in rows):
        raise ValueError("labels must be active or decoy")
    canonical = [row["canonical_smiles"] for row in rows]
    if len(canonical) != len(set(canonical)):
        raise ValueError("canonical SMILES duplicates must be resolved before splitting")
    return rows


def split_rows(
    rows: list[dict[str, str]], train_fraction: float, validation_fraction: float, seed: int
) -> list[dict[str, str]]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")

    rng = random.Random(seed)
    output: list[dict[str, str]] = []
    for label in ("active", "decoy"):
        label_rows = [row.copy() for row in rows if row["label"] == label]
        rng.shuffle(label_rows)
        train_count = max(1, round(len(label_rows) * train_fraction))
        validation_count = max(1, round(len(label_rows) * validation_fraction))
        if train_count + validation_count >= len(label_rows):
            raise ValueError(f"not enough {label} rows for three non-empty splits")
        for row in label_rows[:train_count]:
            row["split"] = "train"
        for row in label_rows[train_count : train_count + validation_count]:
            row["split"] = "validation"
        for row in label_rows[train_count + validation_count :]:
            row["split"] = "test"
        output.extend(label_rows)
    return sorted(output, key=lambda row: row["ligand_id"])


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()

    rows = split_rows(
        read_rows(args.input), args.train_fraction, args.validation_fraction, args.seed
    )
    write_csv(args.output, rows)
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        counts.setdefault(row["split"], {})[row["label"]] = (
            counts.setdefault(row["split"], {}).get(row["label"], 0) + 1
        )
    summary = {
        "input_rows": len(rows),
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "test_fraction": 1 - args.train_fraction - args.validation_fraction,
        "counts": counts,
        "split_disjoint": len({row["ligand_id"] for row in rows}) == len(rows),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
