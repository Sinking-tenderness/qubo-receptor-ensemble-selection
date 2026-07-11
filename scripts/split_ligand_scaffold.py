"""Create a deterministic scaffold-disjoint train/validation/test split."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


LABELS = ("active", "decoy")
SPLITS = ("train", "validation", "test")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ligand_id", "label", "canonical_smiles"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"input is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("input contains no rows")
    if any(row["label"] not in LABELS for row in rows):
        raise ValueError("labels must be active or decoy")
    if len({row["ligand_id"] for row in rows}) != len(rows):
        raise ValueError("ligand_id values must be unique")
    return rows


def scaffold_for_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"cannot parse canonical_smiles: {smiles}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    # Acyclic molecules have an empty Murcko scaffold. Keep them separate by
    # canonical SMILES rather than putting all acyclic molecules in one group.
    return scaffold or Chem.MolToSmiles(molecule, canonical=True)


def target_counts(rows: list[dict[str, str]], fractions: dict[str, float]) -> dict[str, dict[str, float]]:
    totals = {label: sum(row["label"] == label for row in rows) for label in LABELS}
    return {
        split: {label: totals[label] * fractions[split] for label in LABELS}
        for split in SPLITS
    }


def assign_groups(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    fractions = {"train": 0.60, "validation": 0.20, "test": 0.20}
    targets = target_counts(rows, fractions)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        row = row.copy()
        row["scaffold_smiles"] = scaffold_for_smiles(row["canonical_smiles"])
        groups[row["scaffold_smiles"]].append(row)

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)
    assigned: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    counts = {split: {label: 0 for label in LABELS} for split in SPLITS}

    for scaffold, group_rows in group_items:
        group_counts = {label: sum(row["label"] == label for row in group_rows) for label in LABELS}

        def cost(split: str) -> tuple[float, float, str]:
            projected = {
                label: counts[split][label] + group_counts[label] for label in LABELS
            }
            deficit = sum(
                max(0.0, projected[label] - targets[split][label])
                / max(1.0, targets[split][label])
                for label in LABELS
            )
            distance = sum(
                abs(projected[label] - targets[split][label])
                / max(1.0, targets[split][label])
                for label in LABELS
            )
            return deficit, distance, split

        selected = min(SPLITS, key=cost)
        for row in group_rows:
            row["split"] = selected
            assigned[selected].append(row)
            counts[selected][row["label"]] += 1

    output = [row for split in SPLITS for row in assigned[split]]
    return sorted(output, key=lambda row: row["ligand_id"])


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, str]], seed: int) -> dict[str, object]:
    scaffold_splits: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        scaffold_splits[row["scaffold_smiles"]].add(row["split"])
        counts.setdefault(row["split"], {})[row["label"]] = (
            counts.setdefault(row["split"], {}).get(row["label"], 0) + 1
        )
    duplicated = {
        scaffold: sorted(splits)
        for scaffold, splits in scaffold_splits.items()
        if len(splits) > 1
    }
    return {
        "seed": seed,
        "input_rows": len(rows),
        "scaffold_count": len(scaffold_splits),
        "counts": counts,
        "scaffold_disjoint": not duplicated,
        "scaffolds_in_multiple_splits": duplicated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    rows = assign_groups(read_rows(args.input), args.seed)
    summary = build_summary(rows, args.seed)
    if not summary["scaffold_disjoint"]:
        raise RuntimeError("scaffold leakage detected")
    write_csv(args.output, rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
