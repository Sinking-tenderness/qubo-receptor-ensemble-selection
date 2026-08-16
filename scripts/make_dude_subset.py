"""Create a small DUD-E ligand table for teaching-scale screening tests."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DudeRecord:
    smiles: str
    source_molecule_id: str
    source_extra_id: str
    source_line_number: int


def parse_ism(path: Path) -> list[DudeRecord]:
    records: list[DudeRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_number} has fewer than 2 columns")
            smiles = parts[0]
            source_molecule_id = parts[1]
            source_extra_id = parts[2] if len(parts) > 2 else ""
            records.append(
                DudeRecord(
                    smiles=smiles,
                    source_molecule_id=source_molecule_id,
                    source_extra_id=source_extra_id,
                    source_line_number=line_number,
                )
            )
    return records


def unique_by_exact_smiles(records: list[DudeRecord]) -> list[DudeRecord]:
    seen: set[str] = set()
    unique: list[DudeRecord] = []
    for record in records:
        if record.smiles in seen:
            continue
        seen.add(record.smiles)
        unique.append(record)
    return unique


def sample_records(records: list[DudeRecord], count: int, seed: int) -> list[DudeRecord]:
    if count > len(records):
        raise ValueError(f"requested {count} records, but only {len(records)} are available")
    rng = random.Random(seed)
    sampled = rng.sample(records, count)
    return sorted(sampled, key=lambda item: item.source_line_number)


def write_subset(
    output_csv: Path,
    active_records: list[DudeRecord],
    decoy_records: list[DudeRecord],
    target_id: str,
    source: str,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ligand_id",
        "smiles",
        "label",
        "source",
        "target_id",
        "source_molecule_id",
        "source_extra_id",
        "source_line_number",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(active_records, start=1):
            writer.writerow(
                {
                    "ligand_id": f"{target_id}_A{index:04d}",
                    "smiles": record.smiles,
                    "label": "active",
                    "source": source,
                    "target_id": target_id,
                    "source_molecule_id": record.source_molecule_id,
                    "source_extra_id": record.source_extra_id,
                    "source_line_number": record.source_line_number,
                }
            )
        for index, record in enumerate(decoy_records, start=1):
            writer.writerow(
                {
                    "ligand_id": f"{target_id}_D{index:04d}",
                    "smiles": record.smiles,
                    "label": "decoy",
                    "source": source,
                    "target_id": target_id,
                    "source_molecule_id": record.source_molecule_id,
                    "source_extra_id": record.source_extra_id,
                    "source_line_number": record.source_line_number,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actives", type=Path, required=True, help="DUD-E actives_final.ism")
    parser.add_argument("--decoys", type=Path, required=True, help="DUD-E decoys_final.ism")
    parser.add_argument("--output", type=Path, required=True, help="Output subset CSV")
    parser.add_argument("--n-actives", type=int, default=10)
    parser.add_argument("--n-decoys", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--target-id", default="CDK2")
    parser.add_argument("--source", default="DUD-E")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    active_records = unique_by_exact_smiles(parse_ism(args.actives))
    decoy_records = unique_by_exact_smiles(parse_ism(args.decoys))
    active_sample = sample_records(active_records, args.n_actives, args.seed)
    decoy_sample = sample_records(decoy_records, args.n_decoys, args.seed + 1)
    write_subset(args.output, active_sample, decoy_sample, args.target_id, args.source)
    print(f"actives_available={len(active_records)}")
    print(f"decoys_available={len(decoy_records)}")
    print(f"actives_written={len(active_sample)}")
    print(f"decoys_written={len(decoy_sample)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
