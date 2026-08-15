"""Adapt the shared checkpointed ligand preparer to the frozen PPARD Pilot-96."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from scripts.batch_prepare_ligand_pdbqt import file_sha256
from scripts.experimental.unidock import prepare_development_ligand_inputs as base


def validate_source(
    root: Path, config: dict[str, object]
) -> tuple[list[dict[str, str]], dict[str, Path], Counter[str]]:
    inputs = {
        key: base.verified(root, dict(value))
        for key, value in dict(config["inputs"]).items()
    }
    allocation = base.read_json(inputs["panel_allocation_summary"])
    source = dict(config["source"])
    if allocation["status"] != source["required_allocation_status"]:
        raise ValueError("PPARD pilot allocation did not pass")
    realized = dict(allocation["outputs"])["pilot_manifest_csv"]
    if realized["path"] != inputs["train_manifest"].relative_to(root).as_posix():
        raise ValueError("PPARD realized pilot-manifest path differs")
    if realized["sha256"].upper() != file_sha256(inputs["train_manifest"]):
        raise ValueError("PPARD realized pilot-manifest hash differs")
    for key in source["required_zero_allocation_boundaries"]:
        if int(allocation["data_boundary"][key]) != 0:
            raise ValueError(f"PPARD allocation crossed score boundary: {key}")

    rows = base.read_csv(inputs["train_manifest"])
    expected = dict(config["expected"])
    if len(rows) != int(expected["ligand_count"]):
        raise ValueError("PPARD pilot ligand count differs")
    labels = Counter(row["label"] for row in rows)
    frozen_labels = Counter(
        {key: int(value) for key, value in dict(expected["label_counts"]).items()}
    )
    if labels != frozen_labels:
        raise ValueError("PPARD pilot label balance differs")
    if {row["split"] for row in rows} != {source["required_split"]}:
        raise ValueError("PPARD pilot exposed another split")
    if {row["selection_role"] for row in rows} != {
        source["required_selection_role"]
    }:
        raise ValueError("PPARD pilot selection role differs")
    for column, required_value in dict(source["required_row_values"]).items():
        if {row[column] for row in rows} != {str(required_value)}:
            raise ValueError(f"PPARD pilot {column} differs")
    fold_counts = Counter(
        (row["pilot_outer_fold"], row["label"]) for row in rows
    )
    if fold_counts != Counter(
        {(str(fold), label): 12 for fold in range(4) for label in ("active", "decoy")}
    ):
        raise ValueError("PPARD pilot fold balance differs")
    if len({row["ligand_id"] for row in rows}) != len(rows):
        raise ValueError("PPARD pilot contains duplicate ligand IDs")
    return rows, inputs, labels


def run(
    config_path: Path,
    root: Path,
    audit_only: bool,
    resume: bool,
    overwrite: bool,
) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = base.read_json(config_path)
    adapter = dict(config["implementation"])["pilot_adapter"]
    if Path(str(adapter["path"])).resolve() != Path(__file__).resolve():
        raise ValueError("Stage58a pilot-adapter path differs")
    if file_sha256(Path(__file__)) != str(adapter["sha256"]).upper():
        raise ValueError("Stage58a pilot-adapter SHA-256 differs")
    validate_source(root, config)
    original = base.validate_source
    try:
        base.validate_source = validate_source
        return base.run(config_path, root, audit_only, resume, overwrite)
    finally:
        base.validate_source = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.audit_only, args.resume, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
