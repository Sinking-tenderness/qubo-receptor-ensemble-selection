"""Adapt the checkpointed ligand preparer to PPARD Remaining-144."""

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
    stage60 = base.read_json(inputs["stage60_result"])
    stage60_audit = base.read_json(inputs["stage60_audit"])
    freeze = base.read_json(inputs["remaining_freeze_summary"])
    source = dict(config["source"])
    if allocation["status"] != source["required_allocation_status"]:
        raise ValueError("PPARD allocation did not pass")
    if stage60["status"] != "stage60_ppard_transferred_qubo_and_k_rule_frozen":
        raise ValueError("Stage60 objective freeze did not complete")
    if not stage60["decision"]["remaining_development_ligand_preparation_authorized"]:
        raise ValueError("Stage60 did not authorize ligand preparation")
    if stage60_audit["status"] != "stage60_ppard_transferred_qubo_independent_audit_ok":
        raise ValueError("Stage60 independent audit did not pass")
    if freeze["status"] != "stage61a_ppard_remaining144_manifest_frozen":
        raise ValueError("PPARD Remaining-144 manifest was not frozen")
    realized_full = dict(allocation["outputs"])["train_manifest_csv"]
    if realized_full["path"] != inputs["full_train_manifest"].relative_to(root).as_posix():
        raise ValueError("PPARD full train-manifest path differs")
    if realized_full["sha256"].upper() != file_sha256(inputs["full_train_manifest"]):
        raise ValueError("PPARD full train-manifest hash differs")
    realized_remaining = dict(freeze["output"])
    if realized_remaining["path"] != inputs["train_manifest"].relative_to(root).as_posix():
        raise ValueError("PPARD Remaining-144 path differs")
    if realized_remaining["sha256"].upper() != file_sha256(inputs["train_manifest"]):
        raise ValueError("PPARD Remaining-144 hash differs")
    for key in source["required_zero_allocation_boundaries"]:
        if int(allocation["data_boundary"][key]) != 0:
            raise ValueError(f"PPARD allocation crossed score boundary: {key}")

    full_rows = base.read_csv(inputs["full_train_manifest"])
    rows = base.read_csv(inputs["train_manifest"])
    expected_rows = [row for row in full_rows if row["pilot_selected"] == "False"]
    if rows != expected_rows:
        raise ValueError("PPARD Remaining-144 is not the exact Train-240 complement")
    expected = dict(config["expected"])
    if len(rows) != int(expected["ligand_count"]):
        raise ValueError("PPARD Remaining-144 ligand count differs")
    labels = Counter(row["label"] for row in rows)
    frozen_labels = Counter(
        {key: int(value) for key, value in dict(expected["label_counts"]).items()}
    )
    if labels != frozen_labels:
        raise ValueError("PPARD Remaining-144 label balance differs")
    if {row["split"] for row in rows} != {source["required_split"]}:
        raise ValueError("PPARD Remaining-144 exposed another split")
    if {row["selection_role"] for row in rows} != {
        source["required_selection_role"]
    }:
        raise ValueError("PPARD Remaining-144 selection role differs")
    for column, required_value in dict(source["required_row_values"]).items():
        if {row[column] for row in rows} != {str(required_value)}:
            raise ValueError(f"PPARD Remaining-144 {column} differs")
    if len({row["ligand_id"] for row in rows}) != len(rows):
        raise ValueError("PPARD Remaining-144 contains duplicate ligand IDs")
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
    adapter = dict(config["implementation"])["remaining_adapter"]
    if Path(str(adapter["path"])).resolve() != Path(__file__).resolve():
        raise ValueError("Stage61a adapter path differs")
    if file_sha256(Path(__file__)) != str(adapter["sha256"]).upper():
        raise ValueError("Stage61a adapter SHA-256 differs")
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
