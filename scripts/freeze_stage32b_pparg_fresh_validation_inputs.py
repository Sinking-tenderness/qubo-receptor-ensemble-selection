"""Freeze Stage32b validation identities and selected receptor source rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.stage32b_common import descriptor, read_csv, read_json, write_csv


def identity_sha256(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def freeze(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    allocation = read_json(root / config["inputs"]["stage19a_allocation_summary"])
    if allocation.get("status") != "stage19a_pparg_ligand_panels_frozen":
        raise ValueError("Stage19a allocation gate differs")
    all_rows = read_csv(root / config["inputs"]["stage19a_all_panel_manifest"])
    panel = config["fresh_validation_panel"]
    validation = [
        row for row in all_rows
        if row["selection_role"] == panel["selection_role"] and row["split"] == panel["split"]
    ]
    if len(validation) != int(panel["ligand_count"]):
        raise ValueError("Stage32b validation count differs")
    if Counter(row["label"] for row in validation) != Counter({"active": 75, "decoy": 1501}):
        raise ValueError("Stage32b validation labels differ")
    if identity_sha256([row["ligand_id"] for row in validation]) != panel["identity_sha256"]:
        raise ValueError("Stage32b validation identity hash differs")
    if len({row["ligand_id"] for row in validation}) != len(validation):
        raise ValueError("Stage32b validation ligand IDs are not unique")
    source_path = root / config["outputs"]["fresh_validation_source_manifest"]
    write_csv(source_path, validation)

    selection = read_json(root / config["outputs"]["train_selection_json"])
    if selection.get("status") != "stage32b_pparg_md_pair_train_selection_frozen":
        raise ValueError("Stage32b train selection differs")
    selected_ids = set(selection["selected_pair"]["receptor_ids"])
    receptors = [
        row for row in read_csv(root / config["inputs"]["stage32_prepared_receptor_manifest"])
        if row["conformer_id"] in selected_ids
    ]
    if len(receptors) != 2 or {row["conformer_id"] for row in receptors} != selected_ids:
        raise ValueError("Stage32b selected receptor source rows differ")
    receptor_path = root / config["outputs"]["selected_receptor_manifest"]
    write_csv(receptor_path, receptors)
    result = {
        "schema_version": "1.0",
        "status": "stage32b_validation_inputs_frozen_awaiting_remote_preparation",
        "experiment_id": config["experiment_id"],
        "config": descriptor(root, config_path),
        "fresh_validation_source_manifest": descriptor(root, source_path),
        "selected_receptor_source_manifest": descriptor(root, receptor_path),
        "counts": {"receptors": 2, "ligands": 1576, "active": 75, "decoy": 1501, "locked_test_rows": 0},
        "data_boundary": {"train_scores_read": 0, "fresh_validation_scores_read": 0, "locked_test_rows_read": 0},
        "next_gate": "prepare all fresh-validation ligands and copy the two frozen receptor PDBQT files from the completed Stage32 workspace",
        "decision_boundary": config["decision_boundary"],
    }
    output = root / config["outputs"]["preparation_result"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage32b_pparg_md_pair_fresh_validation.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    freeze(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
