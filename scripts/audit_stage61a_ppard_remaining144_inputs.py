"""Independently audit prepared PPARD Remaining-144 Uni-Dock inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
    macrocycle_closure_atom_types,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_descriptor(root: Path, descriptor: dict[str, Any]) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file() or sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage61a descriptor differs: {path}")
    return path


def run(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    stage60 = read_json(root / "data/stage60_ppard_transferred_qubo_freeze_result.json")
    stage60_audit = read_json(root / "data/stage60_ppard_transferred_qubo_freeze_audit.json")
    freeze = read_json(root / "data/stage61a_ppard_remaining144_manifest_freeze.json")
    preparation = read_json(
        root / "data/stage61a_ppard_remaining144_unidock_input_summary.json"
    )
    if stage60["status"] != "stage60_ppard_transferred_qubo_and_k_rule_frozen":
        raise ValueError("Stage60 freeze did not complete")
    if stage60_audit["status"] != "stage60_ppard_transferred_qubo_independent_audit_ok":
        raise ValueError("Stage60 audit did not pass")
    if freeze["status"] != "stage61a_ppard_remaining144_manifest_frozen":
        raise ValueError("Stage61a source manifest was not frozen")
    if preparation["status"] != "stage61a_ppard_remaining144_unidock_inputs_ok":
        raise ValueError("Stage61a preparation did not pass")

    source_path = verify_descriptor(root, freeze["output"])
    prepared_path = verify_descriptor(root, preparation["output"])
    if sha256(source_path) != str(preparation["source_manifest"]["sha256"]).upper():
        raise ValueError("Stage61a source manifest differs")
    source = read_csv(source_path)
    prepared = read_csv(prepared_path)
    if len(source) != 144 or len(prepared) != 144:
        raise ValueError("Stage61a Remaining-144 dimensions differ")
    if Counter(row["label"] for row in prepared) != {
        "active": 72,
        "decoy": 72,
    }:
        raise ValueError("Stage61a label balance differs")
    if {row["split"] for row in prepared} != {"train"} or {
        row["selection_role"] for row in prepared
    } != {"development_train"}:
        raise ValueError("Stage61a crossed a protected split boundary")
    if {row["pilot_selected"] for row in prepared} != {"False"} or {
        row["pilot_role"] for row in prepared
    } != {""}:
        raise ValueError("Stage61a contains a Pilot-96 row")
    source_columns = list(source[0])
    if any(
        any(frozen.get(key) != realized.get(key) for key in source_columns)
        for frozen, realized in zip(source, prepared, strict=True)
    ):
        raise ValueError("Stage61a changed the frozen source order or identity")

    variants: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    verified_file_count = 0
    for row in prepared:
        for path_column, hash_column in (
            ("sdf_path", "sdf_sha256"),
            ("pdbqt_path", "pdbqt_sha256"),
        ):
            path = root / row[path_column]
            if not path.is_file() or sha256(path) != row[hash_column].upper():
                raise ValueError(f"Stage61a prepared file differs: {row['ligand_id']}")
            verified_file_count += 1
        pdbqt = root / row["pdbqt_path"]
        if macrocycle_closure_atom_types(pdbqt):
            raise ValueError(f"Stage61a closure pseudoatom remains: {row['ligand_id']}")
        lines = pdbqt.read_text(encoding="utf-8", errors="replace").splitlines()
        atom_count = sum(line.startswith(("ATOM", "HETATM")) for line in lines)
        torsdof = [line.split()[-1] for line in lines if line.startswith("TORSDOF")]
        if atom_count != int(row["pdbqt_atom_count"]) or torsdof != [row["torsdof"]]:
            raise ValueError(f"Stage61a PDBQT audit differs: {row['ligand_id']}")
        if row["prep_status"] == "warning":
            warnings[row["prep_message"]] += 1
        elif row["prep_status"] != "ok":
            raise ValueError(f"Stage61a preparation failed: {row['ligand_id']}")
        variants[row["preparation_variant"]] += 1

    audit = {
        "schema_version": "1.0",
        "audit_id": "stage61a-ppard-remaining144-independent-input-audit-v1",
        "status": "independent_stage61a_ppard_remaining144_input_audit_ok",
        "experiment_class": "prospective_remaining_development_train",
        "ligand_count": len(prepared),
        "label_counts": dict(sorted(Counter(row["label"] for row in prepared).items())),
        "pilot_overlap_count": 0,
        "verified_prepared_file_count": verified_file_count,
        "preparation_variant_counts": dict(sorted(variants.items())),
        "sdf_warning_counts": dict(sorted(warnings.items())),
        "failed_ligand_count": 0,
        "macrocycle_closure_pseudoatom_ligand_count": 0,
        "future_receptor_count": 29,
        "future_seed_count": 3,
        "future_pair_count": 12528,
        "data_boundary": {
            "development_rows_prepared": 144,
            "docking_scores_read": 0,
            "fresh_validation_structures_prepared": 0,
            "locked_test_structures_prepared": 0,
        },
        "next_gate": "dock Remaining-144 against all 29 frozen receptors with three seeds, then combine with Pilot-96",
        "decision_boundary": "Input integrity only; no QUBO efficacy, classical superiority, independent validation, or quantum claim is established.",
    }
    output = output if output.is_absolute() else root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage61a_ppard_remaining144_unidock_input_audit.json"),
    )
    args = parser.parse_args()
    run(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
