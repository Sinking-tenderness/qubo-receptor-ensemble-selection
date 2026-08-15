"""Independently audit the frozen Stage58a PPARD Pilot-96 inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
    macrocycle_closure_atom_types,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def verify_descriptor(root: Path, descriptor: dict[str, object]) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file() or sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage58a descriptor differs: {path}")
    return path


def run(root: Path, output: Path) -> dict[str, object]:
    root = root.resolve()
    allocation = read_json(root / "data/stage56_ppard_ligand_panel_allocation_summary.json")
    preparation = read_json(root / "data/stage58a_ppard_pilot96_unidock_input_summary.json")
    stage57 = read_json(root / "data/stage57_ppard_cognate_redocking_summary.json")
    if allocation["status"] != "stage56_ppard_ligand_panels_and_pilot_frozen":
        raise ValueError("Stage56 PPARD allocation did not pass")
    if preparation["status"] != "stage58a_ppard_pilot96_unidock_inputs_ok":
        raise ValueError("Stage58a PPARD preparation did not pass")
    if stage57["status"] != "stage57_ppard_cognate_redocking_gate_ok":
        raise ValueError("Stage57 PPARD redocking gate did not pass")
    if int(stage57["passed_receptor_count"]) != 29:
        raise ValueError("Stage57 passing-receptor count differs")

    source_path = verify_descriptor(root, dict(allocation["outputs"])["pilot_manifest_csv"])
    prepared_path = verify_descriptor(root, dict(preparation["output"]))
    if sha256(source_path) != str(preparation["source_manifest"]["sha256"]).upper():
        raise ValueError("Stage58a source manifest differs")
    source = read_csv(source_path)
    prepared = read_csv(prepared_path)
    if len(source) != 96 or len(prepared) != 96:
        raise ValueError("Stage58a Pilot-96 dimensions differ")
    if Counter(row["label"] for row in prepared) != Counter(
        {"active": 48, "decoy": 48}
    ):
        raise ValueError("Stage58a label balance differs")
    if {row["split"] for row in prepared} != {"train"} or {
        row["selection_role"] for row in prepared
    } != {"development_train"}:
        raise ValueError("Stage58a crossed a protected split boundary")
    if {row["pilot_selected"] for row in prepared} != {"True"} or {
        row["pilot_role"] for row in prepared
    } != {"development_train_pilot"}:
        raise ValueError("Stage58a pilot role differs")
    fold_counts = Counter(
        (row["pilot_outer_fold"], row["label"]) for row in prepared
    )
    if fold_counts != Counter(
        {(str(fold), label): 12 for fold in range(4) for label in ("active", "decoy")}
    ):
        raise ValueError("Stage58a four-fold balance differs")
    if len({row["ligand_id"] for row in prepared}) != 96:
        raise ValueError("Stage58a ligand IDs are not unique")
    source_columns = list(source[0])
    if any(
        any(frozen.get(key) != realized.get(key) for key in source_columns)
        for frozen, realized in zip(source, prepared, strict=True)
    ):
        raise ValueError("Stage58a changed the frozen Pilot-96 order or identity")

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
                raise ValueError(f"Stage58a prepared file differs: {row['ligand_id']}")
            verified_file_count += 1
        pdbqt = root / row["pdbqt_path"]
        if macrocycle_closure_atom_types(pdbqt):
            raise ValueError(f"Stage58a closure pseudoatom remains: {row['ligand_id']}")
        lines = pdbqt.read_text(encoding="utf-8", errors="replace").splitlines()
        atom_count = sum(line.startswith(("ATOM", "HETATM")) for line in lines)
        torsdof = [line.split()[-1] for line in lines if line.startswith("TORSDOF")]
        if atom_count != int(row["pdbqt_atom_count"]) or torsdof != [row["torsdof"]]:
            raise ValueError(f"Stage58a PDBQT audit differs: {row['ligand_id']}")
        if row["prep_status"] == "warning":
            warnings[row["prep_message"]] += 1
        elif row["prep_status"] != "ok":
            raise ValueError(f"Stage58a preparation failed: {row['ligand_id']}")
        variants[row["preparation_variant"]] += 1

    result = {
        "schema_version": "1.0",
        "audit_id": "stage58a-ppard-pilot96-independent-input-audit-v1",
        "status": "independent_stage58a_ppard_pilot96_input_audit_ok",
        "experiment_class": "prospective_outcome_blind_development_pilot",
        "receptor_count_frozen_for_next_stage": 29,
        "ligand_count": len(prepared),
        "label_counts": dict(sorted(Counter(row["label"] for row in prepared).items())),
        "fold_label_counts": {
            f"fold{fold}_{label}": fold_counts[(str(fold), label)]
            for fold in range(4)
            for label in ("active", "decoy")
        },
        "verified_prepared_file_count": verified_file_count,
        "preparation_variant_counts": dict(sorted(variants.items())),
        "sdf_warning_counts": dict(sorted(warnings.items())),
        "failed_ligand_count": 0,
        "macrocycle_closure_pseudoatom_ligand_count": 0,
        "data_boundary": {
            "docking_scores_read": 0,
            "fresh_validation_structures_prepared": 0,
            "locked_test_structures_prepared": 0,
        },
        "inputs": {
            "allocation_summary": {
                "path": "data/stage56_ppard_ligand_panel_allocation_summary.json",
                "sha256": sha256(root / "data/stage56_ppard_ligand_panel_allocation_summary.json"),
            },
            "stage57_summary": {
                "path": "data/stage57_ppard_cognate_redocking_summary.json",
                "sha256": sha256(root / "data/stage57_ppard_cognate_redocking_summary.json"),
            },
            "prepared_manifest": {
                "path": "data/processed/stage58a_ppard_pilot96_unidock_pdbqt_manifest.csv",
                "sha256": sha256(prepared_path),
            },
        },
        "next_gate": "dock Pilot-96 against all 29 Stage57-passing receptors with three frozen seeds",
        "decision_boundary": "This audit establishes input integrity only; it does not establish enrichment, functional complementarity, QUBO superiority, quantum execution, or quantum advantage.",
    }
    output = output if output.is_absolute() else root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage58a_ppard_pilot96_unidock_input_independent_audit.json"),
    )
    args = parser.parse_args()
    run(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
