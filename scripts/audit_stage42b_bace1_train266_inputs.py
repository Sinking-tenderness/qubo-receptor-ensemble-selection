"""Independently audit the frozen Stage 42 BACE1 Train-266 inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
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
        raise ValueError(f"Stage 42b descriptor differs: {path}")
    return path


def run(root: Path, output: Path) -> dict[str, object]:
    root = root.resolve()
    preregistration = read_json(
        root / "configs/stage42_bace1_redocking_qualified_development_preregistration.json"
    )
    allocation = read_json(root / "data/stage42a_bace1_ligand_panel_allocation_summary.json")
    preparation = read_json(root / "data/stage42b_bace1_train266_unidock_input_summary.json")
    if preregistration["preregistration_id"] != "stage42-bace1-redocking-qualified-development-20260804-v1":
        raise ValueError("Stage 42 preregistration differs")
    if allocation["status"] != "stage42a_bace1_ligand_panels_frozen":
        raise ValueError("Stage 42a allocation did not pass")
    if preparation["status"] != "stage42b_bace1_train266_unidock_inputs_ok":
        raise ValueError("Stage 42b preparation did not pass")

    selected_path = verify_descriptor(
        root, dict(allocation["outputs"])["selected_panel_manifest_csv"]
    )
    train_path = verify_descriptor(root, dict(allocation["outputs"])["train_manifest_csv"])
    prepared_path = verify_descriptor(root, dict(preparation["output"]))
    if sha256(train_path) != str(preparation["source_manifest"]["sha256"]).upper():
        raise ValueError("Stage 42b source manifest differs")

    selected = read_csv(selected_path)
    train = read_csv(train_path)
    prepared = read_csv(prepared_path)
    roles = Counter(row["selection_role"] for row in selected)
    expected_roles = Counter(
        {"development_train": 266, "fresh_validation": 1576, "locked_test": 1576}
    )
    if roles != expected_roles or len(selected) != 3418:
        raise ValueError("Stage 42 selected-panel dimensions differ")
    if len(train) != 266 or len(prepared) != 266:
        raise ValueError("Stage 42 Train-266 dimensions differ")
    labels = Counter(row["label"] for row in prepared)
    if labels != Counter({"active": 133, "decoy": 133}):
        raise ValueError("Stage 42b label balance differs")
    if {row["split"] for row in prepared} != {"train"} or {
        row["selection_role"] for row in prepared
    } != {"development_train"}:
        raise ValueError("Stage 42b crossed a protected split boundary")
    if len({row["ligand_id"] for row in prepared}) != 266:
        raise ValueError("Stage 42b ligand IDs are not unique")
    source_columns = list(train[0])
    if any(
        any(source.get(key) != realized.get(key) for key in source_columns)
        for source, realized in zip(train, prepared, strict=True)
    ):
        raise ValueError("Stage 42b changed the frozen Train-266 order or identity")

    rows_by_role: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        rows_by_role[row["selection_role"]].append(row)
    disjoint_columns = ("ligand_id", "canonical_smiles", "scaffold_smiles", "split_group_id")
    for column in disjoint_columns:
        role_sets = {
            role: {row[column] for row in rows}
            for role, rows in rows_by_role.items()
        }
        names = list(role_sets)
        if any(
            role_sets[first] & role_sets[second]
            for index, first in enumerate(names)
            for second in names[index + 1 :]
        ):
            raise ValueError(f"Stage 42 role leakage detected: {column}")

    warning_messages: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    verified_file_count = 0
    for row in prepared:
        for path_column, hash_column in (
            ("sdf_path", "sdf_sha256"),
            ("pdbqt_path", "pdbqt_sha256"),
        ):
            path = root / row[path_column]
            if not path.is_file() or sha256(path) != row[hash_column].upper():
                raise ValueError(f"Stage 42b prepared file differs: {row['ligand_id']}")
            verified_file_count += 1
        pdbqt = root / row["pdbqt_path"]
        if macrocycle_closure_atom_types(pdbqt):
            raise ValueError(f"Stage 42b closure pseudoatom remains: {row['ligand_id']}")
        lines = pdbqt.read_text(encoding="utf-8", errors="replace").splitlines()
        atom_count = sum(line.startswith(("ATOM", "HETATM")) for line in lines)
        torsdof = [line.split()[-1] for line in lines if line.startswith("TORSDOF")]
        if atom_count != int(row["pdbqt_atom_count"]) or torsdof != [row["torsdof"]]:
            raise ValueError(f"Stage 42b PDBQT structure differs: {row['ligand_id']}")
        if row["prep_status"] == "warning":
            warning_messages[row["prep_message"]] += 1
        elif row["prep_status"] != "ok":
            raise ValueError(f"Stage 42b preparation status differs: {row['ligand_id']}")
        variants[row["preparation_variant"]] += 1

    if warning_messages != Counter({"MMFF94_not_converged_code_1": 22}):
        raise ValueError("Stage 42b warning distribution differs")
    if variants != Counter({"meeko_flexible": 242, "meeko_rigid_macrocycles": 24}):
        raise ValueError("Stage 42b preparation variants differ")

    result = {
        "schema_version": "1.0",
        "audit_id": "stage42b-bace1-train266-independent-input-audit-v1",
        "status": "independent_stage42b_bace1_train266_input_audit_ok",
        "experiment_class": "outcome-informed_posthoc_development",
        "receptor_count_frozen_for_next_stage": 34,
        "selected_panel_count": len(selected),
        "ligand_count": len(prepared),
        "label_counts": dict(sorted(labels.items())),
        "verified_prepared_file_count": verified_file_count,
        "preparation_variant_counts": dict(sorted(variants.items())),
        "sdf_warning_counts": dict(sorted(warning_messages.items())),
        "failed_ligand_count": 0,
        "macrocycle_closure_pseudoatom_ligand_count": 0,
        "protected_role_disjointness": {column: True for column in disjoint_columns},
        "data_boundary": {
            "docking_scores_read": 0,
            "fresh_validation_structures_prepared": 0,
            "locked_test_structures_prepared": 0,
        },
        "inputs": {
            "allocation_summary": {
                "path": "data/stage42a_bace1_ligand_panel_allocation_summary.json",
                "sha256": sha256(root / "data/stage42a_bace1_ligand_panel_allocation_summary.json"),
            },
            "preparation_summary": {
                "path": "data/stage42b_bace1_train266_unidock_input_summary.json",
                "sha256": sha256(root / "data/stage42b_bace1_train266_unidock_input_summary.json"),
            },
            "prepared_manifest": {
                "path": "data/processed/stage42b_bace1_train266_unidock_pdbqt_manifest.csv",
                "sha256": sha256(prepared_path),
            },
        },
        "next_gate": "run the complete 34-receptor by 266-ligand by three-seed Uni-Dock development matrix",
        "decision_boundary": "This audit establishes input integrity only; it does not establish enrichment, QUBO superiority, quantum execution, or quantum advantage.",
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
        default=Path("data/stage42b_bace1_train266_unidock_input_independent_audit.json"),
    )
    args = parser.parse_args()
    run(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
