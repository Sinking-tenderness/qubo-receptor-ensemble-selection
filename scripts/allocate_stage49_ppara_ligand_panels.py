"""Freeze molecule- and scaffold-disjoint PPARA ligand panels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def verified(root: Path, record: dict[str, Any]) -> Path:
    path = root / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"].upper():
        raise ValueError(f"Stage49 input identity differs: {path}")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def scaffold_for(molecule: Chem.Mol) -> str:
    copy = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(copy)
    for bond in copy.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    scaffold = MurckoScaffold.GetScaffoldForMol(copy)
    Chem.RemoveStereochemistry(scaffold)
    for bond in scaffold.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    if scaffold.GetNumAtoms() == 0:
        scaffold = copy
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def read_ism(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if len(fields) < 2:
                raise ValueError(f"PPARA {label} row is incomplete: {line_number}")
            molecule = Chem.MolFromSmiles(fields[0])
            if molecule is None:
                raise ValueError(f"PPARA {label} SMILES does not parse: {line_number}")
            rows.append(
                {
                    "source_line_number": line_number,
                    "source_molecule_id": fields[1],
                    "source_smiles": fields[0],
                    "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
                    "scaffold_smiles": scaffold_for(molecule),
                    "label": label,
                }
            )
    return rows


def connected_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for key in ("source_molecule_id", "canonical_smiles", "scaffold_smiles"):
        first_by_value: dict[str, int] = {}
        for index, row in enumerate(rows):
            value = str(row[key])
            if value in first_by_value:
                union(index, first_by_value[value])
            else:
                first_by_value[value] = index
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    return [groups[key] for key in sorted(groups)]


def group_id(group: list[dict[str, Any]]) -> str:
    label = str(group[0]["label"]).upper()
    identities = sorted(f"{row['source_line_number']:06d}|{row['source_molecule_id']}" for row in group)
    digest = hashlib.sha256("\n".join(identities).encode("ascii")).hexdigest()[:20].upper()
    return f"PPARA_{label}_G{digest}"


def exact_subset(
    groups: list[list[dict[str, Any]]], target: int, seed: str
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    ranked = sorted(
        groups,
        key=lambda group: (
            hashlib.sha256(f"{seed}|{group_id(group)}".encode("ascii")).hexdigest(),
            group_id(group),
        ),
    )
    parent: dict[int, tuple[int, int] | None] = {0: None}
    for index, group in enumerate(ranked):
        size = len(group)
        for current in sorted(list(parent), reverse=True):
            new = current + size
            if new <= target and new not in parent:
                parent[new] = (current, index)
        if target in parent:
            break
    if target not in parent:
        raise ValueError(f"cannot construct exact PPARA group panel of size {target}")
    chosen_indices: set[int] = set()
    current = target
    while current:
        previous, index = parent[current]  # type: ignore[misc]
        chosen_indices.add(index)
        current = previous
    selected = [group for index, group in enumerate(ranked) if index in chosen_indices]
    selected_ids = {group_id(group) for group in selected}
    remaining = [group for group in groups if group_id(group) not in selected_ids]
    return selected, remaining


def assign_rows(
    selected_by_role: dict[str, list[list[dict[str, Any]]]],
    seeds: dict[str, str],
) -> list[dict[str, Any]]:
    split_by_role = {"development_train": "train", "fresh_validation": "validation", "locked_test": "test"}
    output: list[dict[str, Any]] = []
    for role in ("development_train", "fresh_validation", "locked_test"):
        for group in selected_by_role[role]:
            identity = group_id(group)
            rank_hash = hashlib.sha256(f"{seeds[role]}|{identity}".encode("ascii")).hexdigest()
            for row in group:
                label = row["label"]
                output.append(
                    {
                        **row,
                        "ligand_id": f"PPARA_{label}_L{row['source_line_number']:06d}",
                        "smiles": row["source_smiles"],
                        "source": "DUD-E",
                        "target_id": "PPARA",
                        "split_group_id": identity,
                        "allocation_rank_sha256": rank_hash,
                        "selection_role": role,
                        "split": split_by_role[role],
                    }
                )
    return output


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage49 implementation path differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    source = read_json(inputs["source_audit"])
    audit = read_json(inputs["source_independent_audit"])
    if source.get("status") != "stage48_ppara_source_audit_ok" or audit.get("status") != "stage48_ppara_source_independent_audit_ok":
        raise ValueError("Stage48 source evidence did not pass")

    active_rows = read_ism(inputs["actives"], "active")
    decoy_rows = read_ism(inputs["decoys"], "decoy")
    expected = config["expected"]
    if len(active_rows) != expected["source_active_count"] or len(decoy_rows) != expected["source_decoy_count"]:
        raise ValueError("PPARA source row counts differ")
    active_groups = connected_groups(active_rows)
    decoy_groups = connected_groups(decoy_rows)
    active_scaffolds = {row["scaffold_smiles"] for row in active_rows}
    collision_groups = [group for group in decoy_groups if any(row["scaffold_smiles"] in active_scaffolds for row in group)]
    collision_ids = {group_id(group) for group in collision_groups}
    eligible_decoy_groups = [group for group in decoy_groups if group_id(group) not in collision_ids]

    seeds = config["allocation_seeds"]
    active_validation, active_remaining = exact_subset(active_groups, expected["fresh_validation_active_count"], seeds["active_fresh_validation"])
    active_test, active_train = exact_subset(active_remaining, expected["locked_test_active_count"], seeds["active_locked_test"])
    if sum(map(len, active_train)) != expected["train_active_count"]:
        raise ValueError("PPARA active train remainder differs")
    active_selected = {
        "development_train": active_train,
        "fresh_validation": active_validation,
        "locked_test": active_test,
    }
    active_seed_by_role = {
        "development_train": seeds["active_train_remainder"],
        "fresh_validation": seeds["active_fresh_validation"],
        "locked_test": seeds["active_locked_test"],
    }

    decoy_train, remaining = exact_subset(eligible_decoy_groups, expected["train_decoy_count"], seeds["decoy_train"])
    decoy_validation, remaining = exact_subset(remaining, expected["fresh_validation_decoy_count"], seeds["decoy_fresh_validation"])
    decoy_test, remaining = exact_subset(remaining, expected["locked_test_decoy_count"], seeds["decoy_locked_test"])
    decoy_selected = {
        "development_train": decoy_train,
        "fresh_validation": decoy_validation,
        "locked_test": decoy_test,
    }
    decoy_seed_by_role = {
        "development_train": seeds["decoy_train"],
        "fresh_validation": seeds["decoy_fresh_validation"],
        "locked_test": seeds["decoy_locked_test"],
    }
    rows = assign_rows(active_selected, active_seed_by_role) + assign_rows(decoy_selected, decoy_seed_by_role)
    rows.sort(key=lambda row: ({"train": 0, "validation": 1, "test": 2}[row["split"]], {"active": 0, "decoy": 1}[row["label"]], row["source_line_number"]))

    observed = Counter((row["split"], row["label"]) for row in rows)
    required = Counter({
        ("train", "active"): expected["train_active_count"],
        ("train", "decoy"): expected["train_decoy_count"],
        ("validation", "active"): expected["fresh_validation_active_count"],
        ("validation", "decoy"): expected["fresh_validation_decoy_count"],
        ("test", "active"): expected["locked_test_active_count"],
        ("test", "decoy"): expected["locked_test_decoy_count"],
    })
    if observed != required:
        raise ValueError(f"PPARA panel counts differ: {observed}")
    for key in ("ligand_id", "source_molecule_id", "canonical_smiles", "scaffold_smiles", "split_group_id"):
        split_sets: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            split_sets[str(row[key])].add(row["split"])
        if any(len(values) > 1 for values in split_sets.values()):
            raise ValueError(f"PPARA {key} crosses a panel boundary")

    outputs = {key: root / value for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage49 outputs exist; pass --overwrite")
    write_csv(outputs["selected_panel_manifest_csv"], rows)
    write_csv(outputs["train_manifest_csv"], [row for row in rows if row["split"] == "train"])
    write_csv(outputs["fresh_validation_manifest_csv"], [row for row in rows if row["split"] == "validation"])
    write_csv(outputs["locked_test_manifest_csv"], [row for row in rows if row["split"] == "test"])
    summary = {
        "schema_version": "1.0",
        "status": "stage49_ppara_ligand_panels_frozen",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": sha256(config_path)},
        "source_counts": {"active": len(active_rows), "decoy": len(decoy_rows)},
        "group_audit": {
            "active_group_count": len(active_groups),
            "maximum_active_group_size": max(map(len, active_groups)),
            "decoy_group_count": len(decoy_groups),
            "maximum_decoy_group_size": max(map(len, decoy_groups)),
            "active_scaffold_collision_decoy_group_count": len(collision_groups),
            "active_scaffold_collision_decoy_row_count": sum(map(len, collision_groups)),
            "eligible_decoy_group_count": len(eligible_decoy_groups),
            "eligible_decoy_row_count": sum(map(len, eligible_decoy_groups)),
            "unallocated_eligible_decoy_row_count": sum(map(len, remaining)),
        },
        "selected_counts": {f"{split}_{label}": count for (split, label), count in sorted(observed.items())},
        "disjointness": {
            "ligand_id_disjoint": True,
            "source_molecule_id_disjoint": True,
            "canonical_smiles_disjoint": True,
            "bemis_murcko_scaffold_disjoint": True,
            "split_group_disjoint": True,
        },
        "data_boundary": {
            "labels_used_only_for_frozen_allocation": True,
            "docking_scores_read": 0,
            "fresh_validation_docking_scores_read": 0,
            "locked_test_docking_scores_read": 0,
            "new_docking_jobs": 0,
        },
        "decision": {
            "ligand_allocation_gate_passed": True,
            "train_ligand_preparation_authorized": True,
            "fresh_validation_release_authorized": False,
            "locked_test_release_authorized": False,
        },
        "outputs": {
            key: {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
            for key, path in outputs.items()
            if key != "summary_json"
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    outputs["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    outputs["summary_json"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.root / args.config
    run(config_path, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
