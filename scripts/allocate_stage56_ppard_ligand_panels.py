"""Freeze PPARD panels and the outcome-blind Stage 56 pilot subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def verified(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage 56 input identity differs: {path}")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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
                raise ValueError(f"PPARD {label} row is incomplete: {line_number}")
            molecule = Chem.MolFromSmiles(fields[0])
            if molecule is None:
                raise ValueError(f"PPARD {label} SMILES does not parse: {line_number}")
            rows.append(
                {
                    "source_line_number": line_number,
                    "source_molecule_id": fields[1],
                    "source_smiles": fields[0],
                    "canonical_smiles": Chem.MolToSmiles(
                        molecule, canonical=True, isomericSmiles=True
                    ),
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
    identities = sorted(
        f"{int(row['source_line_number']):06d}|{row['source_molecule_id']}"
        for row in group
    )
    digest = hashlib.sha256("\n".join(identities).encode("ascii")).hexdigest()[:20]
    return f"PPARD_{label}_G{digest.upper()}"


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
        raise ValueError(f"cannot construct exact PPARD group panel of size {target}")
    chosen: set[int] = set()
    current = target
    while current:
        previous, index = parent[current]  # type: ignore[misc]
        chosen.add(index)
        current = previous
    selected = [group for index, group in enumerate(ranked) if index in chosen]
    selected_ids = {group_id(group) for group in selected}
    remaining = [group for group in groups if group_id(group) not in selected_ids]
    return selected, remaining


def assign_rows(
    selected_by_role: dict[str, list[list[dict[str, Any]]]],
    seeds: dict[str, str],
) -> list[dict[str, Any]]:
    split_by_role = {
        "development_train": "train",
        "fresh_validation": "validation",
        "locked_test": "test",
    }
    output: list[dict[str, Any]] = []
    for role in ("development_train", "fresh_validation", "locked_test"):
        for group in selected_by_role[role]:
            identity = group_id(group)
            rank_hash = hashlib.sha256(
                f"{seeds[role]}|{identity}".encode("ascii")
            ).hexdigest()
            for row in group:
                label = str(row["label"])
                output.append(
                    {
                        **row,
                        "ligand_id": f"PPARD_{label}_L{int(row['source_line_number']):06d}",
                        "smiles": row["source_smiles"],
                        "source": "DUD-E",
                        "target_id": "PPARD",
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
        raise ValueError("Stage 56 allocation implementation differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    source = read_json(inputs["stage55_result"])
    audit = read_json(inputs["stage55_audit"])
    if source["status"] != "stage55_ppard_small_pilot_source_and_preregistration_ok":
        raise ValueError("Stage55 source gate did not pass")
    if audit["status"] != "stage55_ppard_small_pilot_independent_audit_ok":
        raise ValueError("Stage55 independent audit did not pass")

    active_rows = read_ism(inputs["actives"], "active")
    decoy_rows = read_ism(inputs["decoys"], "decoy")
    expected = config["expected"]
    if len(active_rows) != int(expected["source_active_count"]) or len(
        decoy_rows
    ) != int(expected["source_decoy_count"]):
        raise ValueError("PPARD source row counts differ")
    active_groups = connected_groups(active_rows)
    decoy_groups = connected_groups(decoy_rows)
    active_scaffolds = {str(row["scaffold_smiles"]) for row in active_rows}
    collision_groups = [
        group
        for group in decoy_groups
        if any(str(row["scaffold_smiles"]) in active_scaffolds for row in group)
    ]
    collision_ids = {group_id(group) for group in collision_groups}
    eligible_decoy_groups = [
        group for group in decoy_groups if group_id(group) not in collision_ids
    ]

    seeds = config["allocation_seeds"]
    active_validation, active_remaining = exact_subset(
        active_groups,
        int(expected["fresh_validation_active_count"]),
        seeds["active_fresh_validation"],
    )
    active_test, active_train = exact_subset(
        active_remaining,
        int(expected["locked_test_active_count"]),
        seeds["active_locked_test"],
    )
    if sum(map(len, active_train)) != int(expected["train_active_count"]):
        raise ValueError("PPARD active train remainder differs")
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

    decoy_train, remaining_decoys = exact_subset(
        eligible_decoy_groups,
        int(expected["train_decoy_count"]),
        seeds["decoy_train"],
    )
    decoy_validation, remaining_decoys = exact_subset(
        remaining_decoys,
        int(expected["fresh_validation_decoy_count"]),
        seeds["decoy_fresh_validation"],
    )
    decoy_test, remaining_decoys = exact_subset(
        remaining_decoys,
        int(expected["locked_test_decoy_count"]),
        seeds["decoy_locked_test"],
    )
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

    pilot_group_fold: dict[str, int] = {}
    for label, train_groups in (("active", active_train), ("decoy", decoy_train)):
        available = list(train_groups)
        for fold in range(int(expected["pilot_outer_fold_count"])):
            selected, available = exact_subset(
                available,
                int(expected[f"pilot_{label}_count_per_fold"]),
                str(seeds[f"pilot_{label}_fold{fold}"]),
            )
            for group in selected:
                pilot_group_fold[group_id(group)] = fold

    rows = assign_rows(active_selected, active_seed_by_role) + assign_rows(
        decoy_selected, decoy_seed_by_role
    )
    for row in rows:
        fold = pilot_group_fold.get(str(row["split_group_id"]))
        row["pilot_selected"] = fold is not None
        row["pilot_outer_fold"] = "" if fold is None else fold
        row["pilot_role"] = "development_train_pilot" if fold is not None else ""
    rows.sort(
        key=lambda row: (
            {"train": 0, "validation": 1, "test": 2}[str(row["split"])],
            {"active": 0, "decoy": 1}[str(row["label"])],
            int(row["source_line_number"]),
        )
    )

    observed = Counter((row["split"], row["label"]) for row in rows)
    required = Counter(
        {
            ("train", "active"): int(expected["train_active_count"]),
            ("train", "decoy"): int(expected["train_decoy_count"]),
            ("validation", "active"): int(expected["fresh_validation_active_count"]),
            ("validation", "decoy"): int(expected["fresh_validation_decoy_count"]),
            ("test", "active"): int(expected["locked_test_active_count"]),
            ("test", "decoy"): int(expected["locked_test_decoy_count"]),
        }
    )
    if observed != required:
        raise ValueError(f"PPARD panel counts differ: {observed}")
    for key in (
        "ligand_id",
        "source_molecule_id",
        "canonical_smiles",
        "scaffold_smiles",
        "split_group_id",
    ):
        split_sets: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            split_sets[str(row[key])].add(str(row["split"]))
        if any(len(values) > 1 for values in split_sets.values()):
            raise ValueError(f"PPARD {key} crosses a panel boundary")
    pilot_rows = [row for row in rows if row["pilot_selected"]]
    if any(row["split"] != "train" for row in pilot_rows):
        raise ValueError("PPARD pilot includes a protected-panel row")
    pilot_counts = Counter(
        (int(row["pilot_outer_fold"]), str(row["label"])) for row in pilot_rows
    )
    expected_pilot_counts = Counter(
        {
            (fold, label): int(expected[f"pilot_{label}_count_per_fold"])
            for fold in range(int(expected["pilot_outer_fold_count"]))
            for label in ("active", "decoy")
        }
    )
    if pilot_counts != expected_pilot_counts:
        raise ValueError("PPARD pilot fold counts differ")

    outputs = {key: root / value for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage56 allocation outputs exist; pass --overwrite")
    write_csv(outputs["selected_panel_manifest_csv"], rows)
    write_csv(outputs["train_manifest_csv"], [row for row in rows if row["split"] == "train"])
    write_csv(outputs["pilot_manifest_csv"], pilot_rows)
    write_csv(
        outputs["pilot_fold_assignments_csv"],
        [
            {
                "ligand_id": row["ligand_id"],
                "label": row["label"],
                "scaffold_smiles": row["scaffold_smiles"],
                "split_group_id": row["split_group_id"],
                "outer_fold": row["pilot_outer_fold"],
            }
            for row in pilot_rows
        ],
    )
    write_csv(
        outputs["fresh_validation_manifest_csv"],
        [row for row in rows if row["split"] == "validation"],
    )
    write_csv(
        outputs["locked_test_manifest_csv"],
        [row for row in rows if row["split"] == "test"],
    )
    summary = {
        "schema_version": "1.0",
        "status": "stage56_ppard_ligand_panels_and_pilot_frozen",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": sha256(config_path),
        },
        "source_counts": {"active": len(active_rows), "decoy": len(decoy_rows)},
        "group_audit": {
            "active_group_count": len(active_groups),
            "maximum_active_group_size": max(map(len, active_groups)),
            "decoy_group_count": len(decoy_groups),
            "maximum_decoy_group_size": max(map(len, decoy_groups)),
            "active_scaffold_collision_decoy_group_count": len(collision_groups),
            "active_scaffold_collision_decoy_row_count": sum(
                map(len, collision_groups)
            ),
            "eligible_decoy_group_count": len(eligible_decoy_groups),
            "eligible_decoy_row_count": sum(map(len, eligible_decoy_groups)),
            "unallocated_eligible_decoy_row_count": sum(
                map(len, remaining_decoys)
            ),
        },
        "selected_counts": {
            f"{split}_{label}": count
            for (split, label), count in sorted(observed.items())
        },
        "pilot": {
            "row_count": len(pilot_rows),
            "active_count": sum(row["label"] == "active" for row in pilot_rows),
            "decoy_count": sum(row["label"] == "decoy" for row in pilot_rows),
            "outer_fold_count": int(expected["pilot_outer_fold_count"]),
            "fold_label_counts": {
                f"fold{fold}_{label}": pilot_counts[(fold, label)]
                for fold in range(int(expected["pilot_outer_fold_count"]))
                for label in ("active", "decoy")
            },
        },
        "disjointness": {
            "ligand_id_disjoint": True,
            "source_molecule_id_disjoint": True,
            "canonical_smiles_disjoint": True,
            "bemis_murcko_scaffold_disjoint": True,
            "split_group_disjoint": True,
            "pilot_rows_development_train_only": True,
            "pilot_scaffold_groups_fold_disjoint": True,
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
            "pilot_ligand_preparation_authorized": True,
            "coordinate_audit_authorized": True,
            "pilot_production_docking_authorized": False,
            "full_training_matrix_authorized": False,
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
    outputs["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
