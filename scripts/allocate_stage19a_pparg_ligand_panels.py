"""Freeze scaffold-disjoint PPARG train, validation, and test ligand panels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem

try:
    from scripts.audit_stage18a_pparg_source import scaffold_for
    from scripts.select_stage13_egfr_coordinate_pool import (
        file_sha256,
        read_csv,
        read_json,
        write_csv,
        write_json,
    )
except ModuleNotFoundError:
    from audit_stage18a_pparg_source import scaffold_for
    from select_stage13_egfr_coordinate_pool import (
        file_sha256,
        read_csv,
        read_json,
        write_csv,
        write_json,
    )


def verified(root: Path, descriptor: dict[str, object]) -> Path:
    path = Path(str(descriptor["path"]))
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def identifier_sha256(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def read_decoys(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if len(fields) < 2:
                raise ValueError(f"PPARG decoy row is incomplete: {line_number}")
            molecule = Chem.MolFromSmiles(fields[0])
            if molecule is None:
                raise ValueError(f"PPARG decoy SMILES does not parse: {line_number}")
            rows.append({
                "source_line_number": line_number,
                "source_molecule_id": fields[1],
                "source_smiles": fields[0],
                "canonical_smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
                "scaffold_smiles": scaffold_for(molecule),
            })
    return rows


def connected_decoy_groups(rows: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
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
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    return [groups[key] for key in sorted(groups)]


def group_id(group: list[dict[str, object]]) -> str:
    identities = sorted(
        f"{int(row['source_line_number']):06d}|{row['source_molecule_id']}"
        for row in group
    )
    digest = hashlib.sha256("\n".join(identities).encode("ascii")).hexdigest()[:20]
    return f"PPARG_DECOY_G{digest.upper()}"


def select_exact_groups(
    groups: list[list[dict[str, object]]], target: int, seed: str
) -> tuple[list[list[dict[str, object]]], list[list[dict[str, object]]]]:
    ranked = sorted(
        groups,
        key=lambda group: (
            hashlib.sha256(f"{seed}|{group_id(group)}".encode("ascii")).hexdigest(),
            group_id(group),
        ),
    )
    selected: list[list[dict[str, object]]] = []
    count = 0
    for group in ranked:
        if count + len(group) <= target:
            selected.append(group)
            count += len(group)
        if count == target:
            break
    if count != target:
        raise ValueError(f"cannot construct exact PPARG decoy panel of size {target}")
    selected_ids = {group_id(group) for group in selected}
    remaining = [group for group in groups if group_id(group) not in selected_ids]
    return selected, remaining


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if file_sha256(Path(__file__)) != str(config["implementation"]["sha256"]).upper():
        raise ValueError("Stage 19a implementation SHA-256 differs")
    inputs = {
        key: verified(root, dict(value))
        for key, value in dict(config["inputs"]).items()
    }
    preregistration = read_json(inputs["preregistration"])
    source_audit = read_json(inputs["source_audit"])
    if preregistration["preregistration_id"] != "stage18-pparg-replacement-exploratory-20260801-v1":
        raise ValueError("PPARG preregistration differs")
    if source_audit["status"] != "stage18a_pparg_source_and_active_allocation_ok":
        raise ValueError("PPARG source audit did not pass")
    if int(source_audit["data_boundary"]["docking_scores_read"]) != 0:
        raise ValueError("PPARG source audit read docking scores")

    active_rows = read_csv(inputs["active_allocation"])
    if len(active_rows) != int(config["expected"]["source_active_count"]):
        raise ValueError("PPARG active allocation count differs")
    active_canonical = {row["canonical_smiles"] for row in active_rows}
    active_scaffolds = {row["scaffold_smiles"] for row in active_rows}
    if len(active_scaffolds) != len(active_rows):
        raise ValueError("PPARG active scaffold identity differs")

    decoy_rows = read_decoys(inputs["decoy_ism"])
    if len(decoy_rows) != int(config["expected"]["source_decoy_count"]):
        raise ValueError("PPARG decoy source count differs")
    groups = connected_decoy_groups(decoy_rows)
    collision_groups = [
        group for group in groups
        if any(
            str(row["canonical_smiles"]) in active_canonical
            or str(row["scaffold_smiles"]) in active_scaffolds
            for row in group
        )
    ]
    collision_ids = {group_id(group) for group in collision_groups}
    eligible_groups = [group for group in groups if group_id(group) not in collision_ids]
    group_audit = {
        "connected_decoy_group_count": len(groups),
        "maximum_decoy_group_size": max(len(group) for group in groups),
        "active_identity_collision_group_count": len(collision_groups),
        "active_identity_collision_row_count": sum(len(group) for group in collision_groups),
        "eligible_decoy_group_count": len(eligible_groups),
        "eligible_decoy_row_count": sum(len(group) for group in eligible_groups),
    }
    for key, value in group_audit.items():
        if int(config["expected"][key]) != value:
            raise ValueError(f"PPARG decoy group audit differs: {key}")

    allocation = dict(config["decoy_allocation"])
    role_specs = (
        ("development_train", "train", int(allocation["train_decoy_count"]), str(allocation["train_hash_seed"])),
        ("fresh_validation", "validation", int(allocation["fresh_validation_decoy_count"]), str(allocation["fresh_validation_hash_seed"])),
        ("locked_test", "test", int(allocation["locked_test_decoy_count"]), str(allocation["locked_test_hash_seed"])),
    )
    selected_by_role: dict[str, list[list[dict[str, object]]]] = {}
    remaining = eligible_groups
    for role, _split, count, seed in role_specs:
        selected, remaining = select_exact_groups(remaining, count, seed)
        selected_by_role[role] = selected

    decoy_selected_rows: list[dict[str, object]] = []
    for role, split, _count, _seed in role_specs:
        for group in selected_by_role[role]:
            identity = group_id(group)
            rank_hash = hashlib.sha256(
                f"{_seed}|{identity}".encode("ascii")
            ).hexdigest()
            for row in group:
                decoy_selected_rows.append({
                    **row,
                    "ligand_id": f"PPARG_decoy_L{int(row['source_line_number']):06d}",
                    "smiles": row["source_smiles"],
                    "label": "decoy",
                    "source": "DUD-E",
                    "target_id": "PPARG",
                    "split_group_id": identity,
                    "allocation_rank_sha256": rank_hash,
                    "selection_role": role,
                    "split": split,
                })

    active_selected_rows: list[dict[str, object]] = []
    for row in active_rows:
        split_group = "PPARG_ACTIVE_G" + hashlib.sha256(
            f"{row['source_molecule_id']}|{row['canonical_smiles']}|{row['scaffold_smiles']}".encode("utf-8")
        ).hexdigest()[:20].upper()
        active_selected_rows.append({
            **row,
            "ligand_id": f"PPARG_active_L{int(row['source_line_number']):06d}",
            "smiles": row["source_smiles"],
            "label": "active",
            "source": "DUD-E",
            "target_id": "PPARG",
            "split_group_id": split_group,
        })

    selected_rows = active_selected_rows + decoy_selected_rows
    split_order = {"train": 0, "validation": 1, "test": 2}
    label_order = {"active": 0, "decoy": 1}
    selected_rows.sort(
        key=lambda row: (
            split_order[str(row["split"])],
            label_order[str(row["label"])],
            int(row["source_line_number"]),
        )
    )
    expected_counts = {
        ("train", "active"): int(config["expected"]["train_active_count"]),
        ("train", "decoy"): int(config["expected"]["train_decoy_count"]),
        ("validation", "active"): int(config["expected"]["fresh_validation_active_count"]),
        ("validation", "decoy"): int(config["expected"]["fresh_validation_decoy_count"]),
        ("test", "active"): int(config["expected"]["locked_test_active_count"]),
        ("test", "decoy"): int(config["expected"]["locked_test_decoy_count"]),
    }
    observed_counts = Counter((str(row["split"]), str(row["label"])) for row in selected_rows)
    if observed_counts != Counter(expected_counts):
        raise ValueError("PPARG selected panel counts differ")
    for key in ("ligand_id", "canonical_smiles", "split_group_id"):
        split_sets: dict[str, set[str]] = defaultdict(set)
        for row in selected_rows:
            split_sets[str(row[key])].add(str(row["split"]))
        if any(len(splits) != 1 for splits in split_sets.values()):
            raise ValueError(f"PPARG {key} crosses a panel boundary")
    source_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in selected_rows:
        source_splits[(str(row["label"]), str(row["source_molecule_id"]))].add(str(row["split"]))
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise ValueError("PPARG source molecule ID crosses a panel boundary")
    scaffold_splits: dict[str, set[str]] = defaultdict(set)
    for row in selected_rows:
        scaffold_splits[str(row["scaffold_smiles"])].add(str(row["split"]))
    if any(len(splits) != 1 for splits in scaffold_splits.values()):
        raise ValueError("PPARG Bemis-Murcko scaffold crosses a panel boundary")

    train_rows = [row for row in selected_rows if row["split"] == "train"]
    outputs = dict(config["outputs"])
    selected_path = root / str(outputs["selected_panel_manifest_csv"])
    train_path = root / str(outputs["train_manifest_csv"])
    summary_path = root / str(outputs["summary_json"])
    if not overwrite and any(path.exists() for path in (selected_path, train_path, summary_path)):
        raise FileExistsError("Stage 19a outputs exist; pass --overwrite")
    write_csv(selected_path, selected_rows)
    write_csv(train_path, train_rows)
    role_counts = Counter(str(row["selection_role"]) for row in selected_rows)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage19a_pparg_ligand_panels_frozen",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "source_counts": {"active": len(active_rows), "decoy": len(decoy_rows)},
        "decoy_group_audit": {
            **group_audit,
        },
        "selected_counts": {
            "total": len(selected_rows),
            "train": len(train_rows),
            "fresh_validation": role_counts["fresh_validation"],
            "locked_test": role_counts["locked_test"],
            "by_role": dict(sorted(role_counts.items())),
            "by_split_and_label": {
                f"{split}_{label}": observed_counts[(split, label)]
                for split, label in expected_counts
            },
        },
        "disjointness": {
            "ligand_id_disjoint": True,
            "source_molecule_id_within_label_disjoint": True,
            "canonical_smiles_disjoint": True,
            "bemis_murcko_scaffold_disjoint": True,
            "split_group_disjoint": True,
        },
        "identity_hashes": {
            role: identifier_sha256([str(row["ligand_id"]) for row in selected_rows if row["selection_role"] == role])
            for role in ("development_train", "fresh_validation", "locked_test")
        },
        "data_boundary": {
            "labels_used_only_for_frozen_panel_allocation": True,
            "docking_scores_read": 0,
            "fresh_validation_docking_scores_read": 0,
            "test_docking_scores_read": 0,
        },
        "outputs": {
            "selected_panel_manifest_csv": {"path": selected_path.relative_to(root).as_posix(), "sha256": file_sha256(selected_path)},
            "train_manifest_csv": {"path": train_path.relative_to(root).as_posix(), "sha256": file_sha256(train_path)},
        },
        "next_gate": "prepare only the frozen Train-668 ligands for Uni-Dock",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
