"""Deterministically allocate scaffold-grouped EGFR and FA10 Phase A panels."""

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
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if len(fields) < 2:
                raise ValueError(f"incomplete {label} row: {line_number}")
            molecule = Chem.MolFromSmiles(fields[0])
            if molecule is None:
                raise ValueError(f"unparseable {label} SMILES: {line_number}")
            rows.append({
                "source_line_number": line_number,
                "source_molecule_id": fields[1],
                "source_smiles": fields[0],
                "canonical_smiles": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
                "scaffold_smiles": scaffold_for(molecule),
                "label": label,
            })
    return rows


def connected_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first, second = find(first), find(second)
        if first != second:
            parent[max(first, second)] = min(first, second)

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
    return [groups[index] for index in sorted(groups)]


def group_id(target: str, group: list[dict[str, Any]]) -> str:
    label = str(group[0]["label"]).upper()
    payload = "\n".join(sorted(f"{int(row['source_line_number']):06d}|{row['source_molecule_id']}" for row in group))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20].upper()
    return f"{target}_{label}_G{digest}"


def exact_groups(target: str, groups: list[list[dict[str, Any]]], count: int, seed: str) -> list[list[dict[str, Any]]]:
    ranked = sorted(groups, key=lambda group: (hashlib.sha256(f"{seed}|{group_id(target, group)}".encode("utf-8")).hexdigest(), group_id(target, group)))
    parent: dict[int, tuple[int, int] | None] = {0: None}
    for index, group in enumerate(ranked):
        size = len(group)
        for current in sorted(list(parent), reverse=True):
            new = current + size
            if new <= count and new not in parent:
                parent[new] = (current, index)
        if count in parent:
            break
    if count not in parent:
        raise ValueError(f"cannot allocate exactly {count} {target} ligands")
    selected_indices: set[int] = set()
    current = count
    while current:
        previous, index = parent[current]  # type: ignore[misc]
        selected_indices.add(index)
        current = previous
    return [group for index, group in enumerate(ranked) if index in selected_indices]


def assign_folds(target: str, groups: list[list[dict[str, Any]]], seed: str) -> dict[str, int]:
    counts = [0] * 5
    assignment: dict[str, int] = {}
    ranked = sorted(groups, key=lambda group: (-len(group), hashlib.sha256(f"{seed}|{group_id(target, group)}".encode("utf-8")).hexdigest()))
    for group in ranked:
        fold = min(range(5), key=lambda index: (counts[index], index))
        assignment[group_id(target, group)] = fold + 1
        counts[fold] += len(group)
    return assignment


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    panel = config["phase_a_development_expansion"]["ligand_panel_per_target"]
    result: dict[str, Any] = {"schema_version": "1.0", "status": "stage102a_phase_a_ligands_allocated", "targets": {}, "docking_scores_read": 0}
    for target, spec in config["phase_a_development_expansion"]["targets"].items():
        actives = read_ism(root / spec["active_source"], "active")
        decoys = read_ism(root / spec["decoy_source"], "decoy")
        active_groups = connected_groups(actives)
        selected_active = exact_groups(target, active_groups, int(panel["active_count"]), f"STAGE102A|{target}|ACTIVE")
        active_scaffolds = {str(row["scaffold_smiles"]) for group in selected_active for row in group}
        decoy_groups = [group for group in connected_groups(decoys) if not any(str(row["scaffold_smiles"]) in active_scaffolds for row in group)]
        selected_decoy = exact_groups(target, decoy_groups, int(panel["decoy_count"]), f"STAGE102A|{target}|DECOY")
        selected_groups = selected_active + selected_decoy
        folds = assign_folds(target, selected_active, f"STAGE102A|{target}|ACTIVE|FOLD")
        folds.update(assign_folds(target, selected_decoy, f"STAGE102A|{target}|DECOY|FOLD"))
        rows = []
        for group in selected_groups:
            identity = group_id(target, group)
            rank_hash = hashlib.sha256(f"STAGE102A|{target}|{identity}".encode("utf-8")).hexdigest()
            for row in group:
                rows.append({
                    **row,
                    "ligand_id": f"{target}_{row['label']}_L{int(row['source_line_number']):06d}",
                    "smiles": row["source_smiles"],
                    "source": "DUD-E",
                    "target_id": target,
                    "split_group_id": identity,
                    "allocation_rank_sha256": rank_hash,
                    "selection_role": "development_train",
                    "split": "train",
                    "outer_fold": folds[identity],
                })
        rows.sort(key=lambda row: (row["label"], row["allocation_rank_sha256"], int(row["source_line_number"])))
        counts = Counter((row["label"], int(row["outer_fold"])) for row in rows)
        if any(counts[("active", fold)] < int(panel["minimum_active_count_per_outer_fold"]) for fold in range(1, 6)):
            raise ValueError(f"{target} active outer-fold balance failed")
        if any(counts[("decoy", fold)] < int(panel["minimum_decoy_count_per_outer_fold"]) for fold in range(1, 6)):
            raise ValueError(f"{target} decoy outer-fold balance failed")
        output = root / f"data/processed/stage102a_{target.lower()}_phase_a_ligand_manifest.csv"
        write_csv(output, rows)
        result["targets"][target] = {
            "source_active_count": len(actives),
            "source_decoy_count": len(decoys),
            "selected_active_count": sum(row["label"] == "active" for row in rows),
            "selected_decoy_count": sum(row["label"] == "decoy" for row in rows),
            "outer_fold_counts": {f"{label}_fold{fold}": counts[(label, fold)] for label in ("active", "decoy") for fold in range(1, 6)},
            "manifest": {"path": output.relative_to(root).as_posix(), "sha256": sha256(output)},
        }
    output = root / "data/stage102a_phase_a_ligand_allocation_summary.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage102_prospective_marginal_learning.json"))
    args = parser.parse_args()
    run((args.root / args.config).resolve(), args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
