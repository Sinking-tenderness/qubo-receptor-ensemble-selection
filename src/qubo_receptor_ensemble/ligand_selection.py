"""Deterministic scaffold-aware ligand allocation for development panels."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def _scaffold_for(molecule: Chem.Mol) -> str:
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


def scaffold_smiles(smiles: str) -> str:
    """Return the canonical, stereo-insensitive Murcko scaffold for SMILES."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return _scaffold_for(molecule)


def _read_ism(path: Path, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.strip().split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number} has fewer than two columns")
            molecule = Chem.MolFromSmiles(fields[0])
            if molecule is None:
                raise ValueError(f"{path}:{line_number} has an invalid SMILES")
            rows.append(
                {
                    "source_line_number": line_number,
                    "source_molecule_id": fields[1],
                    "source_extra_id": fields[2] if len(fields) > 2 else "",
                    "source_smiles": fields[0],
                    "canonical_smiles": Chem.MolToSmiles(
                        molecule, canonical=True, isomericSmiles=True
                    ),
                    "scaffold_smiles": _scaffold_for(molecule),
                    "label": label,
                }
            )
    if not rows:
        raise ValueError(f"{label} ISM contains no ligands: {path}")
    return rows


def _connected_groups(rows: list[dict[str, object]]) -> list[list[dict[str, object]]]:
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

    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    return [groups[index] for index in sorted(groups)]


def _group_id(target_id: str, group: list[dict[str, object]]) -> str:
    label = str(group[0]["label"]).upper()
    payload = "\n".join(
        sorted(
            f"{int(row['source_line_number']):06d}|{row['source_molecule_id']}"
            for row in group
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20].upper()
    return f"{target_id}_{label}_G{digest}"


def _select_exact_groups(
    target_id: str,
    groups: list[list[dict[str, object]]],
    count: int,
    seed: str,
) -> list[list[dict[str, object]]]:
    ranked = sorted(
        groups,
        key=lambda group: (
            hashlib.sha256(f"{seed}|{_group_id(target_id, group)}".encode("utf-8"))
            .hexdigest(),
            _group_id(target_id, group),
        ),
    )
    reachable: dict[int, tuple[int, int] | None] = {0: None}
    for index, group in enumerate(ranked):
        size = len(group)
        for current in sorted(list(reachable), reverse=True):
            new_count = current + size
            if new_count <= count and new_count not in reachable:
                reachable[new_count] = (current, index)
        if count in reachable:
            break
    if count not in reachable:
        raise ValueError(f"cannot allocate exactly {count} {target_id} ligands")

    selected_indices: set[int] = set()
    current = count
    while current:
        previous, index = reachable[current]  # type: ignore[misc]
        selected_indices.add(index)
        current = previous
    return [
        group for index, group in enumerate(ranked) if index in selected_indices
    ]


def _assign_outer_folds(
    target_id: str,
    groups: list[list[dict[str, object]]],
    seed: str,
    fold_count: int,
) -> dict[str, int]:
    counts = [0] * fold_count
    assignment: dict[str, int] = {}
    ranked = sorted(
        groups,
        key=lambda group: (
            -len(group),
            hashlib.sha256(f"{seed}|{_group_id(target_id, group)}".encode("utf-8"))
            .hexdigest(),
        ),
    )
    for group in ranked:
        fold = min(range(fold_count), key=lambda index: (counts[index], index))
        assignment[_group_id(target_id, group)] = fold + 1
        counts[fold] += len(group)
    return assignment


def _policy_value(policy: Mapping[str, object], key: str, default: object) -> object:
    value = policy.get(key, default)
    if value is None:
        return default
    return value


def select_scaffold_hash_ligands(
    active_path: str | Path,
    decoy_path: str | Path,
    *,
    target_id: str,
    label_counts: Mapping[str, int],
    policy: Mapping[str, object] | None = None,
    source: str = "DUD-E",
) -> list[dict[str, object]]:
    """Select a scaffold-aware development panel without using docking data."""
    policy = policy or {}
    namespace = str(_policy_value(policy, "hash_namespace", "STAGE102A"))
    fold_count = int(_policy_value(policy, "outer_fold_count", 5))
    minimum_counts_value = _policy_value(
        policy,
        "minimum_label_counts_per_outer_fold",
        {"active": 20, "decoy": 80},
    )
    if fold_count <= 0:
        raise ValueError("outer_fold_count must be positive")
    if not isinstance(minimum_counts_value, Mapping):
        raise ValueError("minimum_label_counts_per_outer_fold must be an object")
    minimum_counts = {
        str(label): int(value) for label, value in minimum_counts_value.items()
    }

    active_count = int(label_counts.get("active", 0))
    decoy_count = int(label_counts.get("decoy", 0))
    if active_count <= 0 or decoy_count <= 0:
        raise ValueError("active and decoy counts must be positive")
    active_groups = _connected_groups(_read_ism(Path(active_path), "active"))
    decoy_groups = _connected_groups(_read_ism(Path(decoy_path), "decoy"))
    selected_active = _select_exact_groups(
        target_id,
        active_groups,
        active_count,
        f"{namespace}|{target_id}|ACTIVE",
    )
    active_scaffolds = {
        str(row["scaffold_smiles"])
        for group in selected_active
        for row in group
    }
    eligible_decoy_groups = [
        group
        for group in decoy_groups
        if not any(str(row["scaffold_smiles"]) in active_scaffolds for row in group)
    ]
    selected_decoy = _select_exact_groups(
        target_id,
        eligible_decoy_groups,
        decoy_count,
        f"{namespace}|{target_id}|DECOY",
    )
    selected_groups = selected_active + selected_decoy
    fold_assignments = _assign_outer_folds(
        target_id,
        selected_active,
        f"{namespace}|{target_id}|ACTIVE|FOLD",
        fold_count,
    )
    fold_assignments.update(
        _assign_outer_folds(
            target_id,
            selected_decoy,
            f"{namespace}|{target_id}|DECOY|FOLD",
            fold_count,
        )
    )

    rows: list[dict[str, object]] = []
    for group in selected_groups:
        allocation_group_id = _group_id(target_id, group)
        allocation_rank_sha256 = hashlib.sha256(
            f"{namespace}|{target_id}|{allocation_group_id}".encode("utf-8")
        ).hexdigest()
        for row in group:
            label = str(row["label"])
            rows.append(
                {
                    "ligand_id": f"{target_id}_{label}_L{int(row['source_line_number']):06d}",
                    "smiles": str(row["source_smiles"]),
                    "label": label,
                    "source": source,
                    "target_id": target_id,
                    "source_molecule_id": str(row["source_molecule_id"]),
                    "source_extra_id": str(row["source_extra_id"]),
                    "source_line_number": int(row["source_line_number"]),
                    "canonical_smiles": str(row["canonical_smiles"]),
                    "scaffold_smiles": str(row["scaffold_smiles"]),
                    "allocation_group_id": allocation_group_id,
                    "allocation_rank_sha256": allocation_rank_sha256,
                    "selection_role": "development_train",
                    "split": "train",
                    "outer_fold": fold_assignments[allocation_group_id],
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["label"]),
            str(row["allocation_rank_sha256"]),
            int(row["source_line_number"]),
        )
    )

    fold_counts = Counter(
        (str(row["label"]), int(row["outer_fold"])) for row in rows
    )
    for label, minimum in minimum_counts.items():
        for fold in range(1, fold_count + 1):
            if fold_counts[(label, fold)] < minimum:
                raise ValueError(
                    f"{target_id} {label} outer fold {fold} has "
                    f"{fold_counts[(label, fold)]} rows; minimum is {minimum}"
                )
    if len({str(row["ligand_id"]) for row in rows}) != len(rows):
        raise ValueError("scaffold hash allocation produced duplicate ligand IDs")
    return rows


def summarize_scaffold_hash_allocation(
    rows: list[dict[str, object]], *, policy: Mapping[str, object]
) -> dict[str, object]:
    """Build a compact, score-free audit record for a selected panel."""
    label_counts = Counter(str(row["label"]) for row in rows)
    fold_counts = Counter(
        f"{row['label']}_fold{int(row['outer_fold'])}" for row in rows
    )
    return {
        "status": "ok",
        "method": "scaffold_hash_allocation",
        "policy": dict(policy),
        "ligand_count": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "outer_fold_counts": dict(sorted(fold_counts.items())),
        "allocation_group_count": len(
            {str(row["allocation_group_id"]) for row in rows}
        ),
        "docking_scores_read": 0,
    }
