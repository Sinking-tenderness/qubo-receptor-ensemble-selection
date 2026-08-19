from pathlib import Path

from qubo_receptor_ensemble.ligand_selection import select_scaffold_hash_ligands


def _write_ism(path: Path, rows: list[tuple[str, str]]) -> Path:
    path.write_text(
        "".join(f"{smiles} {source_id}\n" for smiles, source_id in rows),
        encoding="utf-8",
    )
    return path


def test_scaffold_hash_allocation_is_grouped_deterministic_and_scaffold_disjoint(
    tmp_path: Path,
) -> None:
    active = _write_ism(
        tmp_path / "actives.ism",
        [
            ("CCO", "ACTIVE-GROUP-1"),
            ("OCC", "ACTIVE-GROUP-1"),
            ("c1ccccc1O", "ACTIVE-GROUP-2"),
        ],
    )
    decoy = _write_ism(
        tmp_path / "decoys.ism",
        [
            ("c1ccccc1N", "DECOY-ACTIVE-SCAFFOLD"),
            ("c1ccncc1", "DECOY-PYRIDINE"),
            ("c1ccc2ccccc2c1", "DECOY-NAPHTHALENE"),
        ],
    )
    policy = {
        "hash_namespace": "STAGE102A",
        "outer_fold_count": 2,
        "minimum_label_counts_per_outer_fold": {"active": 1, "decoy": 1},
    }

    rows = select_scaffold_hash_ligands(
        active,
        decoy,
        target_id="TEST",
        label_counts={"active": 3, "decoy": 2},
        policy=policy,
    )
    repeated = select_scaffold_hash_ligands(
        active,
        decoy,
        target_id="TEST",
        label_counts={"active": 3, "decoy": 2},
        policy=policy,
    )

    assert rows == repeated
    assert len(rows) == 5
    assert sum(row["label"] == "active" for row in rows) == 3
    assert sum(row["label"] == "decoy" for row in rows) == 2
    assert sum(row["source_molecule_id"] == "ACTIVE-GROUP-1" for row in rows) == 2
    assert not any(row["source_molecule_id"] == "DECOY-ACTIVE-SCAFFOLD" for row in rows)
    assert all(row["selection_role"] == "development_train" for row in rows)
    assert all(row["split"] == "train" for row in rows)
    for label, minimum in policy["minimum_label_counts_per_outer_fold"].items():
        for fold in range(1, policy["outer_fold_count"] + 1):
            assert sum(
                row["label"] == label and int(row["outer_fold"]) == fold
                for row in rows
            ) >= minimum


def test_acyclic_scaffolds_fall_back_to_the_full_molecule(tmp_path: Path) -> None:
    active = _write_ism(
        tmp_path / "actives.ism",
        [("CCO", "ACTIVE-ETHANOL"), ("CCCO", "ACTIVE-PROPANOL")],
    )
    decoy = _write_ism(tmp_path / "decoys.ism", [("CCCC", "DECOY-BUTANE")])

    rows = select_scaffold_hash_ligands(
        active,
        decoy,
        target_id="TEST",
        label_counts={"active": 1, "decoy": 1},
        policy={
            "hash_namespace": "STAGE102A",
            "outer_fold_count": 1,
            "minimum_label_counts_per_outer_fold": {"active": 1, "decoy": 1},
        },
    )

    assert sum(row["label"] == "active" for row in rows) == 1
