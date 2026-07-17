from scripts.split_ligand_scaffold import assign_groups, build_summary, scaffold_for_smiles


def test_scaffold_split_is_disjoint_and_deterministic():
    rows = [
        {"ligand_id": "a1", "label": "active", "canonical_smiles": "c1ccccc1C"},
        {"ligand_id": "a2", "label": "active", "canonical_smiles": "c1ccccc1CC"},
        {"ligand_id": "a3", "label": "active", "canonical_smiles": "CCO"},
        {"ligand_id": "d1", "label": "decoy", "canonical_smiles": "c1ccccc1N"},
        {"ligand_id": "d2", "label": "decoy", "canonical_smiles": "c1ccccc1O"},
        {"ligand_id": "d3", "label": "decoy", "canonical_smiles": "CCN"},
    ]
    first = assign_groups(rows, seed=7)
    second = assign_groups(rows, seed=7)
    assert first == second
    summary = build_summary(first, seed=7)
    assert summary["scaffold_disjoint"] is True
    assert summary["scaffolds_in_multiple_splits"] == {}


def test_duplicate_source_ids_link_different_scaffolds():
    rows = [
        {
            "ligand_id": "d1",
            "label": "decoy",
            "canonical_smiles": "c1ccccc1N",
            "source_molecule_id": "duplicate",
        },
        {
            "ligand_id": "d2",
            "label": "decoy",
            "canonical_smiles": "C1CCCCC1O",
            "source_molecule_id": "duplicate",
        },
        {
            "ligand_id": "a1",
            "label": "active",
            "canonical_smiles": "CCO",
            "source_molecule_id": "unique",
        },
    ]

    assigned = assign_groups(rows, seed=11)
    duplicates = [row for row in assigned if row["source_molecule_id"] == "duplicate"]
    summary = build_summary(assigned, seed=11)

    assert len({row["split"] for row in duplicates}) == 1
    assert len({row["split_group_id"] for row in duplicates}) == 1
    assert summary["source_id_disjoint"] is True
    assert summary["source_ids_in_multiple_splits"] == {}


def test_scaffold_ignores_double_bond_stereochemistry():
    assert scaffold_for_smiles("F/C=C/F") == scaffold_for_smiles("F/C=C\\F")
