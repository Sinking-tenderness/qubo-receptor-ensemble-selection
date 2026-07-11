from scripts.split_ligand_scaffold import assign_groups, build_summary


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
