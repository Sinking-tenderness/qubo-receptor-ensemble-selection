from scripts.cross_validate_scaffold_ensemble import make_scaffold_folds


def test_scaffold_folds_do_not_split_same_scaffold():
    rows = [
        {"ligand_id": "a1", "label": "active", "canonical_smiles": "c1ccccc1C"},
        {"ligand_id": "a2", "label": "active", "canonical_smiles": "c1ccccc1CC"},
        {"ligand_id": "a3", "label": "active", "canonical_smiles": "CCO"},
        {"ligand_id": "d1", "label": "decoy", "canonical_smiles": "c1ccccc1N"},
        {"ligand_id": "d2", "label": "decoy", "canonical_smiles": "c1ccccc1O"},
        {"ligand_id": "d3", "label": "decoy", "canonical_smiles": "CCN"},
    ]
    folds = make_scaffold_folds(rows, 3, 5)
    assert folds["a1"] == folds["a2"]
    assert folds["d1"] == folds["d2"]
