from scripts.repeat_ensemble_mvp import run_one


def test_repeat_run_preserves_split_separation() -> None:
    matrix = [
        {"ligand_id": f"A{i}", "label": "active", "r1": "-10", "r2": "-9"}
        for i in range(4)
    ] + [
        {"ligand_id": f"D{i}", "label": "decoy", "r1": "-5", "r2": "-6"}
        for i in range(8)
    ]
    ligands = [
        {"ligand_id": f"A{i}", "label": "active", "canonical_smiles": f"A{i}"}
        for i in range(4)
    ] + [
        {"ligand_id": f"D{i}", "label": "decoy", "canonical_smiles": f"D{i}"}
        for i in range(8)
    ]
    result = run_one(matrix, ligands, ["r1", "r2"], 7, 1)
    assert result["split_counts"]["train"]["active_count"] == 2
    assert result["split_counts"]["validation"]["active_count"] == 1
    assert result["split_counts"]["test"]["active_count"] == 1
    assert set(result["qubo_subset"]).issubset({"r1", "r2"})
