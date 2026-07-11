from scripts.cross_validate_ensemble_mvp import make_folds


def test_make_folds_is_stratified_and_deterministic() -> None:
    rows = [
        {"ligand_id": f"a{i}", "label": "active"} for i in range(6)
    ] + [{"ligand_id": f"d{i}", "label": "decoy"} for i in range(12)]
    first = make_folds(rows, 3, 19)
    second = make_folds(rows, 3, 19)
    assert first == second
    for fold in range(3):
        assert sum(first[row["ligand_id"]] == fold for row in rows if row["label"] == "active") == 2
        assert sum(first[row["ligand_id"]] == fold for row in rows if row["label"] == "decoy") == 4
