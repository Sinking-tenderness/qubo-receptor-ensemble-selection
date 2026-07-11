from scripts.split_ligand_benchmark import split_rows


def test_split_is_stratified_and_deterministic() -> None:
    rows = [
        {"ligand_id": f"A{i}", "label": "active", "canonical_smiles": f"A{i}"}
        for i in range(5)
    ] + [
        {"ligand_id": f"D{i}", "label": "decoy", "canonical_smiles": f"D{i}"}
        for i in range(10)
    ]

    first = split_rows(rows, 0.6, 0.2, 17)
    second = split_rows(rows, 0.6, 0.2, 17)

    assert first == second
    assert {(row["label"], row["split"]) for row in first} >= {
        ("active", "train"),
        ("active", "validation"),
        ("active", "test"),
        ("decoy", "train"),
        ("decoy", "validation"),
        ("decoy", "test"),
    }
