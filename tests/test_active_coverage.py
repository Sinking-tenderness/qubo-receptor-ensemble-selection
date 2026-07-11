from scripts.analyze_active_coverage import top_active_ids


def test_top_active_ids_use_ascending_vina_scores() -> None:
    rows = [
        {"ligand_id": "d", "label": "decoy", "r": "-12"},
        {"ligand_id": "a", "label": "active", "r": "-10"},
        {"ligand_id": "a2", "label": "active", "r": "-5"},
    ]

    assert top_active_ids(rows, "r", 2 / 3) == {"a"}
