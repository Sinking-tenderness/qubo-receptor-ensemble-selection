from scripts.cross_validate_ensemble_mvp import normalize_rows_by_train_minmax


def test_minmax_uses_train_bounds_for_held_out_values():
    train = [
        {"ligand_id": "a", "R1": "-10"},
        {"ligand_id": "b", "R1": "-5"},
    ]
    validation = [{"ligand_id": "c", "R1": "-12"}]
    train_out, [validation_out] = normalize_rows_by_train_minmax(train, [validation], ["R1"])
    assert train_out[0]["R1"] == 0.0
    assert train_out[1]["R1"] == 1.0
    assert validation_out[0]["R1"] < 0.0
