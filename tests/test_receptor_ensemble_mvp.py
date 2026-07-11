from scripts.run_receptor_ensemble_mvp import (
    enumerate_fixed_size,
    parse_grid,
    validate_inputs,
)


def test_parse_grid_and_fixed_size() -> None:
    assert parse_grid("0, 0.5,1") == [0.0, 0.5, 1.0]
    assert enumerate_fixed_size(["r1", "r2", "r3"], 2) == [
        ("r1", "r2"),
        ("r1", "r3"),
        ("r2", "r3"),
    ]


def test_validate_inputs_requires_complete_matrix() -> None:
    matrix = [{"ligand_id": "l1", "label": "active", "r1": "-8"}]
    splits = [{"ligand_id": "l1", "split": "train"}]
    try:
        validate_inputs(matrix, splits, ["r1"])
    except ValueError as error:
        assert "train, validation, and test" in str(error)
    else:
        raise AssertionError("incomplete split labels must fail")
