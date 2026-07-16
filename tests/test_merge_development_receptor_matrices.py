from scripts.merge_development_receptor_matrices import (
    merge_matrix_groups,
    merge_warning_rows,
)


def test_merge_matrix_groups_uses_fixed_receptor_order():
    first = {"L1": {"target_id": "T", "ligand_id": "L1", "label": "active", "R2": "-7"}}
    second = {"L1": {"target_id": "T", "ligand_id": "L1", "label": "active", "R1": "-8"}}

    rows = merge_matrix_groups(["R1", "R2"], first, second)

    assert rows == [{
        "target_id": "T",
        "ligand_id": "L1",
        "label": "active",
        "R1": -8.0,
        "R2": -7.0,
    }]


def test_merge_warning_rows_filters_locked_md_rows_and_preserves_non_md():
    md = [
        {"ligand_id": "DEV", "receptor_id": "M1"},
        {"ligand_id": "LOCKED", "receptor_id": "M1"},
    ]
    non_md = [{"ligand_id": "DEV", "receptor_id": "X1"}]

    rows, skipped = merge_warning_rows(md, non_md, {"DEV"}, {"M1"}, {"X1"})

    assert skipped == 1
    assert [(row["receptor_id"], row["source_group"]) for row in rows] == [
        ("M1", "md"),
        ("X1", "non_md"),
    ]
