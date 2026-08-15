from pathlib import Path

from scripts.audit_stage16a_dpp4_source import (
    audit_ism_with_frozen_exclusions,
    ca_coordinate_set,
)


def test_ca_coordinate_set_can_filter_chain(tmp_path: Path) -> None:
    path = tmp_path / "reference.pdb"
    path.write_text(
        "ATOM      1  CA  GLY A   1       1.000   2.000   3.000\n"
        "ATOM      2  CA  GLY B   1       4.000   5.000   6.000\n",
        encoding="ascii",
    )

    assert ca_coordinate_set(path, "B") == {(4.0, 5.0, 6.0)}
    assert len(ca_coordinate_set(path, None)) == 2


def test_frozen_invalid_source_row_is_excluded_without_rewriting(tmp_path: Path) -> None:
    path = tmp_path / "decoys.ism"
    path.write_text("CC good\ninvalid bad\nCCC good2\n", encoding="ascii")

    result = audit_ism_with_frozen_exclusions(
        path,
        {
            "row_count": 3,
            "valid_row_count": 2,
            "valid_unique_source_id_count": 2,
            "valid_duplicate_source_id_count": 0,
            "maximum_source_id_multiplicity": 1,
            "allowed_invalid_rows": [
                {
                    "line_number": 2,
                    "source_id": "bad",
                    "smiles": "invalid",
                    "reason": "frozen test exclusion",
                }
            ],
        },
    )

    assert result["raw_row_count"] == 3
    assert result["valid_row_count"] == 2
    assert result["excluded_invalid_row_count"] == 1
