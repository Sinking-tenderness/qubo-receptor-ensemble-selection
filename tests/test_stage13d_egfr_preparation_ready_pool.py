from scripts.finalize_stage13d_egfr_preparation_ready_pool import (
    preparation_ready_rows,
)


def test_preparation_ready_rows_exclude_any_incomplete_residue() -> None:
    rows = [
        {
            "pdb_id": "GOOD",
            "status": "coordinate_eligible",
            "global_incomplete_standard_amino_acid_residue_count": "0",
        },
        {
            "pdb_id": "BAD",
            "status": "coordinate_eligible",
            "global_incomplete_standard_amino_acid_residue_count": "1",
        },
    ]
    amendment = {"selection": {"candidate_pdb_ids": ["GOOD"]}}

    assert preparation_ready_rows(rows, amendment) == [rows[0]]
