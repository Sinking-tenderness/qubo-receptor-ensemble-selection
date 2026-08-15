from scripts.select_stage13b_egfr_expanded_coordinate_pool import (
    select_expanded_metadata_candidates,
)


def test_expansion_admits_only_resolution_only_failures() -> None:
    rows = [
        {
            "pdb_id": "REF",
            "status": "metadata_eligible",
            "exclusion_reasons": "",
            "resolution_angstrom": "2.0",
        },
        {
            "pdb_id": "NEW",
            "status": "metadata_excluded",
            "exclusion_reasons": "resolution_above_limit",
            "resolution_angstrom": "2.8",
        },
        {
            "pdb_id": "MUT",
            "status": "metadata_excluded",
            "exclusion_reasons": "mutation_count_differs;resolution_above_limit",
            "resolution_angstrom": "2.8",
        },
    ]
    amendment = {
        "metadata_pool_expansion": {
            "expanded_maximum_resolution_angstrom": 3.0,
            "newly_admitted_pdb_ids": ["NEW"],
            "original_metadata_eligible_count": 1,
            "expanded_metadata_candidate_count": 2,
        }
    }

    expanded, newly = select_expanded_metadata_candidates(rows, amendment)

    assert [row["pdb_id"] for row in expanded] == ["NEW", "REF"]
    assert [row["pdb_id"] for row in newly] == ["NEW"]
