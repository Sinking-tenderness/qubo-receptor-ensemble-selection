from scripts.audit_mk14_rcsb_coordinate_selection import reconstruct_maxmin


def test_independent_maxmin_reconstruction_uses_lexical_tie_break():
    ids = ["E1", "E2", "N1", "N2", "N3"]
    distances = {
        tuple(sorted(pair)): value
        for pair, value in {
            ("E1", "E2"): 1.0,
            ("E1", "N1"): 4.0,
            ("E2", "N1"): 3.0,
            ("E1", "N2"): 2.0,
            ("E2", "N2"): 5.0,
            ("E1", "N3"): 3.0,
            ("E2", "N3"): 3.0,
            ("N1", "N2"): 4.0,
            ("N1", "N3"): 1.0,
            ("N2", "N3"): 5.0,
        }.items()
    }

    result = reconstruct_maxmin(ids, ["E1", "E2"], distances, 2)

    assert [row["conformer_id"] for row in result] == ["N1", "N2"]
    assert [row["minimum_standardized_distance_to_selected_pool"] for row in result] == [
        3.0,
        2.0,
    ]
