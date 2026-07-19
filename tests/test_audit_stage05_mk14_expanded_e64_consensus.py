from scripts.audit_stage05_mk14_expanded_e64_consensus import compare_case


def test_compare_case_accepts_recomputed_values() -> None:
    row = {
        "case_id": "L1__R1",
        "ligand_id": "L1",
        "receptor_id": "R1",
        "successful_runs": 3,
        "e64_consensus_replicate_count": 2,
        "e64_consensus_passed": True,
        "diagnostic_classification": "stable_two_of_three_consensus_at_e64",
        "e64_minimum_score": -10.0,
        "e64_median_score": -9.9,
        "e64_maximum_score": -8.0,
        "e64_median_minus_minimum": 0.1,
        "absolute_e32_e64_minimum_delta": 0.0,
        "absolute_e32_e64_median_delta": 0.2,
    }

    compare_case(row, dict(row))
