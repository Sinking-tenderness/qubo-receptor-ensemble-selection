from scripts.run_stage05_mk14_expanded_e64_consensus_diagnostics import summarize_case


def case():
    return {
        "case_id": "L1__R1",
        "ligand_id": "L1",
        "receptor_id": "R1",
        "selection_role": "development_train",
        "actual_seeds": {"seed0": 101, "seed1": 102, "seed2": 103},
        "source_e32_scores": {"seed0": -8.0, "seed1": -9.4, "seed2": -8.1},
    }


def gate():
    return {
        "maximum_consensus_delta_kcal_per_mol": 0.5,
        "minimum_consensus_replicates_per_pair": 2,
        "maximum_allowed_median_minus_minimum_kcal_per_mol": 0.5,
    }


def rows(scores):
    return [
        {"case_id": "L1__R1", "seed": seed, "status": "ok", "docking_score": score}
        for seed, score in zip([101, 102, 103], scores, strict=True)
    ]


def test_e64_summary_passes_two_of_three_consensus() -> None:
    result = summarize_case(case(), rows([-9.5, -9.4, -8.0]), gate())

    assert result["e64_consensus_passed"] is True
    assert result["e64_consensus_replicate_count"] == 2


def test_e64_summary_rejects_single_best_seed() -> None:
    result = summarize_case(case(), rows([-9.5, -8.0, -8.1]), gate())

    assert result["e64_consensus_passed"] is False
    assert result["diagnostic_classification"] == "persistent_e64_search_instability"
