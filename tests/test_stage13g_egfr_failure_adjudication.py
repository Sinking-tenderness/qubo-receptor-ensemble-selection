from scripts.adjudicate_stage13f_egfr_cognate_redocking_failure import (
    failure_diagnostics,
    truth,
)


def test_truth_parses_csv_booleans() -> None:
    assert truth(True)
    assert truth("True")
    assert not truth(False)
    assert not truth("False")


def test_failure_diagnostics_preserve_three_seed_failure() -> None:
    rows = [
        {
            "conformer_id": "R1",
            "seed_id": f"seed{index}",
            "top_ranked_rmsd_angstrom": rmsd,
            "top_ranked_affinity_kcal_per_mol": affinity,
            "top_ranked_pose_success": "False",
        }
        for index, (rmsd, affinity) in enumerate(
            ((3.1, -8.0), (3.2, -8.1), (3.3, -8.2))
        )
    ]
    result = failure_diagnostics(rows, ["R1"], {"R1"})[0]

    assert result["successful_seed_count"] == 0
    assert result["median_rmsd_angstrom"] == 3.2
    assert result["citation_mentions_covalent_or_irreversible_mechanism"] is True
