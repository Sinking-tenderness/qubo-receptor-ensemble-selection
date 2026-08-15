from scripts.adjudicate_stage14d_fa10_cognate_redocking_failure import (
    failure_diagnostics,
)


def rows_for(receptor_id: str, rmsds: tuple[float, float, float]):
    return [
        {
            "conformer_id": receptor_id,
            "seed_id": f"seed{index}",
            "top_ranked_rmsd_angstrom": rmsd,
            "top_ranked_affinity_kcal_per_mol": -8.0 - index / 10,
            "top_ranked_pose_success": "False",
        }
        for index, rmsd in enumerate(rmsds)
    ]


def test_failure_diagnostics_distinguish_near_threshold_failure() -> None:
    result = failure_diagnostics(
        rows_for("R1", (2.05, 2.10, 2.20)), ["R1"], 2.0
    )[0]

    assert result["successful_seed_count"] == 0
    assert result["median_rmsd_angstrom"] == 2.10
    assert result["failure_class"] == "three-seed stable near-threshold pose mismatch"


def test_failure_diagnostics_preserve_large_alternative_pose() -> None:
    result = failure_diagnostics(
        rows_for("R2", (7.2, 7.3, 7.4)), ["R2"], 2.0
    )[0]

    assert result["successful_seed_count"] == 0
    assert result["failure_class"] == "three-seed stable alternative pose"
    assert result["unique_cause_established"] is False
