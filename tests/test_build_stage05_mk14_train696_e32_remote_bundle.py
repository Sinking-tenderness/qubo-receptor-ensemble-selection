from scripts.build_stage05_mk14_train696_e32_remote_bundle import FIXED_PATHS


def test_train696_bundle_contains_new_run_merge_and_reused_evidence() -> None:
    for seed_index in range(3):
        assert (
            f"configs/stage05_mk14_expanded8_train536new_e32_seed{seed_index}_linux.json"
            in FIXED_PATHS
        )
        assert (
            f"results/runs/stage05_mk14_expanded8_train160_e32_seed{seed_index}_linux/summary.json"
            in FIXED_PATHS
        )
        assert (
            f"results/runs/stage05_mk14_expanded8_train160_e32_seed{seed_index}_linux/representative_scores.csv"
            in FIXED_PATHS
        )
    assert "scripts/merge_stage05_mk14_train696_e32.py" in FIXED_PATHS
    assert "scripts/run_stage05_mk14_train696_e32_remote.sh" in FIXED_PATHS
    assert (
        "results/runs/stage05_mk14_expanded8_train160_e32_aggregated/summary.json"
        in FIXED_PATHS
    )
