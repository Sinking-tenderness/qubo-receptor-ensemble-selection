from scripts.build_stage05_mk14_expanded_e64_consensus_bundle import FIXED_PATHS


def test_bundle_contains_protocol_selection_inputs() -> None:
    assert "configs/stage05_mk14_expanded_train_e64_cpu2.txt" in FIXED_PATHS
    assert "scripts/run_stage05_mk14_expanded_e64_consensus_diagnostics.py" in FIXED_PATHS
    assert "data/processed/stage05_mk14_expanded_e32_matrix_flagged_pairs.csv" in FIXED_PATHS
