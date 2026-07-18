from scripts.build_stage05_mk14_expanded_train_e32_remote_bundle import FIXED_PATHS


def test_e32_bundle_contains_consensus_audit_and_all_seed_configs() -> None:
    assert "scripts/audit_stage05_expanded_e32_matrix.py" in FIXED_PATHS
    assert "configs/stage05_mk14_expanded_train_search_protocol_amendment01.json" in FIXED_PATHS
    for seed_index in range(3):
        assert (
            f"configs/stage05_mk14_expanded8_train160_e32_seed{seed_index}_linux.json"
            in FIXED_PATHS
        )
