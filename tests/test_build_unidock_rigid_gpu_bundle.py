from scripts.experimental.unidock.build_unidock_rigid_gpu_bundle import CONFIGS, FIXED_PATHS


def test_rigid_gpu_bundle_declares_both_profiles_and_remote_runner():
    assert len(CONFIGS) == 2
    assert any("detail" in path for path in CONFIGS)
    assert any("enhanced" in path for path in CONFIGS)
    assert (
        "scripts/experimental/unidock/run_unidock_rigid_gpu_diagnostics_remote.sh"
        in FIXED_PATHS
    )
