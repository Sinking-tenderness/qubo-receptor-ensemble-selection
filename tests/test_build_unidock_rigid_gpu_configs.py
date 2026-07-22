from scripts.experimental.unidock.build_unidock_rigid_gpu_configs import build_config


def test_build_config_freezes_boundary_and_profile():
    profile = {
        "profile_id": "test",
        "exhaustiveness": 1024,
        "max_step": 80,
        "run_name": "fixture_run",
        "purpose": "fixture",
    }
    config = build_config(
        profile,
        {"path": "r.csv", "sha256": "R"},
        {"path": "l.csv", "sha256": "L"},
        [],
        {"pair_count": 1, "elapsed_seconds": 1.0, "comparison_note": "x"},
    )

    assert config["data_boundary"]["validation_rows_permitted"] == 0
    assert config["unidock"]["exhaustiveness"] == 1024
    assert config["unidock"]["max_step"] == 80
    assert config["unidock"]["macrocycle_closure_pseudoatom_policy"] == "reject"
    assert config["expected"]["cpu_reference_pair_count_per_seed"] == 800
