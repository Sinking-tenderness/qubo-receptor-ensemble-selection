import json
from pathlib import Path

from scripts.build_stage19i_objective_adequacy_noise_screen_bundle import bundle_paths
from scripts.diagnose_stage19i_objective_adequacy_noise_screen import (
    build_qubo,
    build_terms,
    certify_states,
    file_sha256,
)


def synthetic_terms(threshold: int) -> dict[str, object]:
    return {
        "coverage_fraction": 0.1,
        "active_threshold": threshold,
        "decoy_threshold": 1,
        "bedroc_alpha": 20.0,
        "active_ids": ["A"],
        "decoy_ids": ["D"],
        "active_incidence": {"A": ["R1", "R2", "R3"]},
        "decoy_incidence": {"D": ["R2"]},
        "active_weights": {"A": 1.0},
        "decoy_weights": {"D": 1.0},
        "correlations": {"R1__R2": 0.0, "R1__R3": 0.0, "R2__R3": 0.0},
    }


def test_threshold_auxiliary_encoding_certifies_all_triples() -> None:
    for threshold in (1, 2, 3):
        terms = synthetic_terms(threshold)
        qubo = build_qubo(terms, ["R1", "R2", "R3"], 2, 1.0, 0.0, 20.0, 100.0)
        certificate = certify_states(
            terms, qubo, ["R1", "R2", "R3"], 2, 1.0, 0.0
        )
        assert certificate["state_count"] == 3
        assert certificate["equivalence_residual"] < 1e-9
        assert certificate["selected_subset"] in (
            ["R1", "R2"],
            ["R1", "R3"],
            ["R2", "R3"],
        )


def test_stage19i_config_freezes_candidates_and_hardware_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/stage19i_objective_adequacy_noise_screen.json"
    config = json.loads(config_path.read_text(encoding="ascii"))

    assert file_sha256(root / config["implementation"]["path"]) == (
        config["implementation"]["sha256"]
    )
    assert [item["active_threshold"] for item in config["diagnostic"]["candidates"]] == [
        1,
        2,
        3,
        2,
    ]
    assert config["diagnostic"]["noise_repeats"] == 32
    assert [item["mode"] for item in config["diagnostic"]["noise_models"]] == [
        "absolute",
        "relative",
    ]
    assert config["evidence_timing"]["quantum_hardware_execution"] is False
    assert config["hardware_readiness_gate"][
        "bace1_method_amendment_authorized"
    ] is False


def test_stage19i_result_blocks_hardware_when_gap_and_stability_fail() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "data/stage19i_objective_adequacy_noise_screen_result.json").read_text(
            encoding="ascii"
        )
    )

    assert result["status"] == (
        "stage19i_no_candidate_hardware_ready_do_not_execute_quantum"
    )
    assert result["ready_candidates"] == []
    assert all(
        value["ready_for_hardware_pilot"] is False
        for value in result["candidate_gate"].values()
    )
    assert all(
        float(target_info["scaled_best_second_gap"]) < 0.01
        for candidate_info in result["hardware_checks"].values()
        for target_info in candidate_info["per_target"].values()
    )
    assert result["data_boundary"]["quantum_hardware_jobs"] == 0


def test_stage19i_audit_reproduces_all_screen_rows() -> None:
    root = Path(__file__).resolve().parents[1]
    audit = json.loads(
        (root / "data/stage19i_objective_adequacy_noise_screen_audit.json").read_text(
            encoding="ascii"
        )
    )

    assert audit["status"] == "stage19i_objective_adequacy_noise_screen_audit_ok"
    assert audit["coverage"]["fold_rows_recomputed"] == 56
    assert audit["coverage"]["candidate_outer_selections_recomputed"] == 32
    assert audit["coverage"]["full_train_models_recomputed"] == 8
    assert audit["coverage"]["noise_summary_rows_recomputed"] == 64
    assert audit["coverage"]["noise_trial_rows_recomputed_from_summary"] == 2048
    assert audit["checks"]["no_candidate_passed_hardware_gate"] is True
    assert file_sha256(root / audit["result"]["path"]) == audit["result"]["sha256"]


def test_stage19i_bundle_paths_exclude_protected_panels() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = bundle_paths(root)

    assert "data/stage19i_objective_adequacy_noise_screen_audit.json" in paths
    assert not any(
        marker in path.lower()
        for path in paths
        for marker in ("fresh_validation", "locked_test", "bace1_docking")
    )
