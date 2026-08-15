import json
from pathlib import Path

from scripts.build_stage20_adaptive_k_objective_screen_bundle import bundle_paths
from scripts.diagnose_stage20_adaptive_k_objective_screen import (
    singleton_certificate,
    singleton_assignment,
    threshold_certificate,
    write_singleton_qubo,
)
from scripts.diagnose_stage19i_objective_adequacy_noise_screen import (
    build_qubo,
)
from scripts.prepare_receptor import file_sha256


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


def test_singleton_qubo_matches_singleton_ordering() -> None:
    receptors = ["R1", "R2", "R3"]
    values = {"R1": 0.2, "R2": 0.9, "R3": 0.4}
    qubo = write_singleton_qubo(receptors, values, 20.0)
    certificate = singleton_certificate(qubo, receptors, values)

    assert certificate["selected_subset"] == ["R2"]
    assert certificate["state_count"] == 3
    assert singleton_assignment(qubo, ("R2",))["R2"] == 1


def test_adaptive_threshold_qubo_certifies_each_scheduled_threshold() -> None:
    receptors = ["R1", "R2", "R3"]
    for threshold in (1, 2, 3):
        terms = synthetic_terms(threshold)
        qubo = build_qubo(terms, receptors, 2, 1.0, 0.0, 20.0, 100.0)
        certificate = threshold_certificate(terms, qubo, receptors, 2, 1.0, 0.0)

        assert certificate["state_count"] == 3
        assert certificate["equivalence_residual_sample"] < 1e-9
        assert len(certificate["selected_subset"]) == 2


def test_stage20_config_freezes_k_schedule_and_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/stage20_adaptive_k_objective_screen.json"
    config = json.loads(config_path.read_text(encoding="ascii"))

    assert file_sha256(root / config["implementation"]["path"]) == (
        config["implementation"]["sha256"]
    )
    assert [item["k"] for item in config["diagnostic"]["k_schedule"]] == [1, 2, 3, 4, 5, 6]
    assert [item["active_threshold"] for item in config["diagnostic"]["k_schedule"]] == [1, 1, 2, 2, 3, 3]
    assert config["diagnostic"]["stop_after_consecutive_failures"] == 2
    assert config["evidence_timing"]["quantum_hardware_execution"] is False
    assert config["stopping_rule"]["bace1_method_amendment_authorized"] is False


def test_stage20_result_recommends_small_pool_and_stops_after_two_failures() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "data/stage20_adaptive_k_objective_screen_result.json").read_text(
            encoding="ascii"
        )
    )

    assert result["status"] == "stage20_adaptive_k_train_only_screen_complete"
    assert result["one_standard_error"]["best_k"] == 1
    assert result["one_standard_error"]["recommended_smallest_k"] == 1
    assert result["stopping_recommendation"]["recommended_stop_k"] == 3
    assert len(result["global_candidate_curve"]) == 6
    assert result["data_boundary"]["new_docking_jobs"] == 0
    assert result["data_boundary"]["quantum_hardware_jobs"] == 0


def test_stage20_audit_recomputes_all_cardinalities() -> None:
    root = Path(__file__).resolve().parents[1]
    audit = json.loads(
        (root / "data/stage20_adaptive_k_objective_screen_audit.json").read_text(
            encoding="ascii"
        )
    )

    assert audit["status"] == "stage20_adaptive_k_objective_screen_audit_ok"
    assert audit["coverage"]["fold_rows_recomputed"] == 192
    assert audit["coverage"]["full_train_models_recomputed"] == 12
    assert audit["coverage"]["incremental_rows_checked"] == 12
    assert audit["coverage"]["subset_states_by_target_represented"] == 29784
    assert audit["checks"]["stopping_rule_reproduced"] is True
    assert file_sha256(root / audit["result"]["path"]) == audit["result"]["sha256"]


def test_stage20_bundle_paths_exclude_protected_panels() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = bundle_paths(root)

    assert "data/stage20_adaptive_k_objective_screen_audit.json" in paths
    assert not any(
        marker in path.lower()
        for path in paths
        for marker in ("fresh_validation", "locked_test", "bace1_docking")
    )
