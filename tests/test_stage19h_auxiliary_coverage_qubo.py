import json
from pathlib import Path

from scripts.diagnose_stage19h_auxiliary_coverage_qubo import (
    assignment_for_subset,
    build_auxiliary_qubo,
    binary_assignment,
    build_coverage_terms,
    qubo_energy,
    select_subset,
)
from scripts.build_stage19h_auxiliary_coverage_qubo_bundle import bundle_paths
from scripts.prepare_receptor import file_sha256


def test_binary_slack_encoding_covers_every_integer() -> None:
    for maximum in range(1, 12):
        weights = __import__(
            "scripts.diagnose_stage19h_auxiliary_coverage_qubo",
            fromlist=["slack_weights"],
        ).slack_weights(maximum)
        for value in range(maximum + 1):
            bits = binary_assignment(weights, value)
            assert sum(weights[index] * bit for index, bit in bits.items()) == value


def test_auxiliary_qubo_reduces_to_union_objective() -> None:
    terms = {
        "active_incidence": {"A": ["R1", "R2"]},
        "decoy_incidence": {"D": ["R2"]},
        "active_weights": {"A": 1.0},
        "decoy_weights": {"D": 1.0},
        "singleton_utility": {"R1": 1.0, "R2": 0.0},
        "correlations": {"R1__R2": 0.0},
    }
    qubo = build_auxiliary_qubo(
        terms, ["R1", "R2"], 1, 1.0, 0.5, 0.0, 20.0, 100.0
    )
    assignment = assignment_for_subset(terms, qubo, ("R1",))
    assert assignment["y__A"] == 1
    assert assignment["z__D"] == 0
    assert abs(qubo_energy(qubo, assignment) + 1.5) < 1e-9

    subset, details, _ = select_subset(
        terms,
        ["R1", "R2"],
        1,
        {
            "coverage_fraction": 0.1,
            "decoy_weight": 1.0,
            "singleton_weight": 0.5,
            "redundancy_weight": 0.0,
        },
        20.0,
        100.0,
        certify_all=True,
    )
    assert subset == ("R1",)
    assert details["feasible"] is True
    assert details["max_reduced_energy_residual"] < 1e-9
    assert details["equivalence_states_evaluated"] == 2


def test_stage19h_config_freezes_grid_and_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/stage19h_auxiliary_coverage_qubo.json"
    config = json.loads(config_path.read_text(encoding="ascii"))

    assert file_sha256(root / config["implementation"]["path"]) == (
        config["implementation"]["sha256"]
    )
    assert config["diagnostic"]["candidate_count"] == 54
    assert config["diagnostic"]["target_size"] == 3
    assert config["development_support_gate"][
        "bace1_method_amendment_authorized_by_this_stage"
    ] is False
    assert config["evidence_timing"]["bace1_benchmark_docking_started"] is False


def test_stage19h_result_preserves_failed_gate_and_compact_summary() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "data/stage19h_auxiliary_coverage_qubo_result.json").read_text(
            encoding="ascii"
        )
    )

    assert result["status"] == (
        "stage19h_auxiliary_coverage_not_supported_do_not_amend_bace1"
    )
    assert result["development_gate"]["passed"] is False
    assert result["development_gate"]["bace1_method_amendment_authorized"] is False
    assert all(
        value is False
        for value in result["development_gate"]["comparison_checks"].values()
    )
    assert "qubo" not in result["full_train_models"]["MK14"]
    assert "terms" not in result["full_train_models"]["PPARG"]
    assert result["data_boundary"]["new_docking_jobs"] == 0


def test_stage19h_independent_audit_certifies_all_full_train_triples() -> None:
    root = Path(__file__).resolve().parents[1]
    audit = json.loads(
        (root / "data/stage19h_auxiliary_coverage_qubo_audit.json").read_text(
            encoding="ascii"
        )
    )

    assert audit["status"] == "stage19h_auxiliary_coverage_qubo_audit_ok"
    assert audit["coverage"]["inner_candidate_rows_reselected"] == 1296
    assert audit["coverage"]["outer_candidate_rows_checked"] == 432
    assert audit["coverage"]["comparison_rows_checked"] == 40
    assert audit["coverage"]["full_train_qubo_states_certified"] == 1120
    assert audit["checks"]["failed_gate_reproduced"] is True
    assert audit["checks"]["cardinality_active_and_decoy_constraints_verified"] is True
    result_path = root / audit["result"]["path"]
    assert file_sha256(result_path) == audit["result"]["sha256"]


def test_stage19h_bundle_paths_exclude_protected_panels() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = bundle_paths(root)

    assert "data/stage19h_auxiliary_coverage_qubo_audit.json" in paths
    assert "data/stage19h_auxiliary_coverage_qubo_model_record.json" in paths
    assert not any(
        marker in path.lower()
        for path in paths
        for marker in ("fresh_validation", "locked_test", "bace1_docking")
    )
