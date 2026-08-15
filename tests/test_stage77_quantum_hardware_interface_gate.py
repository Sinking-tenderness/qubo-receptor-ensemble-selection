import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import scripts.run_stage77_quantum_hardware_interface_gate as s77


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage77_implementation_inputs_and_literature_are_hash_locked():
    config = read_json("configs/stage77_quantum_hardware_interface_gate.json")
    for section in ("implementation", "inputs"):
        for descriptor in config[section].values():
            path = ROOT / descriptor["path"]
            assert sha256(path) == descriptor["sha256"]
            assert path.stat().st_size == descriptor["size_bytes"]
    papers = (
        ROOT
        / "analysis/literature-search-20260807-quantum-hardware-variable-k-cqm/papers.md"
    ).read_text(encoding="utf-8")
    assert "Advantage2" in papers
    assert "Warm-starting quantum optimization" in papers
    assert "policy-excluded venues were omitted" in papers


def test_stage77_direct_conversion_fails_precision_but_not_ideal_embedding():
    result = read_json("data/stage77_quantum_hardware_interface_gate_result.json")
    summary = result["direct_encoding_summary"]
    assert summary["cqm_model_count"] == 80
    assert summary["cqm_hash_match_count"] == 80
    assert summary["maximum_direct_bqm_variable_count"] == 117
    assert summary["maximum_direct_bqm_auxiliary_variable_count"] == 14
    assert summary["maximum_direct_bqm_interaction_count"] == 6786
    assert summary["maximum_ideal_zephyr_physical_qubit_count"] == 1053
    assert summary["maximum_ideal_zephyr_chain_length"] == 9
    assert summary["maximum_required_signed_precision_bits"] == 30
    assert summary["maximum_coefficient_dynamic_range"] == pytest.approx(
        65_857_050.0
    )
    assert result["route_gate"]["direct_full_bqm_ideal_embedding_passed"] is True
    assert result["route_gate"]["direct_full_bqm_precision_passed"] is False
    assert result["decision"]["full_direct_qpu_bqm_route_authorized"] is False


def test_stage77_local_swap_bqms_cover_every_fixed_k_frontier():
    rows = read_csv(
        "results/runs/stage77_quantum_hardware_interface_gate/local_swap_bqm_metrics.csv"
    )
    assert len(rows) == 500
    assert Counter(row["target_id"] for row in rows) == {
        "BACE1": 140,
        "PPARG": 140,
        "PPARA": 100,
        "PPARD": 120,
    }
    assert min(int(row["eligible_quality_nonincreasing_move_count"]) for row in rows) == 7
    assert max(int(row["encoded_move_variable_count"]) for row in rows) == 40
    assert max(int(row["ideal_zephyr_physical_qubit_count"]) for row in rows) == 160
    assert max(int(row["ideal_zephyr_maximum_chain_length"]) for row in rows) == 4
    assert all(row["all_encoded_moves_quality_nonincreasing"] == "True" for row in rows)
    assert max(float(row["maximum_objective_identity_residual"]) for row in rows) < 1e-9
    assert min(float(row["minimum_conflicting_pair_energy_margin"]) for row in rows) > 0
    assert sum(row["improving_single_move_available"] == "True" for row in rows) == 15
    assert sum(
        row["hardware_resolvable_single_move_improvement"] == "True"
        for row in rows
    ) == 10
    result = read_json("data/stage77_quantum_hardware_interface_gate_result.json")
    assert result["local_swap_bqm_summary"][
        "unique_improving_fixed_k_instance_count"
    ] == 3
    assert result["local_swap_bqm_summary"][
        "unique_hardware_resolvable_fixed_k_instance_count"
    ] == 2
    assert min(float(row["quantized_bias_retention_fraction"]) for row in rows) == pytest.approx(
        0.9629629629629629
    )


def test_stage77_quantized_noisy_emulation_is_complete_and_not_success_only():
    rows = read_csv(
        "results/runs/stage77_quantum_hardware_interface_gate/emulation_trials.csv"
    )
    assert len(rows) == 10_000
    assert Counter(row["condition"] for row in rows) == {
        "float64_clean": 2000,
        "q10_clean": 2000,
        "q10_noise_0p25": 2000,
        "q10_noise_0p50": 2000,
        "q10_noise_1p00": 2000,
    }
    assert all(int(row["read_count"]) == 8 for row in rows)
    assert all(int(row["feasible_read_count"]) == 8 for row in rows)
    assert all(int(row["conflict_read_count"]) == 0 for row in rows)
    assert all(row["warm_guard_nonworse"] == "True" for row in rows)
    summaries = {
        row["condition"]: row
        for row in read_csv(
            "results/runs/stage77_quantum_hardware_interface_gate/emulation_summary.csv"
        )
    }
    assert len(summaries) == 5
    for summary in summaries.values():
        assert int(summary["subproblem_count"]) == 500
        assert int(summary["run_count"]) == 2000
        assert int(summary["read_count"]) == 16_000
        assert float(summary["feasible_read_fraction"]) == pytest.approx(1.0)
        assert float(summary["conflict_read_fraction"]) == pytest.approx(0.0)
        assert int(summary["strict_improvement_subproblem_count"]) == 15
        assert int(summary["hardware_resolvable_opportunity_subproblem_count"]) == 10
        assert int(summary["hardware_resolvable_opportunity_recovered_count"]) == 10
        assert float(
            summary["hardware_resolvable_opportunity_recovery_fraction"]
        ) == pytest.approx(1.0)


def test_stage77_local_encoding_smoke_is_exact_and_proxy_is_seed_deterministic():
    config = read_json("configs/stage77_quantum_hardware_interface_gate.json")
    inputs = {
        key: ROOT / descriptor["path"]
        for key, descriptor in config["inputs"].items()
    }
    cell = s77.source_cells(config, inputs)[0]
    k = next(iter(cell["frontiers"]))
    first = s77.build_swap_bqm(cell, k, config)
    second = s77.build_swap_bqm(cell, k, config)
    assert first["moves"] == second["moves"]
    assert first["bqm"].is_equal(second["bqm"])
    metrics = s77.local_identity_metrics(cell, first, config)
    assert metrics["maximum_objective_identity_residual"] < 1e-9
    condition = config["warm_start_emulation"]["conditions"][-1]
    left = s77.hardware_proxy_bqm(
        first["bqm"], condition, np.random.default_rng(77001)
    )
    right = s77.hardware_proxy_bqm(
        first["bqm"], condition, np.random.default_rng(77001)
    )
    assert left.is_equal(right)


def test_stage77_route_decision_and_claim_boundaries_are_explicit():
    result = read_json("data/stage77_quantum_hardware_interface_gate_result.json")
    decision = result["decision"]
    assert decision == {
        "advantage2_local_reverse_annealing_poc_ready_for_budget_request": True,
        "frozen_variable_k_cqm_remains_scientific_model": True,
        "full_direct_qpu_bqm_route_authorized": False,
        "ibm_warm_start_qaoa_full_problem_route_authorized": False,
        "leap_hybrid_cqm_application_route_recommended": True,
        "neutral_atom_full_problem_route_authorized": False,
        "paid_cloud_execution_authorized": False,
        "paid_qpu_execution_authorized": False,
        "quantum_advantage_claim_authorized": False,
        "quantum_scaling_claim_authorized": False,
        "trapped_ion_full_problem_route_authorized": False,
    }
    assert result["data_boundary"] == {
        "historical_development_targets_read": 4,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "cloud_cqm_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
    for descriptor in result["outputs"].values():
        path = ROOT / descriptor["path"]
        assert sha256(path) == descriptor["sha256"]
        assert path.stat().st_size == descriptor["size_bytes"]


def test_stage77_independent_audit_rebuilt_and_replayed_everything():
    audit = read_json("data/stage77_quantum_hardware_interface_gate_audit.json")
    assert audit["status"] == "stage77_quantum_hardware_interface_independent_audit_ok"
    assert audit["cqm_models_independently_rebuilt"] == 80
    assert audit["direct_bqm_metrics_independently_recomputed"] == 80
    assert audit["local_swap_bqms_independently_rebuilt"] == 500
    assert audit["emulation_runs_deterministically_replayed"] == 10_000
    assert audit["emulation_summaries_independently_recomputed"] == 5
    assert audit["full_direct_qpu_route_authorized"] is False
    assert audit["local_reverse_annealing_poc_ready_for_budget_request"] is True
    assert audit["paid_qpu_execution_authorized"] is False
    assert audit["quantum_advantage_claim_authorized"] is False
