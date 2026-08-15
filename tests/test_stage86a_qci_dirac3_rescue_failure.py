from pathlib import Path

import scripts.run_stage84_mixed_radix_dirac_iqp_gate as s84


ROOT = Path(__file__).resolve().parents[1]


def test_stage86a_physical_rescue_failure_is_frozen():
    result = s84.read_json(
        ROOT / "data/stage86a_qci_dirac3_rescue_failure_adjudication.json"
    )
    assert (
        result["status"]
        == "stage86_dirac3_physical_rescue_failed_stop_global_penalty_route"
    )
    assert result["execution"]["device_jobs"] == 1
    assert result["execution"]["sample_count"] == 25
    assert result["execution"]["device_usage_seconds"] == 22.0
    assert result["constraint_fidelity"]["cardinality_ok_count"] == 6
    assert result["constraint_fidelity"]["quality_ok_count"] == 13
    assert result["constraint_fidelity"]["receptor_constraints_ok_count"] == 1
    assert result["constraint_fidelity"]["fully_feasible_count"] == 0
    assert result["additional_qci_dirac3_global_penalty_jobs_authorized"] == 0


def test_stage86a_auxiliary_repair_is_valid_but_not_competitive():
    result = s84.read_json(
        ROOT / "data/stage86a_qci_dirac3_rescue_failure_adjudication.json"
    )
    repaired = result["canonical_auxiliary_repair_diagnostic"]
    assert repaired["canonical_repair_feasible"] is True
    assert repaired["raw_residual_l1"] == 1
    assert repaired["objective_gap"] > 1.9
    assert repaired["rank_among_feasible_subsets"] == 13540
    assert repaired["feasible_subset_count"] == 18552
    assert repaired["better_than_feasible_median"] is False
