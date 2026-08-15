import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="ascii"))


def test_stage87_finds_only_small_posthoc_greedy_traps():
    result = read_json(ROOT / "data/stage87_quantum_value_instance_gate_result.json")
    summary = result["candidate_summary"]
    assert summary["stage68_screened_cell_count"] == 240
    assert summary["biological_single_start_trap_candidate_count"] == 4
    assert summary["nontrivial_certified_candidate_count"] == 0
    assert summary["maximum_candidate_total_states"] == 38760
    assert summary["maximum_candidate_feasible_states"] == 415


def test_stage87_prior_strong_classical_evidence_has_no_certified_miss():
    result = read_json(ROOT / "data/stage87_quantum_value_instance_gate_result.json")
    summary = result["historical_hardness_summary"]
    assert summary["stage74_certified_exact_cell_count"] == 120
    assert summary["stage74_strong_classical_miss_count"] == 0
    assert summary["stage75_exact_frontier_cell_count"] == 20
    assert summary["stage75_joint_classical_miss_count"] == 0
    assert summary["stage80_local_subproblem_count"] == 100
    assert summary["stage80_multi_move_trap_count"] == 0


def test_stage87_blocks_qaoa_and_hardware():
    result = read_json(ROOT / "data/stage87_quantum_value_instance_gate_result.json")
    assert result["strict_instance_gate_passed"] is False
    assert result["constraint_preserving_qaoa_simulation_authorized"] is False
    assert result["new_quantum_hardware_jobs_authorized"] == 0
    assert result["new_docking_jobs_authorized"] == 0
    audit = read_json(ROOT / "data/stage87_quantum_value_instance_gate_audit.json")
    assert audit["status"] == "stage87_quantum_value_instance_gate_independent_audit_ok"
