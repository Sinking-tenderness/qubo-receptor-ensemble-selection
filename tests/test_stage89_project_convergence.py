import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="ascii"))


def test_stage89_freezes_a_boundary_study_not_an_advantage_claim():
    result = read_json(ROOT / "data/stage89_project_convergence_result.json")
    assert result["status"] == "stage89_project_converged_claims_frozen"
    assert result["project_positioning"] == "feasibility_and_boundary_study"
    assert result["supported_claim_count"] == 3
    assert result["authorization"]["new_objective_search_authorized"] is False
    assert result["authorization"]["new_target_docking_authorized"] is False
    assert result["authorization"]["new_quantum_hardware_jobs_authorized"] is False
    assert result["authorization"]["manuscript_preparation_authorized"] is True


def test_stage89_preserves_the_positive_and_negative_evidence():
    result = read_json(ROOT / "data/stage89_project_convergence_result.json")
    metrics = result["critical_results"]
    assert metrics["mk14_primary_bedroc"]["qubo"] == metrics["mk14_primary_bedroc"]["greedy"]
    assert metrics["mk14_primary_bedroc"]["qubo"] > metrics["mk14_primary_bedroc"]["single_receptor"]
    assert metrics["stage79_physical_confirmation_optimum_hits"] == "500/500"
    assert metrics["stage80_multi_move_local_traps"] == 0
    assert metrics["stage86_fully_feasible_physical_samples"] == 0
    assert metrics["stage87_quantum_worthy_instance_gate_passed"] is False


def test_stage89_claim_ledger_blocks_overclaiming():
    path = ROOT / "results/runs/stage89_project_convergence/claim_evidence_matrix.csv"
    with path.open(encoding="ascii", newline="") as handle:
        rows = {row["claim_id"]: row for row in csv.DictReader(handle)}
    assert rows["C1"]["status"] == "supported_with_scope"
    assert rows["C2"]["status"] == "not_supported"
    assert rows["C4"]["status"] == "supported_as_poc"
    assert rows["C7"]["status"] == "not_established"
    assert rows["C8"]["status"] == "not_tested"


def test_stage89_independent_audit_passes():
    audit = read_json(ROOT / "data/stage89_project_convergence_audit.json")
    assert audit["status"] == "stage89_project_convergence_independent_audit_ok"
    assert audit["check_count"] == 11
    assert audit["failed_checks"] == []
    assert all(audit["checks"].values())
