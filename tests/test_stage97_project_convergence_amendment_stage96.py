import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage97_preserves_stage89_and_adds_negative_stage96_evidence():
    result = json.loads((ROOT / "data/stage97_project_convergence_amendment_stage96_result.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "data/stage97_project_convergence_amendment_stage96_audit.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage97_project_convergence_amended_stage96"
    assert result["base_freeze"] == "stage89_project_converged_claims_frozen"
    assert result["stage96_policy_gate_passed"] is False
    assert result["stage96_solver_value_passed"] is False
    assert audit["status"] == "stage97_audit_ok"
    assert audit["stage89_preserved"] is True
    assert audit["new_docking_jobs"] == 0
    assert audit["quantum_hardware_jobs"] == 0
