import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="ascii"))


def test_stage88_has_four_targets_and_sixteen_folds():
    result = read_json(ROOT / "data/stage88_chemotype_balanced_portfolio_gate_result.json")
    assert result["summary"]["target_count"] == 4
    assert result["summary"]["fold_count"] == 16
    assert result["summary"]["minimum_global_cluster_count"] == 1
    assert result["chemotype_balanced_cqm_design_authorized"] is False


def test_stage88_never_authorizes_qaoa_or_hardware_directly():
    result = read_json(ROOT / "data/stage88_chemotype_balanced_portfolio_gate_result.json")
    assert result["constraint_preserving_qaoa_simulation_authorized"] is False
    assert result["new_quantum_hardware_jobs_authorized"] == 0
    assert result["new_docking_jobs_authorized"] == 0
    audit = read_json(ROOT / "data/stage88_chemotype_balanced_portfolio_gate_audit.json")
    assert audit["status"] == "stage88_chemotype_balanced_portfolio_gate_independent_audit_ok"
