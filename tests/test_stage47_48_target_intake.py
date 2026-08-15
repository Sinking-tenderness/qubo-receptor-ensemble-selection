import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage48_independent_audit(tmp_path):
    module = load_script("stage48_audit", "scripts/audit_stage48_ppara_source_result.py")
    result = module.audit(ROOT, tmp_path / "audit.json")
    assert result["status"] == "stage48_ppara_source_independent_audit_ok"
    assert result["metadata_eligible_count"] == 75
    assert result["primary_subset_size"] == 6


def test_stage47b_selected_ppara():
    import json

    result = json.loads((ROOT / "data/stage47b_expanded_new_target_feasibility_screen_result.json").read_text())
    assert result["selected_target"]["target_id"] == "PPARA"
    assert result["selected_target"]["metadata_eligible_count"] == 75
    assert result["decision"]["new_target_source_intake_authorized"] is True


def test_src_branch_closed_without_docking():
    import json

    result = json.loads((ROOT / "data/stage46b_src_human_metadata_pool_failure_adjudication.json").read_text())
    assert result["decision"]["close_src_branch"] is True
    assert result["query"]["metadata_eligible_count"] == 10
    assert result["data_boundary"]["new_docking_jobs"] == 0
