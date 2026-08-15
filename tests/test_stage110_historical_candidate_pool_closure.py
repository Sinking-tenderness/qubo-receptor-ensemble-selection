import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage110_closes_only_the_finite_historical_registry():
    result = json.loads((ROOT / "data/stage110_historical_candidate_pool_closure_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage110_historical_candidate_pool_closed"
    assert "not a global target-discovery claim" in result["scope"]
    assert result["counts"]["remaining_outcome_unseen_protocol_eligible_candidates"] == 0
    entries = {entry["target_id"]: entry for entry in result["registry_entries"]}
    assert entries["ESR2"]["status"] == "reference No-Go"
    assert entries["PARP1"]["status"] == "historical exploratory No-Go"
    assert entries["PPARD"]["status"] == "already used and closed"


def test_stage110_does_not_release_new_compute_or_hardware_work():
    result = json.loads((ROOT / "data/stage110_historical_candidate_pool_closure_result.json").read_text(encoding="utf-8"))
    assert all(value == 0 for value in result["data_boundary"].values())
    assert all(value is False for key, value in result["decision"].items() if key.endswith("_authorized"))
    audit = json.loads((ROOT / "data/stage110_historical_candidate_pool_closure_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "stage110_independent_audit_ok"
    assert audit["later_stages_locked"] is True
