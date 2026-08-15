import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage112_preserves_stage110_and_corrects_only_thrb():
    result = json.loads((ROOT / "data/stage112_historical_candidate_pool_amendment01_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage112_historical_candidate_pool_amendment01_ok"
    assert result["historical_record"]["stage110_file_preserved"] is True
    assert result["historical_record"]["superseded_for_target_ids"] == ["THRB"]
    assert result["amended_thrb_entry"]["historical_metadata_eligible_count"] == 19
    assert result["amended_thrb_entry"]["corrected_uniprot_accession"] == "P00734"


def test_stage112_reopens_only_a_preregistration_without_releasing_compute():
    result = json.loads((ROOT / "data/stage112_historical_candidate_pool_amendment01_result.json").read_text(encoding="utf-8"))
    assert result["corrected_registry_state"]["remaining_outcome_unseen_candidates_requiring_new_preregistration"] == 1
    assert result["corrected_registry_state"]["protocol_eligibility_established"] is False
    assert all(value == 0 for value in result["data_boundary"].values())
    for key in ("new_target_source_download_authorized", "new_coordinate_audit_authorized", "new_docking_authorized", "quantum_hardware_authorized"):
        assert result["decision"][key] is False
