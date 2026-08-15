import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage111_records_the_historical_dude_slug_mapping_error_without_overwriting_history():
    result = json.loads((ROOT / "data/stage111_thrb_identity_adjudication_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage111_thrb_identity_mismatch_confirmed"
    assert result["historical_mapping"]["uniprot_accession"] == "P10828"
    assert result["authoritative_identity"]["dude_catalog_description"] == "Thrombin"
    assert result["authoritative_identity"]["rcsb_uniprot_accession"] == "P00734"
    assert result["decision"]["historical_record_overwritten"] is False
    assert result["decision"]["thrombin_new_preregistration_authorized"] is True


def test_stage111_does_not_release_thrombin_data_or_compute():
    result = json.loads((ROOT / "data/stage111_thrb_identity_adjudication_result.json").read_text(encoding="utf-8"))
    assert all(value == 0 for value in result["data_boundary"].values())
    for key in ("thrombin_source_download_authorized", "thrombin_coordinate_download_authorized", "thrombin_docking_authorized", "quantum_hardware_authorized"):
        assert result["decision"][key] is False
    audit = json.loads((ROOT / "data/stage111_thrb_identity_adjudication_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "stage111_independent_audit_ok"
    assert audit["later_compute_locked"] is True
