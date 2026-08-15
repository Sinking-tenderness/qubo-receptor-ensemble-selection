import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage107_metadata_gate_records_the_legacy_reference_no_go():
    result = json.loads((ROOT / "data/stage107_parp1_contact_state_metadata_intake_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage107_parp1_contact_state_metadata_intake_no_go"
    assert result["counts"]["metadata_eligible_count"] < 25
    assert result["counts"]["reference_eligible"] is False
    assert result["counts"]["gate_passes"] is False
    assert "mutation_count_differs" in result["reference_metadata_record"]["exclusion_reasons"]
    assert result["decision"]["coordinate_structural_audit_authorized"] is False
    assert result["decision"]["ligand_preparation_authorized"] is False
    assert result["decision"]["redocking_authorized"] is False
    assert result["decision"]["production_docking_authorized"] is False
    assert result["decision"]["parp1_fresh_validation_released"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False
    assert all(value == 0 for value in result["data_boundary"].values())


def test_stage107_independent_audit_keeps_every_later_stage_locked():
    audit = json.loads((ROOT / "data/stage107_parp1_contact_state_metadata_intake_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "stage107_independent_audit_ok"
    assert audit["metadata_eligible_count"] < 25
    assert audit["structural_count_passes"] is False
    assert audit["reference_eligible"] is False
    assert audit["coordinate_structural_audit_authorized"] is False
    assert audit["later_stages_locked"] is True
