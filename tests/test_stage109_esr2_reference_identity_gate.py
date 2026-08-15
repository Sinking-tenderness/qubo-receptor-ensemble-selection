import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage109_records_the_esr2_dude_reference_mutation_no_go():
    result = json.loads((ROOT / "data/stage109_esr2_reference_identity_gate_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage109_esr2_reference_identity_no_go"
    assert result["ranking_provenance"]["historical_eligible_order"] == ["PPARA", "PPARD", "ESR2"]
    assert result["reference_metadata"]["pdb_id"] == "2FSZ"
    assert result["reference_metadata"]["mutation_count"] == 3
    assert result["reference_metadata"]["pdbx_mutation_note"] == "C334S, C369S, C481S"
    assert result["counts"]["metadata_eligible_receptor_count"] == 32
    assert result["gate_checks"] == {"reference_is_wild_type": False, "reference_is_metadata_eligible": False, "receptor_pool_passes": True, "gate_passes": False}


def test_stage109_keeps_the_esr2_branch_before_all_downstream_work():
    result = json.loads((ROOT / "data/stage109_esr2_reference_identity_gate_result.json").read_text(encoding="utf-8"))
    boundary = result["data_boundary"]
    assert boundary["source_active_lines_counted"] == 367
    assert boundary["source_decoy_lines_counted"] == 20199
    assert boundary["source_label_values_parsed_or_used"] == 0
    assert all(value == 0 for key, value in boundary.items() if key not in {"source_active_lines_counted", "source_decoy_lines_counted"})
    assert all(value is False for key, value in result["decision"].items() if key.endswith("_authorized"))
    audit = json.loads((ROOT / "data/stage109_esr2_reference_identity_gate_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "stage109_independent_audit_ok"
    assert audit["later_stages_locked"] is True
