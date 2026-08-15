import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage106_contact_audit_records_the_frozen_positive_and_negative_cases():
    result = json.loads((ROOT / "data/stage106_cocrystal_contact_state_audit_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage106_cocrystal_contact_state_audit_complete"
    positive = result["focal_pairs"]["FA10_positive"]
    assert positive["stage102a_fixed_k2_selection_count"] == 5
    assert positive["stage102a_fixed_k2_min_outer_gain"] > 0.0
    assert result["feasibility"]["fa10_positive_pair_is_not_selected_for_extreme_structural_distance"]
    assert result["feasibility"]["contact_state_signal_is_not_authorized_as_predictor"]
    assert all(value == 0 for value in result["data_boundary"].values())


def test_stage106_contact_and_pair_outputs_are_complete_and_bounded():
    contact_path = ROOT / "results/runs/stage106_cocrystal_contact_state_audit/receptor_contacts.csv"
    pair_path = ROOT / "results/runs/stage106_cocrystal_contact_state_audit/pair_diagnostics.csv"
    with contact_path.open(encoding="utf-8", newline="") as handle:
        contacts = list(csv.DictReader(handle))
    with pair_path.open(encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(contacts) == 25
    assert len(pairs) == 144
    assert all(int(row["protein_contact_residue_count"]) > 0 for row in contacts)
    assert all(0.0 <= float(row["contact_jaccard_distance"]) <= 1.0 for row in pairs)


def test_stage106_independent_audit_keeps_the_protocol_locked():
    audit = json.loads((ROOT / "data/stage106_cocrystal_contact_state_audit_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "stage106_independent_audit_ok"
    assert audit["contact_record_count"] == 25
    assert audit["pair_record_count"] == 144
    assert audit["new_target_protocol_authorized"] is False
    assert audit["parp1_released"] is False
    assert audit["quantum_hardware_authorized"] is False
