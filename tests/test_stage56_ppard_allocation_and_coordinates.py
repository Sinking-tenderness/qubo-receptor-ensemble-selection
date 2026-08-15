import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text())


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_stage56_freezes_disjoint_balanced_pilot():
    summary = read_json("data/stage56_ppard_ligand_panel_allocation_summary.json")
    assert summary["status"] == "stage56_ppard_ligand_panels_and_pilot_frozen"
    assert summary["pilot"]["row_count"] == 96
    assert summary["pilot"]["active_count"] == 48
    assert summary["pilot"]["decoy_count"] == 48
    assert set(summary["pilot"]["fold_label_counts"].values()) == {12}
    assert all(summary["disjointness"].values())


def test_stage56a_identifies_numbering_not_biological_failure():
    result = read_json("data/stage56a_ppard_numbering_failure_adjudication_result.json")
    assert result["diagnosis"]["systematic_author_numbering_failure_confirmed"] is True
    assert result["diagnosis"]["biological_coordinate_gate_failure_established"] is False
    assert result["counts"]["systematic_minus36_failure_count"] == 34
    assert result["counts"]["sequence_mapping_pass_count"] == 51
    assert result["decision"]["threshold_lowering_authorized"] is False


def test_stage56b_keeps_raw_coordinates_and_thresholds_unchanged():
    remapping = read_json("data/stage56b_ppard_sequence_remapped_coordinates_result.json")
    audit = read_json("data/stage56b_ppard_allocation_and_coordinates_audit.json")
    assert remapping["raw_coordinates_modified"] is False
    assert remapping["thresholds_changed"] is False
    assert audit["coordinate_adjudication"]["coordinate_thresholds_exact"] is True
    assert audit["coordinate_adjudication"]["raw_coordinates_modified"] is False


def test_stage56b_retains_all_51_hard_gate_passing_structures():
    summary = read_json("data/stage56b_ppard_coordinate_pool_amendment01_summary.json")
    rows = read_csv(
        "data/processed/stage56b_ppard_coordinate_candidate_audit_amendment01.csv"
    )
    assert summary["status"] == "stage56_ppard_coordinate_pool_hard_gate_ok"
    assert summary["counts"]["coordinate_eligible_count"] == 51
    assert summary["counts"]["redocking_pool_count"] == 51
    assert len(rows) == 51
    assert {row["status"] for row in rows} == {"coordinate_eligible"}
    assert summary["selection_policy"]["max_min_or_outcome_informed_compression_used"] is False


def test_stage56b_does_not_authorize_pilot_or_protected_work():
    summary = read_json("data/stage56b_ppard_coordinate_pool_amendment01_summary.json")
    assert summary["decision"]["cognate_redocking_input_preparation_authorized"] is True
    assert summary["decision"]["pilot_production_docking_authorized"] is False
    assert summary["decision"]["full_training_matrix_authorized"] is False
    assert summary["decision"]["fresh_validation_release_authorized"] is False
    assert summary["decision"]["quantum_hardware_authorized"] is False
    assert summary["data_boundary"]["docking_scores_read"] == 0
