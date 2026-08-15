import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage108_records_that_catalytic_domain_scope_does_not_meet_the_frozen_pool_size():
    result = json.loads((ROOT / "data/stage108_parp1_catalytic_domain_reference_feasibility_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage108_parp1_catalytic_domain_reference_feasibility_no_go"
    assert result["scope"]["protein_scope"] == "catalytic_domain"
    assert result["counts"]["metadata_eligible_count"] == 10
    assert result["counts"]["minimum_required_count"] == 16
    assert result["counts"]["pool_gate_passes"] is False
    assert result["provisional_reference"]["pdb_id"] == "7KK5"
    assert all(value == 0 for value in result["data_boundary"].values())


def test_stage108_keeps_all_downstream_work_locked_and_writes_the_complete_candidate_table():
    result = json.loads((ROOT / "data/stage108_parp1_catalytic_domain_reference_feasibility_result.json").read_text(encoding="utf-8"))
    decision = result["decision"]
    assert decision["coordinate_structural_audit_authorized"] is False
    assert decision["ligand_preparation_authorized"] is False
    assert decision["redocking_authorized"] is False
    assert decision["production_docking_authorized"] is False
    assert decision["fresh_validation_released"] is False
    assert decision["quantum_hardware_authorized"] is False
    with (ROOT / "data/processed/stage108_parp1_catalytic_domain_metadata_eligible_candidates.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert rows[0]["pdb_id"] == "7KK5"
