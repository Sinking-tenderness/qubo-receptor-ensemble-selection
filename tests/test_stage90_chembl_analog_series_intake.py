import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="ascii"))


def test_stage90_is_a_zero_compute_public_data_gate():
    result = read_json(ROOT / "data/stage90_chembl_analog_series_intake_result.json")
    assert result["summary"]["target_count"] == 4
    assert result["data_boundary"] == {
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
    assert result["authorization"]["new_docking_jobs_authorized"] == 0
    assert result["authorization"]["quantum_hardware_jobs_authorized"] == 0


def test_stage90_candidates_are_single_assay_endpoints():
    path = ROOT / "results/runs/stage90_chembl_analog_series_intake/assay_candidates.csv"
    with path.open(encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    keys = {
        (row["target_id"], row["assay_chembl_id"], row["standard_type"])
        for row in rows
    }
    assert len(keys) == len(rows)
    assert all(int(row["unique_molecule_count"]) > 0 for row in rows)


def test_stage90_authorization_matches_full_gate_count():
    result = read_json(ROOT / "data/stage90_chembl_analog_series_intake_result.json")
    expected = result["summary"]["full_rescue_candidate_count"] > 0
    assert result["authorization"]["stage91_ligand_freeze_authorized"] is expected
