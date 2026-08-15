import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="ascii"))


def test_stage90a_freezes_four_disjoint_assay_roles():
    result = read_json(ROOT / "data/stage90a_bace1_rescue_source_freeze_result.json")
    assert result["status"] == "stage90a_bace1_rescue_source_freeze_passed"
    assert [row["role"] for row in result["frozen_assays"]] == [
        "development",
        "confirmation_a",
        "confirmation_b",
        "locked_test",
    ]
    assert len({row["assay_chembl_id"] for row in result["frozen_assays"]}) == 4
    assert len({row["document_chembl_id"] for row in result["frozen_assays"]}) == 4


def test_stage90a_preserves_stage90_failure_and_prior_large_pool_evidence():
    result = read_json(ROOT / "data/stage90a_bace1_rescue_source_freeze_result.json")
    assert result["checks"]["stage90_remains_failed"] is True
    assert result["large_pool_certificate"] == {
        "receptor_count": 34,
        "total_states_k1_to_k6": 1676115,
        "k6_states": 1344904,
        "evidence_timing": "Stage41d certificate existed before Stage90 ChEMBL intake.",
    }


def test_stage90a_has_zero_cross_role_overlap_and_zero_compute():
    with (
        ROOT
        / "results/runs/stage90a_bace1_rescue_source_freeze/assay_overlap_matrix.csv"
    ).open(encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert all(int(row["shared_molecule_count"]) == 0 for row in rows)
    assert all(int(row["shared_scaffold_count"]) == 0 for row in rows)
    assert all(int(row["shared_document_count"]) == 0 for row in rows)
    result = read_json(ROOT / "data/stage90a_bace1_rescue_source_freeze_result.json")
    assert result["authorization"]["new_docking_jobs_authorized"] == 0
    assert result["authorization"]["quantum_hardware_jobs_authorized"] == 0
