import hashlib
import json
from pathlib import Path

import pytest

import scripts.experimental.quantum.run_stage79_qci_dirac3_poc as hw


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage79_implementations_are_hash_locked_and_audited():
    config = read_json("configs/stage79_qci_dirac3_local_move_qubo_poc.json")
    for descriptor in config["implementation"].values():
        path = ROOT / descriptor["path"]
        assert sha256(path) == descriptor["sha256"]
        assert path.stat().st_size == descriptor["size_bytes"]
    result = read_json("data/stage79_qci_dirac3_poc_result.json")
    audit = read_json("data/stage79_qci_dirac3_poc_audit.json")
    assert result["status"].endswith("poc_frozen")
    assert audit["status"].endswith("independent_audit_ok")
    assert audit["instances_audited"] == 6
    assert audit["maximum_coefficient_error"] == pytest.approx(0.0)
    assert audit["qci_device_jobs_observed"] == 0


def test_stage79_translation_is_local_and_float32_safe():
    result = read_json("data/stage79_qci_dirac3_poc_result.json")
    summary = result["instance_summary"]
    assert summary["instance_count"] == 6
    assert summary["confirmation_positive_count"] == 2
    assert summary["confirmation_negative_count"] == 3
    assert summary["calibration_diagnostic_count"] == 1
    assert summary["all_warm_states_are_zero"] is True
    assert summary["all_float32_role_checks_passed"] is True
    for item in result["instances"]:
        mapping = read_json(item["qci_mapping"]["path"])
        payload = read_json(item["qci_polynomial"]["path"])
        polynomial = payload["file_config"]["polynomial"]
        assert mapping["warm_bit_vector"] == [0] * item["logical_variable_count"]
        assert mapping["num_levels"] == [2] * item["logical_variable_count"]
        assert polynomial["min_degree"] == 1
        assert polynomial["max_degree"] == 2
        assert all(term["idx"] != [0, 0] for term in polynomial["data"])


def test_stage79_external_bundle_validates_without_qci_access():
    result = read_json("data/stage79_qci_dirac3_poc_result.json")
    validation = hw.local_validate(ROOT, result)
    assert validation["status"] == "stage79_local_execution_bundle_valid"
    assert validation["instance_count"] == 6
    assert validation["cloud_queries"] == 0
    assert validation["qci_device_jobs"] == 0
    assert all(row["warm_is_all_zero"] for row in validation["instances"])
    assert all(row["exact_certificate"] for row in validation["instances"])


def test_stage79_device_execution_has_a_double_interlock(monkeypatch):
    monkeypatch.delenv("STAGE79_QCI_ACK", raising=False)
    with pytest.raises(PermissionError):
        hw.device_authorized(True)
    monkeypatch.setenv("STAGE79_QCI_ACK", hw.DEVICE_ACKNOWLEDGEMENT)
    with pytest.raises(PermissionError):
        hw.device_authorized(False)
    hw.device_authorized(True)


def test_stage79_response_is_recomputed_with_original_bqm():
    result = read_json("data/stage79_qci_dirac3_poc_result.json")
    instances = hw.load_instances(ROOT, result)
    positive = next(
        item for item in instances if item["mapping"]["role"] == "confirmation_positive"
    )
    variables = positive["mapping"]["variable_order"]
    exact_names = set(
        positive["metadata"]["exact_reference"]["selected_move_variables"]
    )
    exact = [int(name in exact_names) for name in variables]
    response = {
        "status": "COMPLETED",
        "results": {
            "solutions": [[0] * len(variables), exact],
            "energies": [12345.0, 67890.0],
            "counts": [2, 3],
        },
        "job_info": {"job_id": "mock", "job_result": {"device_usage_s": 1}},
    }
    rows, summary = hw.evaluate_response(positive, response, schedule=2)
    assert len(rows) == 2
    assert summary["sample_count"] == 5
    assert summary["strict_improvement_sample_count"] == 3
    assert summary["exact_optimum_sample_count"] == 3
    assert rows[1]["device_reported_energy"] == 67890.0
    assert rows[1]["float64_energy"] == pytest.approx(
        positive["metadata"]["exact_reference"]["objective_energy"]
    )


def test_stage79_free_protocol_fits_documented_trial_budget():
    config = read_json("configs/stage79_qci_dirac3_local_move_qubo_poc.json")
    protocol = config["hardware_protocol"]
    assert protocol["required_initial_free_allocation_seconds"] == 300
    assert protocol["planned_total_job_count"] == 9
    assert protocol["planned_total_sample_count"] == 600
    assert protocol["maximum_recorded_device_usage_seconds"] == 480
    assert protocol["require_unpaid_allocation"] is True
    source = (
        ROOT / "scripts/experimental/quantum/run_stage79_qci_dirac3_poc.py"
    ).read_text(encoding="ascii")
    assert "QCI_TOKEN" in source
    assert "<your_secret_token>" not in source
