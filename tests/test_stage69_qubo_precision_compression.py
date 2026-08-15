import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage69_implementation_and_sources_are_hash_locked():
    config = read_json("configs/stage69_qubo_precision_compression.json")
    for section in ("implementation", "inputs"):
        for value in config[section].values():
            assert sha256(ROOT / value["path"]) == value["sha256"]


def test_stage69_dimensions_and_milp_certificates_are_complete():
    result = read_json("data/stage69_qubo_precision_compression_result.json")
    assert result["status"] == "stage69_qubo_precision_compression_complete"
    assert result["scale_count"] == 8
    assert result["cell_metric_count"] == 640
    assert result["continuous_milp_certificate_count"] == 80
    assert result["quantized_milp_certificate_count"] == 585
    assert result["analysis_payload_sha256"] == (
        "804361702852746488871D307BB19AFF49669C5AFF677731386E0A0A086A932D"
    )


def test_stage69_scale_feasibility_progression_is_frozen():
    rows = read_csv(
        "results/runs/stage69_qubo_precision_compression/scale_summary.csv"
    )
    assert [int(row["quality_integer_scale"]) for row in rows] == [
        31,
        63,
        127,
        255,
        511,
        1023,
        2047,
        4095,
    ]
    assert [int(row["feasible_cell_count"]) for row in rows] == [
        48,
        65,
        74,
        78,
        80,
        80,
        80,
        80,
    ]
    assert all(row["compression_gate_passed"] == "False" for row in rows)


def test_stage69_scale511_is_high_fidelity_but_misses_dynamic_range_gate():
    rows = read_csv(
        "results/runs/stage69_qubo_precision_compression/scale_summary.csv"
    )
    scale511 = next(row for row in rows if row["quality_integer_scale"] == "511")
    assert int(scale511["exact_subset_match_count"]) == 78
    assert float(scale511["mean_subset_jaccard_vs_continuous"]) == pytest.approx(
        0.9922619047619048
    )
    assert float(scale511["minimum_subset_jaccard_vs_continuous"]) == pytest.approx(
        2.0 / 3.0
    )
    assert float(scale511["mean_absolute_holdout_bedroc_gap"]) == pytest.approx(
        0.00038526417795890083
    )
    assert float(scale511["maximum_absolute_holdout_bedroc_gap"]) == pytest.approx(
        0.0295036252645684
    )
    assert float(scale511["maximum_coefficient_dynamic_range"]) == pytest.approx(
        748508.7942006803
    )
    assert float(scale511["dynamic_range_compression_factor_vs_4095"]) == (
        pytest.approx(64.20596205962059)
    )
    assert float(scale511["minimum_actual_quality_floor_margin"]) >= 0.0


def test_stage69_negative_gate_is_kept_separate_from_near_miss():
    result = read_json("data/stage69_qubo_precision_compression_result.json")
    assert result["selected_compression"] == {}
    near_miss = result["best_uniform_near_miss"]
    assert near_miss["quality_integer_scale"] == 511
    assert near_miss["feasible_cell_count"] == 80
    assert result["compression_gate"]["compressed_qubo_freeze_authorized"] is False
    assert result["decision"]["compact_solver_prototype_authorized"] is False
    assert result["direct_qpu_gate"]["direct_qpu_precision_gate_passed"] is False
    assert result["decision"]["direct_qpu_execution_authorized"] is False
    assert result["decision"]["new_target_preregistration_remains_authorized"] is True
    assert result["decision"]["quantum_advantage_claim_authorized"] is False
    assert result["data_boundary"]["fresh_validation_rows_read"] == 0
    assert result["data_boundary"]["locked_test_rows_read"] == 0
    assert result["data_boundary"]["new_docking_jobs"] == 0
    assert result["data_boundary"]["quantum_hardware_jobs"] == 0


def test_stage69_diagnostic_models_and_independent_audit_passed():
    model = read_json("data/stage69_qubo_precision_compression_model_record.json")
    assert model["selected_quality_integer_scale"] == 0
    assert model["diagnostic_model_quality_integer_scale"] == 511
    assert model["model_role"] == "diagnostic_uniform_near_miss"
    assert model["model_count"] == 16
    assert all(record["energy_residual"] == pytest.approx(0.0) for record in model["models"])

    audit = read_json("data/stage69_qubo_precision_compression_audit.json")
    assert audit["status"] == (
        "stage69_qubo_precision_compression_independent_audit_ok"
    )
    assert audit["cell_metrics_independently_checked"] == 640
    assert audit["scale_summaries_independently_recomputed"] == 8
    assert audit["selected_scale_independently_verified"] == 0
    assert audit["diagnostic_near_miss_scale_independently_verified"] == 511
    assert audit["factorized_qubo_models_independently_checked"] == 16
    assert audit["compressed_qubo_freeze_authorized"] is False
    assert audit["direct_qpu_execution_authorized"] is False
    assert audit["source_result"]["sha256"] == sha256(
        ROOT / "data/stage69_qubo_precision_compression_result.json"
    )
