import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_stage70_constraint_aware_qubo_encoding import (
    bounded_slack_weights,
    slack_interval_is_complete,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage70_implementation_and_sources_are_hash_locked():
    config = read_json("configs/stage70_constraint_aware_qubo_encoding.json")
    for section in ("implementation", "inputs"):
        for value in config[section].values():
            assert sha256(ROOT / value["path"]) == value["sha256"]


@pytest.mark.parametrize("maximum,cap", [(0, 4), (1, 4), (37, 4), (78, 16), (250, 64)])
def test_stage70_bounded_slack_represents_every_integer(maximum, cap):
    weights = bounded_slack_weights(maximum, cap)
    assert sum(weights) == maximum
    assert all(weight <= cap for weight in weights)
    assert slack_interval_is_complete(weights, maximum)


def test_stage70_dimensions_and_payload_are_frozen():
    result = read_json("data/stage70_constraint_aware_qubo_encoding_result.json")
    assert result["status"] == "stage70_constraint_aware_qubo_encoding_complete"
    assert result["candidate_count"] == 5
    assert result["cell_metric_count"] == 400
    assert result["analysis_payload_sha256"] == (
        "C5599A9BA7620D89677F57FAFEE50C1893705605A58757497C391B9CFE1EB1C5"
    )


def test_stage70_candidate_progression_and_selection_are_frozen():
    rows = read_csv(
        "results/runs/stage70_constraint_aware_qubo_encoding/candidate_summary.csv"
    )
    assert [int(row["slack_weight_cap"]) for row in rows] == [4, 8, 16, 32, 64]
    assert [float(row["maximum_coefficient_dynamic_range"]) for row in rows] == (
        pytest.approx(
            [
                193345.0345427251,
                193345.0345427251,
                180349.04784656098,
                180349.04784656098,
                180349.04784656098,
            ]
        )
    )
    assert [int(row["maximum_slack_variable_count"]) for row in rows] == [
        33,
        18,
        11,
        8,
        7,
    ]
    assert all(row["encoding_gate_passed"] == "True" for row in rows)
    selected = read_json(
        "data/stage70_constraint_aware_qubo_encoding_result.json"
    )["selected_encoding"]
    assert selected["candidate_id"] == "tight_cap16_centered_pair_upper"
    assert selected["slack_weight_cap"] == 16


def test_stage70_selected_encoding_preserves_fidelity_and_exact_penalty():
    selected = read_json(
        "data/stage70_constraint_aware_qubo_encoding_result.json"
    )["selected_encoding"]
    assert selected["analytic_exact_penalty_certificate_count"] == 80
    assert selected["exact_subset_match_count_vs_continuous"] == 78
    assert selected["mean_subset_jaccard_vs_continuous"] == pytest.approx(
        0.9922619047619048
    )
    assert selected["mean_absolute_holdout_bedroc_gap"] == pytest.approx(
        0.00038526417795890083
    )
    assert selected["maximum_absolute_holdout_bedroc_gap"] == pytest.approx(
        0.0295036252645684
    )
    assert selected["minimum_analytic_invalid_state_gap_lower_bound"] >= (
        1.0 - 1e-12
    )
    assert selected["maximum_factorized_energy_residual"] == pytest.approx(0.0)


def test_stage70_selected_encoding_reduces_range_but_not_to_direct_qpu_gate():
    result = read_json("data/stage70_constraint_aware_qubo_encoding_result.json")
    selected = result["selected_encoding"]
    assert selected["maximum_coefficient_dynamic_range"] == pytest.approx(
        180349.04784656098
    )
    assert selected["dynamic_range_improvement_factor_vs_stage69"] == pytest.approx(
        4.150334050210809
    )
    assert selected["maximum_logical_variable_count"] == 101
    assert selected["maximum_quadratic_coefficient_count"] == 5050
    assert result["encoding_gate"]["compact_logical_qubo_freeze_authorized"] is True
    assert result["decision"]["coefficient_noise_simulation_authorized"] is True
    assert result["direct_qpu_gate"]["direct_qpu_precision_gate_passed"] is False
    assert result["decision"]["direct_qpu_execution_authorized"] is False
    assert result["decision"]["new_target_preregistration_remains_authorized"] is True
    assert result["decision"]["quantum_advantage_claim_authorized"] is False


def test_stage70_all_rows_have_complete_exact_certificates():
    rows = read_csv(
        "results/runs/stage70_constraint_aware_qubo_encoding/cell_metrics.csv"
    )
    assert len(rows) == 400
    assert all(row["slack_interval_complete"] == "True" for row in rows)
    assert all(row["analytic_exact_penalty_certificate"] == "True" for row in rows)
    assert all(float(row["factorized_energy_residual"]) <= 1e-10 for row in rows)
    assert all(
        float(row["analytic_invalid_state_gap_lower_bound"]) >= 1.0 - 1e-12
        for row in rows
    )
    assert all(float(row["source_actual_quality_floor_margin"]) >= 0.0 for row in rows)
    assert all(
        float(row["cardinality_penalty"])
        == pytest.approx(float(row["pair_off_redundancy_upper_bound"]) + 1.0)
        for row in rows
    )


def test_stage70_model_record_and_independent_audit_passed():
    model = read_json("data/stage70_constraint_aware_qubo_encoding_model_record.json")
    assert model["selected_candidate_id"] == "tight_cap16_centered_pair_upper"
    assert model["reference_k"] == 3
    assert model["model_count"] == 16
    assert all(
        record["factorized_energy_residual"] == pytest.approx(0.0)
        for record in model["models"]
    )

    audit = read_json("data/stage70_constraint_aware_qubo_encoding_audit.json")
    assert audit["status"] == (
        "stage70_constraint_aware_qubo_encoding_independent_audit_ok"
    )
    assert audit["cell_metrics_independently_checked"] == 400
    assert audit["candidate_summaries_independently_recomputed"] == 5
    assert audit["selected_candidate_independently_verified"] == (
        "tight_cap16_centered_pair_upper"
    )
    assert audit["selected_slack_weight_cap_independently_verified"] == 16
    assert audit["factorized_qubo_models_independently_checked"] == 16
    assert audit["compact_logical_qubo_freeze_authorized"] is True
    assert audit["coefficient_noise_simulation_authorized"] is True
    assert audit["direct_qpu_execution_authorized"] is False
    assert audit["source_result"]["sha256"] == sha256(
        ROOT / "data/stage70_constraint_aware_qubo_encoding_result.json"
    )
