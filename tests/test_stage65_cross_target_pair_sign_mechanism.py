import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_stage65_cross_target_pair_sign_mechanism import candidate_pair


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage65_implementation_and_sources_are_hash_locked():
    config = read_json("configs/stage65_cross_target_pair_sign_mechanism.json")
    for section in ("implementation", "inputs"):
        for value in config[section].values():
            assert sha256(ROOT / value["path"]) == value["sha256"]


def test_stage65_candidate_sign_and_lcb_construction():
    statistics = {
        "full_pair": np.array([[0.0, 0.4, -0.3], [0.4, 0.0, 0.1], [-0.3, 0.1, 0.0]]),
        "median_pair": np.array([[0.0, 0.3, -0.2], [0.3, 0.0, 0.05], [-0.2, 0.05, 0.0]]),
        "pair_spread": np.array([[0.0, 0.1, 0.1], [0.1, 0.0, 0.1], [0.1, 0.1, 0.0]]),
        "positive_support": np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.5], [0.0, 0.5, 0.0]]),
        "negative_support": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    }
    base = {
        "pair_scale": 0.25,
        "sign_support_threshold": 0.75,
        "lambda_mad": 1.0,
    }
    positive = candidate_pair(statistics, {**base, "mode": "lcb_positive"})
    negative = candidate_pair(statistics, {**base, "mode": "lcb_negative"})
    assert np.allclose(positive, [[0.0, 0.05, 0.0], [0.05, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(negative, [[0.0, 0.0, -0.025], [0.0, 0.0, 0.0], [-0.025, 0.0, 0.0]])


def test_stage65_dimensions_and_pair_off_reproduction():
    result = read_json("data/stage65_cross_target_pair_sign_mechanism_result.json")
    assert result["status"] == "stage65_cross_target_pair_sign_mechanism_complete"
    assert result["candidate_count"] == 11
    assert result["edge_transfer_row_count"] == 22868
    assert result["fixed_k_metric_count"] == 1056
    assert result["pair_off_reproduction_cell_count"] == 96
    assert result["analysis_payload_sha256"] == (
        "F0FA61801B7EDB2E5AED184BE71D6878770E81994DDE73048A4101969FDCC413"
    )


def test_stage65_edge_signal_exists_but_is_not_uniform():
    result = read_json("data/stage65_cross_target_pair_sign_mechanism_result.json")
    edge = result["edge_transfer"]
    assert edge["mean_fold_spearman"] == pytest.approx(0.2046417446151111)
    assert edge["negative_fold_spearman_count"] == 3
    assert edge["lcb_positive_edge_count"] == 10929
    assert edge["all_edge_holdout_positive_rate"] == pytest.approx(
        0.5516442189959769
    )
    assert edge["lcb_positive_holdout_positive_rate"] == pytest.approx(
        0.620733827431604
    )
    assert edge["lcb_positive_precision_advantage"] == pytest.approx(
        0.06908960843562706
    )
    targets = {
        row["target_id"]: row
        for row in read_csv(
            "results/runs/stage65_cross_target_pair_sign_mechanism/edge_target_summary.csv"
        )
    }
    assert float(targets["PPARA"]["train_holdout_pair_residual_spearman"]) < 0.0
    assert float(targets["PPARD"]["train_holdout_pair_residual_spearman"]) > 0.0


def test_stage65_positive_terms_drive_the_harmful_subset_changes():
    rows = {
        row["candidate_id"]: row
        for row in read_csv(
            "results/runs/stage65_cross_target_pair_sign_mechanism/global_summary.csv"
        )
    }
    assert float(rows["negative_0p25"]["mean_target_gain_over_pair_off"]) == 0.0
    assert float(rows["positive_0p25"]["mean_target_gain_over_pair_off"]) == pytest.approx(
        float(rows["signed_0p25"]["mean_target_gain_over_pair_off"])
    )
    assert float(rows["positive_0p25"]["mean_target_gain_over_pair_off"]) == pytest.approx(
        -0.008570297129599244
    )
    assert int(rows["positive_0p25"]["nonnegative_target_count_over_pair_off"]) == 1


def test_stage65_primary_candidate_fails_performance_gate_and_redirects():
    result = read_json("data/stage65_cross_target_pair_sign_mechanism_result.json")
    primary = result["primary_candidate"]
    assert primary["candidate_id"] == "lcb_positive_0p25"
    assert primary["mean_target_gain_over_pair_off"] == pytest.approx(
        -0.007060074779885409
    )
    assert primary["worst_target_gain_over_pair_off"] == pytest.approx(
        -0.027054231216438053
    )
    assert primary["nonnegative_target_count_over_pair_off"] == 1
    assert result["decision_gate"]["pair_residual_route_supported"] is False
    assert result["decision"]["positive_pair_objective_freeze_authorized"] is False
    assert result["decision"]["auxiliary_coverage_qubo_design_authorized"] is True
    assert result["decision"]["fresh_validation_authorized"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False


def test_stage65_independent_audit_passed():
    audit = read_json("data/stage65_cross_target_pair_sign_mechanism_audit.json")
    assert audit["status"] == (
        "stage65_cross_target_pair_sign_mechanism_independent_audit_ok"
    )
    assert audit["pair_off_reproduction_cells_independently_verified"] == 96
    assert audit["edge_summaries_independently_recomputed"] is True
    assert audit["candidate_summaries_independently_recomputed"] is True
    assert audit["decision_gate_independently_recomputed"] is True
    assert audit["pair_residual_route_supported"] is False
    assert audit["source_result"]["sha256"] == sha256(
        ROOT / "data/stage65_cross_target_pair_sign_mechanism_result.json"
    )
