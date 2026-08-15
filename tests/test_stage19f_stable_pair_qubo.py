import json
from pathlib import Path

import numpy as np

from scripts.diagnose_stage19f_stable_pair_qubo import (
    minmax,
    selected_pair_count,
    stable_pair_coefficients,
)
from scripts.prepare_receptor import file_sha256
from scripts.screen_stage10_mk14_expanded16_qubo_greedy import (
    fixed_cardinality_exact,
)


def test_minmax_preserves_order_and_handles_constant_values() -> None:
    assert minmax(np.asarray([2.0, 4.0, 3.0])).tolist() == [0.0, 1.0, 0.5]
    assert minmax(np.asarray([7.0, 7.0])).tolist() == [0.0, 0.0]


def test_stable_pair_gate_removes_inconsistent_synergy() -> None:
    stats = {
        "receptor_ids": ["R1", "R2", "R3"],
        "pairs": [("R1", "R2"), ("R1", "R3"), ("R2", "R3")],
        "block_ids": [0, 1, 2],
        "linear_raw": np.asarray([0.7, 0.8, 0.6]),
        "linear_normalized": np.asarray([0.5, 1.0, 0.0]),
        "pair_synergy_mean": np.asarray([0.08, 0.04, 0.03]),
        "pair_synergy_std": np.asarray([0.01, 0.08, 0.01]),
        "pair_positive_fraction": np.asarray([1.0, 1.0 / 3.0, 1.0]),
    }
    coefficients, evidence = stable_pair_coefficients(
        stats,
        pair_weight=1.0,
        risk_kappa=1.0,
        target_size=2,
        cardinality_penalty=20.0,
        minimum_positive_block_fraction=2.0 / 3.0,
        minimum_pair_lcb=0.002,
    )

    stable = coefficients["utility"]["pair_stable_normalized"]
    assert stable["R1__R2"] == 1.0
    assert stable["R1__R3"] == 0.0
    assert stable["R2__R3"] > 0.0
    assert evidence["retained_pair_count"] == 2
    subset, _ = fixed_cardinality_exact(coefficients, ["R1", "R2", "R3"], 2)
    assert subset == ("R1", "R2")
    assert selected_pair_count(subset, coefficients) == 1


def test_stage19f_config_freezes_meaningful_pair_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/stage19f_cross_target_stable_pair_qubo.json"
    config = json.loads(config_path.read_text(encoding="ascii"))

    assert file_sha256(root / config["implementation"]["path"]) == (
        config["implementation"]["sha256"]
    )
    assert config["evidence_timing"]["bace1_benchmark_docking_started"] is False
    assert config["diagnostic"]["minimum_pair_lcb"] == 0.002
    assert config["development_support_gate"][
        "minimum_folds_with_selected_nonzero_pair"
    ] == 5
    assert config["development_support_gate"]["all_checks_required"] is True


def test_stage19f_result_preserves_failed_development_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    result_path = root / "data/stage19f_cross_target_stable_pair_qubo_result.json"
    result = json.loads(result_path.read_text(encoding="ascii"))

    assert result["status"] == (
        "stage19f_stable_pair_qubo_not_supported_do_not_amend_bace1"
    )
    assert result["gate"]["passed"] is False
    assert result["gate"]["bace1_amendment_authorized"] is False
    assert result["gate"]["folds_with_selected_nonzero_pair"] == 5
    assert all(
        value is False
        for value in result["gate"]["comparison_checks"].values()
    )
    assert result["data_boundary"]["bace1_docking_rows_read"] == 0


def test_stage19f_independent_audit_identifies_frozen_result() -> None:
    root = Path(__file__).resolve().parents[1]
    audit = json.loads(
        (root / "data/stage19f_cross_target_stable_pair_qubo_audit.json").read_text(
            encoding="ascii"
        )
    )

    assert audit["status"] == "stage19f_cross_target_stable_pair_qubo_audit_ok"
    assert audit["checks"]["failed_gate_reproduced"] is True
    assert audit["checks"]["bace1_amendment_authorized"] is False
    assert audit["coverage"]["inner_candidate_rows_reselected"] == 480
    result_path = root / audit["result"]["path"]
    assert file_sha256(result_path) == audit["result"]["sha256"]
