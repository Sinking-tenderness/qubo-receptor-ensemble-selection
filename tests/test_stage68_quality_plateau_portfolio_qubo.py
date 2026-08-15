import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_stage68_quality_plateau_portfolio_qubo import (
    integerize_quality,
    stable_redundancy,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage68_implementation_and_sources_are_hash_locked():
    config = read_json("configs/stage68_quality_plateau_portfolio_qubo.json")
    for section in ("implementation", "inputs"):
        for value in config[section].values():
            assert sha256(ROOT / value["path"]) == value["sha256"]


def test_stage68_stable_redundancy_requires_all_seed_support():
    ranks = np.asarray(
        [
            [[0.0, 0.0], [0.2, 0.2], [0.8, 0.8], [1.0, 1.0]],
            [[0.0, 0.1], [0.2, 0.3], [0.8, 0.7], [1.0, 0.9]],
            [[0.0, 1.0], [0.2, 0.8], [0.8, 0.2], [1.0, 0.0]],
        ],
        dtype=float,
    )
    redundancy = stable_redundancy(ranks, np.ones(4, dtype=bool))
    assert redundancy[0, 1] == pytest.approx(0.0)
    assert np.array_equal(np.diag(redundancy), np.zeros(2))


def test_stage68_conservative_integerization_preserves_real_floor():
    utility = np.asarray([1.0, 0.98, 0.95, 0.8], dtype=float)
    floor = 0.94
    encoded = integerize_quality(utility, 2, floor, 4095)
    for subset in itertools.combinations(range(4), 2):
        deficit = int(np.sum(encoded["deficits"][list(subset)]))
        if deficit <= encoded["maximum_deficit"]:
            assert float(np.mean(utility[list(subset)])) >= floor - 1e-12


def test_stage68_dimensions_and_frozen_candidate():
    result = read_json("data/stage68_quality_plateau_portfolio_qubo_result.json")
    assert result["status"] == "stage68_quality_plateau_portfolio_qubo_complete"
    assert result["candidate_count"] == 3
    assert result["fixed_k_metric_count"] == 800
    assert result["milp_certificate_count"] == 400
    assert result["analysis_payload_sha256"] == (
        "FE620A5DB6B5F81955DB55841D86BA705D44023527CC415BD18BAFAE195C2267"
    )
    selected = result["selected_candidate"]
    assert selected["candidate_id"] == "uncertainty_0p5x"
    assert selected["uncertainty_multiplier"] == pytest.approx(0.5)
    assert selected["mean_target_gain_over_pair_off"] == pytest.approx(
        -0.0009203326323847434
    )
    assert selected["worst_target_gain_over_pair_off"] == pytest.approx(
        -0.009346393366583744
    )
    assert selected["mean_target_stable_redundancy_reduction"] == pytest.approx(
        0.029811055194393053
    )


def test_stage68_loto_and_qubo_fidelity_pass_but_hardware_stays_blocked():
    result = read_json("data/stage68_quality_plateau_portfolio_qubo_result.json")
    assert result["loto_gate"]["mean_held_target_gain_over_pair_off"] == pytest.approx(
        -0.004057227724122002
    )
    assert result["loto_gate"]["worst_held_target_gain_over_pair_off"] == pytest.approx(
        -0.009346393366583744
    )
    assert result["loto_gate"]["held_target_count_within_0p01"] == 4
    fidelity = result["qubo_fidelity"]
    assert fidelity["cell_count"] == 80
    assert fidelity["mean_subset_jaccard_vs_continuous"] == pytest.approx(1.0)
    assert fidelity["mean_holdout_bedroc_gap_vs_continuous"] == pytest.approx(0.0)
    assert fidelity["maximum_logical_variable_count"] == 105
    assert fidelity["maximum_factorized_energy_residual"] == pytest.approx(0.0)
    assert result["route_gate"]["quality_plateau_qubo_freeze_authorized"] is True
    assert result["decision"]["future_new_target_preregistration_authorized"] is True
    assert result["decision"]["robustness_claim_authorized"] is False
    assert result["decision"]["alternate_partition_probe_passed"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False


def test_stage68_exact_portfolio_exposes_greedy_local_failures():
    rows = read_csv(
        "results/runs/stage68_quality_plateau_portfolio_qubo/fixed_k_metrics.csv"
    )
    exact = {
        (
            row["target_id"],
            row["outer_fold"],
            row["candidate_id"],
            row["subset_size"],
        ): row
        for row in rows
        if row["solver_id"] == "continuous_milp_certificate"
    }
    expected_worse = {
        "same_constraint_direct_greedy": 47,
        "same_constraint_greedy_swap": 13,
    }
    for solver_id, expected_count in expected_worse.items():
        selected = [row for row in rows if row["solver_id"] == solver_id]
        gaps = []
        for row in selected:
            key = (
                row["target_id"],
                row["outer_fold"],
                row["candidate_id"],
                row["subset_size"],
            )
            gap = float(row["stable_redundancy_sum"]) - float(
                exact[key]["stable_redundancy_sum"]
            )
            assert gap >= -1e-8
            gaps.append(gap)
        assert sum(value > 1e-10 for value in gaps) == expected_count


def test_stage68_independent_audit_passed():
    audit = read_json("data/stage68_quality_plateau_portfolio_qubo_audit.json")
    assert audit["status"] == (
        "stage68_quality_plateau_portfolio_qubo_independent_audit_ok"
    )
    assert audit["fixed_k_rows_independently_checked"] == 800
    assert audit["continuous_milp_cells_independently_checked"] == 240
    assert audit["heuristic_dominance_cells_independently_checked"] == 480
    assert audit["qubo_fidelity_cells_independently_checked"] == 80
    assert audit["factorized_qubo_models_independently_checked"] == 16
    assert audit["quality_plateau_qubo_freeze_authorized"] is True
    assert audit["quantum_hardware_authorized"] is False
    assert audit["source_result"]["sha256"] == sha256(
        ROOT / "data/stage68_quality_plateau_portfolio_qubo_result.json"
    )
