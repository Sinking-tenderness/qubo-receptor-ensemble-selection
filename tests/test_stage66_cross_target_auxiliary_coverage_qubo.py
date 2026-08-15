import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_stage66_cross_target_auxiliary_coverage_qubo import (
    CoverageObjective,
    assignment_for_subset,
    build_coverage_terms,
    build_sparse_qubo,
    qubo_energy,
)


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def read_csv(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage66_implementation_and_sources_are_hash_locked():
    config = read_json("configs/stage66_cross_target_auxiliary_coverage_qubo.json")
    for section in ("implementation", "inputs"):
        for value in config[section].values():
            assert sha256(ROOT / value["path"]) == value["sha256"]


def test_stage66_scale_schedule_is_bedroc20_derived_and_normalized():
    config = read_json("configs/stage66_cross_target_auxiliary_coverage_qubo.json")
    development = config["development"]
    assert development["coverage_fractions"] == [0.05, 0.1, 0.2]
    assert sum(development["scale_weights"]) == pytest.approx(1.0)
    assert development["scale_weights"] == pytest.approx(
        [0.6321205588285577, 0.3180923728035784, 0.049787068367863944]
    )


def test_stage66_auxiliary_qubo_matches_reduced_objective_on_toy_states():
    ranks = np.asarray(
        [
            [[0.01, 0.30, 0.20], [0.30, 0.02, 0.25], [0.05, 0.40, 0.03], [0.50, 0.04, 0.30]],
            [[0.02, 0.35, 0.25], [0.25, 0.03, 0.30], [0.04, 0.45, 0.02], [0.55, 0.05, 0.35]],
            [[0.03, 0.32, 0.22], [0.28, 0.04, 0.27], [0.06, 0.42, 0.04], [0.52, 0.06, 0.32]],
        ],
        dtype=float,
    )
    labels = np.asarray([1, 1, 0, 0], dtype=int)
    candidate = {
        "candidate_id": "toy",
        "active_seed_rule": "majority",
        "decoy_seed_rule": "any",
        "decoy_weight": 0.5,
        "singleton_weight": 0.25,
    }
    terms = build_coverage_terms(
        ranks,
        labels,
        ["a0", "a1", "d0", "d1"],
        np.ones(4, dtype=bool),
        candidate,
        [0.05, 0.1, 0.2],
        [0.6, 0.3, 0.1],
        20.0,
    )
    scorer = CoverageObjective(terms, candidate)
    receptors = ["r0", "r1", "r2"]
    qubo = build_sparse_qubo(terms, receptors, 2, candidate, 20.0, 100.0)
    for subset in ((0, 1), (0, 2), (1, 2)):
        assignment = assignment_for_subset(terms, qubo, receptors, subset)
        assert qubo_energy(qubo, assignment) + scorer.score(subset)[0] == pytest.approx(
            0.0, abs=1e-8
        )


def test_stage66_dimensions_and_pair_off_reproduction():
    result = read_json("data/stage66_cross_target_auxiliary_coverage_qubo_result.json")
    assert result["status"] == "stage66_cross_target_auxiliary_coverage_qubo_complete"
    assert result["candidate_count"] == 6
    assert result["fixed_k_metric_count"] == 1248
    assert result["pair_off_reproduction_cell_count"] == 96
    assert result["analysis_payload_sha256"] == (
        "B06B0340CCA578BDC447D46DB083F4093FED0F10E61AABDC64901402C839EFDC"
    )


def test_stage66_best_coverage_candidate_fails_cross_target_performance_gate():
    result = read_json("data/stage66_cross_target_auxiliary_coverage_qubo_result.json")
    selected = result["selected_candidate"]
    assert selected["candidate_id"] == "ms_all_any_d0p25_s0p5"
    assert selected["mean_target_gain_over_pair_off"] == pytest.approx(
        -0.039041279869826015
    )
    assert selected["worst_target_gain_over_pair_off"] == pytest.approx(
        -0.09932325613912128
    )
    assert selected["nonnegative_target_count_over_pair_off"] == 0
    assert result["freeze_gate"]["coverage_objective_freeze_authorized"] is False
    assert result["decision"]["new_target_preregistration_authorized"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False


def test_stage66_search_is_stable_but_cannot_rescue_the_objective():
    result = read_json("data/stage66_cross_target_auxiliary_coverage_qubo_result.json")
    solver = result["solver_audit"]
    assert solver["same_objective_cell_count"] == 576
    assert solver["beam_swap_noninferior_objective_cell_count"] == 576
    assert solver["selected_candidate_selection_difference_fraction_vs_greedy"] == pytest.approx(
        0.15
    )
    assert solver["maximum_qubo_energy_equivalence_residual"] < 1e-7
    rows = {
        row["target_id"]: row
        for row in read_csv(
            "results/runs/stage66_cross_target_auxiliary_coverage_qubo/target_summary.csv"
        )
        if row["candidate_id"] == "ms_all_any_d0p25_s0p5"
    }
    assert float(rows["BACE1"]["mean_gain_over_pair_off"]) == pytest.approx(
        -0.010432099052716198
    )
    assert float(rows["PPARG"]["mean_gain_over_pair_off"]) == pytest.approx(
        -0.09932325613912128
    )
    assert float(rows["PPARA"]["mean_gain_over_pair_off"]) == pytest.approx(
        -0.01164603238169135
    )
    assert float(rows["PPARD"]["mean_gain_over_pair_off"]) == pytest.approx(
        -0.03476373190577523
    )


def test_stage66_independent_audit_passed():
    audit = read_json("data/stage66_cross_target_auxiliary_coverage_qubo_audit.json")
    assert audit["status"] == (
        "stage66_cross_target_auxiliary_coverage_qubo_independent_audit_ok"
    )
    assert audit["pair_off_reproduction_cells_independently_verified"] == 96
    assert audit["same_objective_search_cells_independently_verified"] == 576
    assert audit["qubo_models_independently_energy_checked"] == 4
    assert audit["coverage_objective_freeze_authorized"] is False
    assert audit["source_result"]["sha256"] == sha256(
        ROOT / "data/stage66_cross_target_auxiliary_coverage_qubo_result.json"
    )
