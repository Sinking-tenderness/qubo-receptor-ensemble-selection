import json
from pathlib import Path

from scripts.run_stage05_mk14_uncertainty_qubo_gate import (
    gate_decision,
    jaccard,
    load_config,
    matched_linear_top_k,
    pairwise_jaccard,
    qubo_candidate_configs,
    stability_scores,
)


CONFIG = Path(
    "configs/stage05_mk14_expanded_uncertainty_qubo_gate_preregistration.json"
)


def test_preregistration_keeps_validation_and_test_unavailable() -> None:
    config = load_config(CONFIG)

    assert config["expected"]["validation_rows"] == 0
    assert config["expected"]["test_rows"] == 0
    assert config["model"]["subset_sizes"] == [2, 3]
    assert config["acceptance"]["all_checks_required"] is True


def test_every_qubo_candidate_contains_a_pair_interaction() -> None:
    config = json.loads(CONFIG.read_text(encoding="ascii"))
    candidates = qubo_candidate_configs(config["model"])

    assert len(candidates) == 576
    assert all(
        candidate["weights"]["active_overlap"] > 0.0
        or candidate["weights"]["redundancy"] > 0.0
        for candidate in candidates
    )
    assert {candidate["target_size"] for candidate in candidates} == {2, 3}


def test_stability_reward_prefers_lower_seed_dispersion() -> None:
    seed_terms = {
        "seed0": {"normalized": {"utility": {"R1": 0.5, "R2": 0.0}}},
        "seed1": {"normalized": {"utility": {"R1": 0.5, "R2": 0.5}}},
        "seed2": {"normalized": {"utility": {"R1": 0.5, "R2": 1.0}}},
    }

    raw, normalized = stability_scores(seed_terms, ["R1", "R2"])

    assert raw["R1"] > raw["R2"]
    assert normalized == {"R1": 1.0, "R2": 0.0}


def test_matched_linear_top_k_uses_only_linear_order() -> None:
    coefficients = {
        "linear": {"R1": -1.0, "R2": -3.0, "R3": -2.0},
        "quadratic": {"R1__R2": 99.0, "R1__R3": -99.0, "R2__R3": 0.0},
    }

    assert matched_linear_top_k(coefficients, 2) == ("R2", "R3")


def test_pairwise_jaccard_reports_seed_subset_stability() -> None:
    subsets = [("R1", "R2"), ("R1", "R3"), ("R1", "R2")]
    summary = pairwise_jaccard(subsets)

    assert jaccard(subsets[0], subsets[1]) == 1 / 3
    assert summary["minimum"] == 1 / 3
    assert summary["maximum"] == 1.0


def test_gate_requires_quadratic_difference_and_seed_robustness() -> None:
    candidate = {
        "primary": {"bedroc_alpha_20": 0.80},
        "seed0": {"bedroc_alpha_20": 0.80},
        "seed1": {"bedroc_alpha_20": 0.79},
        "seed2": {"bedroc_alpha_20": 0.78},
    }
    linear = {
        "primary": {"bedroc_alpha_20": 0.79},
        "seed0": {"bedroc_alpha_20": 0.79},
        "seed1": {"bedroc_alpha_20": 0.78},
        "seed2": {"bedroc_alpha_20": 0.77},
    }
    acceptance = load_config(CONFIG)["acceptance"]
    _, checks, passed = gate_decision(
        candidate,
        linear,
        ("R1", "R2"),
        ("R1", "R3"),
        {"maximum_absolute": 1.0, "range": 0.5},
        0.75,
        0.75,
        acceptance,
    )

    assert passed
    assert all(checks.values())

    _, checks, passed = gate_decision(
        candidate,
        linear,
        ("R1", "R2"),
        ("R1", "R2"),
        {"maximum_absolute": 1.0, "range": 0.0},
        0.75,
        0.75,
        acceptance,
    )
    assert not checks["noncardinality_quadratic_terms"]
    assert not checks["selected_subset_differs_from_matched_linear"]
    assert not passed
