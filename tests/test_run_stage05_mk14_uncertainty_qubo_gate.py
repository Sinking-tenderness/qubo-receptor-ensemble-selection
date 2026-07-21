import json
from pathlib import Path

from scripts.run_stage05_mk14_uncertainty_qubo_gate import (
    gate_decision,
    jaccard,
    load_config,
    matched_linear_top_k,
    pair_utility_terms_for_aggregation,
    pairwise_jaccard,
    qubo_candidate_configs,
    stability_scores,
)


CONFIG = Path(
    "configs/stage05_mk14_expanded_uncertainty_qubo_gate_preregistration.json"
)
PAIR_CONFIG = Path(
    "configs/stage05_mk14_train696_pair_utility_qubo_gate_preregistration.json"
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


def test_pair_utility_grid_is_small_and_aggregation_aligned() -> None:
    model = dict(load_config(CONFIG)["model"])
    model["qubo_families"] = ["pair_utility_qubo"]
    model["weight_grids"] = {
        **model["weight_grids"],
        "ensemble_pair_utility": [0.5, 1.0, 2.0],
    }
    candidates = qubo_candidate_configs(model)

    assert len(candidates) == 24
    assert all(
        candidate["weights"]["ensemble_pair_utility"] > 0.0
        and candidate["weights"]["active_overlap"] == 0.0
        and candidate["weights"]["redundancy"] == 0.0
        for candidate in candidates
    )

    terms = {
        "raw": {
            "pair_ensemble_utility": {"R1__R2": -1.0},
            "pair_ensemble_utility_min_score": {"R1__R2": 0.2},
            "pair_ensemble_utility_mean_score": {"R1__R2": 0.8},
        },
        "normalized": {
            "pair_ensemble_utility": {"R1__R2": -1.0},
            "pair_ensemble_utility_min_score": {"R1__R2": 0.3},
            "pair_ensemble_utility_mean_score": {"R1__R2": 0.9},
        },
    }
    selected = pair_utility_terms_for_aggregation(terms, "min_score")
    assert selected["normalized"]["pair_ensemble_utility"] == {
        "R1__R2": 0.3
    }
    assert terms["normalized"]["pair_ensemble_utility"] == {
        "R1__R2": -1.0
    }


def test_pair_utility_preregistration_freezes_dual_baseline_gate() -> None:
    config = load_config(PAIR_CONFIG)
    candidates = qubo_candidate_configs(config["model"])

    assert len(candidates) == config["model"]["candidate_count"] == 24
    assert config["expected"]["validation_rows"] == 0
    assert config["expected"]["test_rows"] == 0
    assert (
        config["acceptance"][
            "minimum_primary_median_bedroc_delta_vs_greedy"
        ]
        == 0.0
    )


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


def test_gate_can_require_noninferiority_to_greedy() -> None:
    candidate = {
        matrix: {"bedroc_alpha_20": value}
        for matrix, value in (
            ("primary", 0.80),
            ("seed0", 0.80),
            ("seed1", 0.80),
            ("seed2", 0.80),
        )
    }
    linear = {
        matrix: {"bedroc_alpha_20": 0.79} for matrix in candidate
    }
    greedy = {
        matrix: {"bedroc_alpha_20": 0.81} for matrix in candidate
    }
    acceptance = {
        **load_config(CONFIG)["acceptance"],
        "minimum_primary_median_bedroc_delta_vs_greedy": 0.0,
        "minimum_mean_seed_bedroc_delta_vs_greedy": 0.0,
        "minimum_worst_seed_bedroc_delta_vs_greedy": 0.0,
    }
    _, checks, passed = gate_decision(
        candidate,
        linear,
        ("R1", "R2"),
        ("R1", "R3"),
        {"maximum_absolute": 1.0, "range": 0.5},
        1.0,
        1.0,
        acceptance,
        greedy,
    )

    assert not checks["primary_median_bedroc_delta_vs_greedy"]
    assert not checks["mean_seed_bedroc_delta_vs_greedy"]
    assert not checks["worst_seed_bedroc_delta_vs_greedy"]
    assert not passed
