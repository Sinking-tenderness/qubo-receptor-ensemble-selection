import pytest

from scripts.normalized_receptor_qubo import (
    build_normalized_terms,
    exact_select,
    maxabs_terms,
    minmax_terms,
)


def test_minmax_terms_uses_zero_for_a_constant_noninformative_term():
    assert minmax_terms({"A": 2.0, "B": 2.0}) == {"A": 0.0, "B": 0.0}
    assert minmax_terms({"A": 2.0, "B": 4.0}) == {"A": 0.0, "B": 1.0}


def test_maxabs_terms_preserves_signed_evidence():
    assert maxabs_terms({"A": -2.0, "B": 1.0, "C": 0.0}) == {
        "A": -1.0,
        "B": 0.5,
        "C": 0.0,
    }
    assert maxabs_terms({"A": 0.0, "B": 0.0}) == {"A": 0.0, "B": 0.0}


def test_build_normalized_terms_scales_every_term_family():
    rows = [
        {"ligand_id": "A1", "label": "active", "R1": -9.0, "R2": -7.0},
        {"ligand_id": "A2", "label": "active", "R1": -8.0, "R2": -9.0},
        {"ligand_id": "D1", "label": "decoy", "R1": -5.0, "R2": -8.0},
        {"ligand_id": "D2", "label": "decoy", "R1": -4.0, "R2": -4.0},
    ]
    terms = build_normalized_terms(rows, ["R1", "R2"], 0.5, "bedroc")
    for name, values in terms["normalized"].items():
        lower = -1.0 if name.startswith("pair_ensemble_synergy") else 0.0
        assert all(lower <= value <= 1.0 for value in values.values())
    assert set(terms["normalized"]["pair_ensemble_utility"]) == {
        "R1__R2"
    }
    assert set(terms["normalized"]["pair_ensemble_utility_min_score"]) == {
        "R1__R2"
    }
    assert set(terms["normalized"]["pair_ensemble_utility_mean_score"]) == {
        "R1__R2"
    }
    assert set(terms["normalized"]["pair_ensemble_synergy_min_score"]) == {
        "R1__R2"
    }
    assert set(terms["normalized"]["pair_ensemble_synergy_mean_score"]) == {
        "R1__R2"
    }
    assert set(terms["active_ids"]) == {"R1", "R2"}
    assert set(terms["decoy_ids"]) == {"R1", "R2"}

    raw = terms["raw"]
    stronger_singleton = max(
        raw["singleton_bedroc"]["R1"],
        raw["singleton_bedroc"]["R2"],
    )
    assert raw["pair_ensemble_synergy_min_score"]["R1__R2"] == (
        pytest.approx(
            raw["pair_ensemble_utility_min_score"]["R1__R2"]
            - stronger_singleton
        )
    )
    assert raw["pair_ensemble_synergy_mean_score"]["R1__R2"] == (
        pytest.approx(
            raw["pair_ensemble_utility_mean_score"]["R1__R2"]
            - stronger_singleton
        )
    )


def test_decoy_exposure_penalty_changes_size_one_exact_solution():
    terms = {
        "normalized": {
            "utility": {"R1": 1.0, "R2": 1.0},
            "active_coverage": {"R1": 0.0, "R2": 0.0},
            "decoy_exposure": {"R1": 1.0, "R2": 0.0},
            "active_overlap": {"R1__R2": 0.0},
            "redundancy": {"R1__R2": 0.0},
        }
    }
    base_weights = {
        "active_coverage": 0.0,
        "decoy_exposure": 0.0,
        "active_overlap": 0.0,
        "redundancy": 0.0,
    }
    subset, _, _ = exact_select(terms, ["R1", "R2"], 1, base_weights, 10.0)
    assert subset == ("R1",)

    discriminative = {**base_weights, "decoy_exposure": 1.0}
    subset, energy, coefficients = exact_select(
        terms, ["R1", "R2"], 1, discriminative, 10.0
    )
    assert subset == ("R2",)
    assert energy == pytest.approx(-1.0)
    assert coefficients["target_size"] == 1
    assert coefficients["exact_search"]["states_evaluated"] == 2


def test_exact_select_falls_back_and_rejects_an_insufficient_size_penalty():
    terms = {
        "normalized": {
            "utility": {"R1": 0.0, "R2": 0.0},
            "active_coverage": {"R1": 0.0, "R2": 0.0},
            "decoy_exposure": {"R1": 0.0, "R2": 0.0},
            "active_overlap": {"R1__R2": 1.0},
            "redundancy": {"R1__R2": 0.0},
        }
    }
    weights = {
        "active_coverage": 0.0,
        "decoy_exposure": 0.0,
        "active_overlap": 2.0,
        "redundancy": 0.0,
    }
    with pytest.raises(ValueError, match="size penalty failed"):
        exact_select(terms, ["R1", "R2"], 2, weights, 0.1)


def test_pair_ensemble_utility_can_reward_a_complementary_pair():
    terms = {
        "normalized": {
            "utility": {"R1": 0.0, "R2": 0.0, "R3": 0.0},
            "active_coverage": {"R1": 0.0, "R2": 0.0, "R3": 0.0},
            "decoy_exposure": {"R1": 0.0, "R2": 0.0, "R3": 0.0},
            "active_overlap": {
                "R1__R2": 0.0,
                "R1__R3": 0.0,
                "R2__R3": 0.0,
            },
            "redundancy": {
                "R1__R2": 0.0,
                "R1__R3": 0.0,
                "R2__R3": 0.0,
            },
            "pair_ensemble_utility": {
                "R1__R2": 1.0,
                "R1__R3": 0.0,
                "R2__R3": 0.0,
            },
        }
    }
    weights = {
        "active_coverage": 0.0,
        "decoy_exposure": 0.0,
        "active_overlap": 0.0,
        "redundancy": 0.0,
        "ensemble_pair_utility": 1.0,
    }
    subset, energy, _ = exact_select(terms, ["R1", "R2", "R3"], 2, weights, 10.0)

    assert subset == ("R1", "R2")
    assert energy == pytest.approx(-1.0)


def test_pair_ensemble_synergy_rewards_gain_and_penalizes_loss():
    terms = {
        "normalized": {
            "utility": {"R1": 0.0, "R2": 0.0, "R3": 0.0},
            "active_coverage": {"R1": 0.0, "R2": 0.0, "R3": 0.0},
            "decoy_exposure": {"R1": 0.0, "R2": 0.0, "R3": 0.0},
            "active_overlap": {
                "R1__R2": 0.0,
                "R1__R3": 0.0,
                "R2__R3": 0.0,
            },
            "redundancy": {
                "R1__R2": 0.0,
                "R1__R3": 0.0,
                "R2__R3": 0.0,
            },
            "pair_ensemble_synergy": {
                "R1__R2": 1.0,
                "R1__R3": -1.0,
                "R2__R3": 0.0,
            },
        }
    }
    weights = {
        "active_coverage": 0.0,
        "decoy_exposure": 0.0,
        "active_overlap": 0.0,
        "redundancy": 0.0,
        "ensemble_pair_synergy": 1.0,
    }
    subset, energy, coefficients = exact_select(
        terms, ["R1", "R2", "R3"], 2, weights, 10.0
    )

    assert subset == ("R1", "R2")
    assert energy == pytest.approx(-1.0)
    assert coefficients["quadratic"]["R1__R3"] == pytest.approx(21.0)


def test_stability_term_can_reward_a_stable_receptor():
    terms = {
        "normalized": {
            "utility": {"R1": 0.0, "R2": 0.0},
            "active_coverage": {"R1": 0.0, "R2": 0.0},
            "decoy_exposure": {"R1": 0.0, "R2": 0.0},
            "active_overlap": {"R1__R2": 0.0},
            "redundancy": {"R1__R2": 0.0},
            "stability": {"R1": 1.0, "R2": 0.0},
        }
    }
    weights = {
        "active_coverage": 0.0,
        "decoy_exposure": 0.0,
        "active_overlap": 0.0,
        "redundancy": 0.0,
        "stability": 1.0,
    }
    subset, energy, _ = exact_select(terms, ["R1", "R2"], 1, weights, 10.0)

    assert subset == ("R1",)
    assert energy == pytest.approx(-1.0)


def test_required_receptors_are_hard_constrained_during_exact_selection():
    terms = {
        "normalized": {
            "utility": {"R1": 0.0, "R2": 1.0, "R3": 0.0},
            "active_coverage": {"R1": 0.0, "R2": 0.0, "R3": 0.0},
            "decoy_exposure": {"R1": 0.0, "R2": 0.0, "R3": 0.0},
            "active_overlap": {
                "R1__R2": 0.0,
                "R1__R3": 0.0,
                "R2__R3": 0.0,
            },
            "redundancy": {
                "R1__R2": 0.0,
                "R1__R3": 0.0,
                "R2__R3": 0.0,
            },
        }
    }
    weights = {
        "active_coverage": 0.0,
        "decoy_exposure": 0.0,
        "active_overlap": 0.0,
        "redundancy": 0.0,
    }
    subset, _, coefficients = exact_select(
        terms,
        ["R1", "R2", "R3"],
        2,
        weights,
        10.0,
        ("R1",),
    )
    assert subset == ("R1", "R2")
    assert coefficients["exact_search"]["method"] == (
        "required_receptor_constrained_cardinality_enumeration"
    )
    assert coefficients["exact_search"]["states_evaluated"] == 2


def test_required_receptors_validate_budget_and_pool():
    terms = {
        "normalized": {
            "utility": {"R1": 0.0, "R2": 0.0},
            "active_coverage": {"R1": 0.0, "R2": 0.0},
            "decoy_exposure": {"R1": 0.0, "R2": 0.0},
            "active_overlap": {"R1__R2": 0.0},
            "redundancy": {"R1__R2": 0.0},
        }
    }
    weights = {
        "active_coverage": 0.0,
        "decoy_exposure": 0.0,
        "active_overlap": 0.0,
        "redundancy": 0.0,
    }
    with pytest.raises(ValueError, match="absent"):
        exact_select(terms, ["R1", "R2"], 1, weights, 10.0, ("R9",))
    with pytest.raises(ValueError, match="exceeds"):
        exact_select(terms, ["R1", "R2"], 1, weights, 10.0, ("R1", "R2"))
