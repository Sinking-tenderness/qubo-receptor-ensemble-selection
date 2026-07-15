import pytest

from scripts.normalized_receptor_qubo import (
    build_normalized_terms,
    exact_select,
    minmax_terms,
)


def test_minmax_terms_uses_zero_for_a_constant_noninformative_term():
    assert minmax_terms({"A": 2.0, "B": 2.0}) == {"A": 0.0, "B": 0.0}
    assert minmax_terms({"A": 2.0, "B": 4.0}) == {"A": 0.0, "B": 1.0}


def test_build_normalized_terms_scales_every_term_family():
    rows = [
        {"ligand_id": "A1", "label": "active", "R1": -9.0, "R2": -7.0},
        {"ligand_id": "A2", "label": "active", "R1": -8.0, "R2": -9.0},
        {"ligand_id": "D1", "label": "decoy", "R1": -5.0, "R2": -8.0},
        {"ligand_id": "D2", "label": "decoy", "R1": -4.0, "R2": -4.0},
    ]
    terms = build_normalized_terms(rows, ["R1", "R2"], 0.5, "bedroc")
    for values in terms["normalized"].values():
        assert all(0.0 <= value <= 1.0 for value in values.values())
    assert set(terms["active_ids"]) == {"R1", "R2"}
    assert set(terms["decoy_ids"]) == {"R1", "R2"}


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
