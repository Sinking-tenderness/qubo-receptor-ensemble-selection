import json
from pathlib import Path

import pytest

from scripts.evaluate_stage05_mk14_greedy_failure_screen import (
    fixed_cardinality_exact,
    fixed_cardinality_greedy,
    metric_exact,
    metric_greedy,
    strict_metric_failure,
    trial_summary,
)
from scripts.prepare_receptor import file_sha256


def coefficients(linear, quadratic):
    return {
        "constant": 0.0,
        "linear": linear,
        "quadratic": quadratic,
    }


def test_quadratic_complementarity_traps_forward_greedy():
    model = coefficients(
        {"A": -10.0, "B": -6.0, "C": -6.0},
        {"A__B": 0.0, "A__C": 0.0, "B__C": -10.0},
    )

    exact, exact_energy = fixed_cardinality_exact(model, ("A", "B", "C"), 2)
    greedy, greedy_energy, path = fixed_cardinality_greedy(
        model, ("A", "B", "C"), 2
    )

    assert exact == ("B", "C")
    assert greedy == ("A", "B")
    assert greedy_energy - exact_energy == pytest.approx(6.0)
    assert path[0]["subset"] == ["A"]


def test_modular_objective_gives_greedy_exact_match():
    model = coefficients(
        {"A": -3.0, "B": -2.0, "C": -1.0},
        {"A__B": 0.0, "A__C": 0.0, "B__C": 0.0},
    )

    exact, exact_energy = fixed_cardinality_exact(model, ("A", "B", "C"), 2)
    greedy, greedy_energy, _ = fixed_cardinality_greedy(
        model, ("A", "B", "C"), 2
    )

    assert exact == greedy == ("A", "B")
    assert exact_energy == greedy_energy


def test_metric_screen_detects_non_additive_local_optimum():
    quality = {
        ("A",): 10.0,
        ("B",): 6.0,
        ("C",): 6.0,
        ("A", "B"): 16.0,
        ("A", "C"): 16.0,
        ("B", "C"): 22.0,
    }

    def evaluate(subset):
        value = quality[tuple(sorted(subset))]
        return {
            "worst_seed_bedroc": value,
            "primary_bedroc": value,
            "mean_seed_bedroc": value,
            "primary_pr_auc": value,
            "primary_roc_auc": value,
        }

    exact, exact_metrics = metric_exact(("A", "B", "C"), 2, evaluate)
    greedy, greedy_metrics = metric_greedy(("A", "B", "C"), 2, evaluate)

    assert exact == ("B", "C")
    assert greedy == ("A", "B")
    assert strict_metric_failure(exact_metrics, greedy_metrics)


def test_trial_summary_counts_only_strict_failures():
    rows = [
        {
            "objective_family": "pair_synergy_qubo",
            "strict_failure": True,
            "objective_regret": 0.5,
            "holdout_primary_bedroc_delta": 0.1,
        },
        {
            "objective_family": "pair_synergy_qubo",
            "strict_failure": False,
            "objective_regret": 0.0,
            "holdout_primary_bedroc_delta": None,
        },
    ]

    summary = trial_summary(rows, ("objective_family",))[0]

    assert summary["trial_count"] == 2
    assert summary["strict_failure_count"] == 1
    assert summary["strict_failure_rate"] == pytest.approx(0.5)
    assert summary["heldout_primary_bedroc_exact_better_fraction"] == 1.0


def test_frozen_config_hash_matches_implementation():
    config = json.loads(
        Path(
            "configs/stage05_mk14_greedy_failure_screen_posthoc.json"
        ).read_text(encoding="ascii")
    )
    implementation = Path(config["implementation"]["path"])

    assert file_sha256(implementation) == config["implementation"]["sha256"]
