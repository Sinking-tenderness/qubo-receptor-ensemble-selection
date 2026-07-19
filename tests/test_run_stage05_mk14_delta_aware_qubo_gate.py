from pathlib import Path

from scripts.run_stage05_mk14_delta_aware_qubo_gate import (
    delta_trial_key,
    gate_decision,
    load_delta_config,
)


CONFIG = Path(
    "configs/stage05_mk14_expanded_delta_aware_qubo_gate_preregistration.json"
)


def test_delta_preregistration_inherits_grid_and_locks_data() -> None:
    config = load_delta_config(CONFIG)

    assert config["candidate_space"]["inherit_original_576_candidates"] is True
    assert config["candidate_space"]["new_weights_or_subsets_added"] is False
    assert config["cross_validation"]["validation_rows_available"] == 0
    assert config["cross_validation"]["test_rows_available"] == 0


def trial(worst: float, primary: float, mean: float) -> dict[str, object]:
    return {
        "config": {
            "family": "coverage_qubo",
            "target_size": 2,
            "aggregation": "min_score",
            "weights": {
                "active_coverage": 1.0,
                "decoy_exposure": 0.0,
                "active_overlap": 1.0,
                "redundancy": 1.0,
                "stability": 1.0,
            },
        },
        "worst_seed_bedroc_delta": worst,
        "primary_bedroc_delta": primary,
        "mean_seed_bedroc_delta": mean,
        "qubo_robust_metrics": {
            "worst_seed_bedroc": 0.8,
            "primary_bedroc": 0.8,
            "primary_pr_auc": 0.8,
            "primary_roc_auc": 0.8,
        },
        "mean_seed_pairwise_jaccard": 0.8,
    }


def test_delta_selection_prioritizes_worst_seed_paired_gain() -> None:
    first = trial(0.01, 0.10, 0.10)
    second = trial(0.02, 0.01, 0.01)

    assert delta_trial_key(second) < delta_trial_key(first)


def test_delta_gate_requires_linear_single_and_random_checks() -> None:
    matrix = {
        "primary": {"bedroc_alpha_20": 0.85},
        "sensitivity": {"bedroc_alpha_20": 0.84},
        "seed0": {"bedroc_alpha_20": 0.85},
        "seed1": {"bedroc_alpha_20": 0.84},
        "seed2": {"bedroc_alpha_20": 0.83},
    }
    linear = {
        key: {"bedroc_alpha_20": value["bedroc_alpha_20"] - 0.01}
        for key, value in matrix.items()
    }
    single = {
        key: {"bedroc_alpha_20": value["bedroc_alpha_20"] - 0.02}
        for key, value in matrix.items()
    }
    metrics = {
        "delta_aware_qubo": matrix,
        "matched_linear_top_k": linear,
        "single_best": single,
    }
    acceptance = load_delta_config(CONFIG)["acceptance"]
    details = {
        "seed_pairwise_jaccard": {"mean": 0.7},
        "noncardinality_quadratic": {
            "maximum_absolute": 1.0,
            "range": 0.5,
        },
    }
    random_context = {
        "primary_bedroc_percentile": 0.75,
        "mean_seed_bedroc_percentile": 0.75,
    }

    _, checks, passed = gate_decision(
        metrics,
        ("R1", "R2"),
        ("R1", "R3"),
        details,
        0.7,
        random_context,
        acceptance,
    )

    assert passed
    assert all(checks.values())

    random_context["primary_bedroc_percentile"] = 0.25
    _, checks, passed = gate_decision(
        metrics,
        ("R1", "R2"),
        ("R1", "R3"),
        details,
        0.7,
        random_context,
        acceptance,
    )
    assert not checks["primary_exact_fixed_subset_percentile"]
    assert not passed
