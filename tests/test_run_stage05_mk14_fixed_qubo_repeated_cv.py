from pathlib import Path

from scripts.run_stage05_mk14_fixed_qubo_repeated_cv import (
    gate_decision,
    load_config,
    nearest_rank,
)


CONFIG = Path(
    "configs/stage05_mk14_expanded_fixed_qubo_repeated_cv_preregistration.json"
)


def test_fixed_candidate_and_repeat_seeds_are_frozen() -> None:
    config = load_config(CONFIG)

    assert config["fixed_candidate"]["new_candidate_search_permitted"] is False
    assert config["fixed_candidate"]["target_size"] == 3
    assert len(config["repeated_cross_validation"]["fold_seeds"]) == 20
    assert config["repeated_cross_validation"]["validation_rows_available"] == 0
    assert config["repeated_cross_validation"]["test_rows_available"] == 0


def test_nearest_rank_quartile_is_deterministic() -> None:
    assert nearest_rank(list(range(1, 21)), 0.25) == 5.0
    assert nearest_rank([3.0], 0.25) == 3.0


def test_repeated_gate_requires_repeat_fraction_and_full_difference() -> None:
    repeat_rows = []
    for index in range(20):
        repeat_rows.append(
            {
                "primary_vs_linear": 0.02,
                "mean_seed_vs_linear": 0.02,
                "worst_seed_vs_linear": 0.01,
                "primary_vs_single_best": 0.01,
                "worst_seed_vs_single_best": 0.005,
                "seed0_vs_linear": 0.01,
                "seed1_vs_linear": 0.01,
                "seed2_vs_linear": 0.01,
                "fold_seed_fit_mean_pairwise_jaccard": 0.75,
                "primary_fixed_subset_percentile": 0.75,
                "mean_seed_fixed_subset_percentile": 0.75,
            }
        )
    details = {
        "seed_pairwise_jaccard": {"mean": 0.75},
        "noncardinality_quadratic": {
            "maximum_absolute": 1.0,
            "range": 0.5,
        },
    }
    acceptance = load_config(CONFIG)["acceptance"]

    _, checks, passed = gate_decision(
        repeat_rows,
        ("R1", "R2", "R3"),
        ("R1", "R2", "R4"),
        details,
        acceptance,
    )

    assert passed
    assert all(checks.values())

    _, checks, passed = gate_decision(
        repeat_rows,
        ("R1", "R2", "R3"),
        ("R1", "R2", "R3"),
        details,
        acceptance,
    )
    assert not checks["full_train_subset_differs_from_matched_linear"]
    assert not passed
