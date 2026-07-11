from scripts.solve_coverage_qubo import (
    coefficient_energy,
    combined_coefficients,
    coverage_objective,
)


def test_coverage_overlap_penalizes_duplicate_active_coverage() -> None:
    base = {
        "target_size": 2,
        "weights": {"count": 0.0, "size": 0.0, "redundancy": 0.0},
        "utilities_train_roc_auc": {"r1": 0.0, "r2": 0.0},
        "redundancy_train_spearman_clipped": {"r1__r2": 0.0},
    }
    rewards = {"r1": 0.5, "r2": 0.5}
    overlaps = {"r1__r2": 0.5}

    duplicate = coverage_objective(("r1", "r2"), base, rewards, overlaps, 1.0, 1.0)
    complementary = coverage_objective(("r1", "r2"), base, rewards, {"r1__r2": 0.0}, 1.0, 1.0)

    assert complementary < duplicate


def test_explicit_coefficients_match_objective() -> None:
    base = {
        "target_size": 2,
        "weights": {"count": 0.0, "size": 1.0, "redundancy": 0.0},
        "utilities_train_roc_auc": {"r1": 0.4, "r2": 0.6},
        "redundancy_train_spearman_clipped": {"r1__r2": 0.0},
        "linear_coefficients": {"r1": -3.4, "r2": -3.6},
        "quadratic_coefficients": {"r1__r2": 2.0},
    }
    rewards = {"r1": 0.5, "r2": 0.25}
    overlaps = {"r1__r2": 0.25}
    coefficients = combined_coefficients(base, rewards, overlaps, 0.5, 0.5)

    direct = coverage_objective(("r1", "r2"), base, rewards, overlaps, 0.5, 0.5)
    assert coefficient_energy(("r1", "r2"), coefficients) == direct
