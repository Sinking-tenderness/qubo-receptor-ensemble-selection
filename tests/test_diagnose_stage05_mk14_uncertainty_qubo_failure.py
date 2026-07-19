from pathlib import Path

from scripts.diagnose_stage05_mk14_uncertainty_qubo_failure import (
    compact_candidate,
    diagnostic_sort_key,
    load_diagnostic_config,
)


CONFIG = Path(
    "configs/stage05_mk14_expanded_uncertainty_qubo_failure_diagnostic.json"
)


def test_failure_diagnostic_preserves_posthoc_boundary() -> None:
    config = load_diagnostic_config(CONFIG)

    assert config["expected"]["candidate_count"] == 576
    assert config["expected"]["validation_rows"] == 0
    assert config["expected"]["test_rows"] == 0
    assert "cannot retroactively pass" in config["interpretation_boundary"]


def test_diagnostic_sort_prioritizes_worst_seed_delta() -> None:
    first = {
        "worst_seed_bedroc_delta": 0.01,
        "primary_bedroc_delta": 0.01,
        "mean_seed_bedroc_delta": 0.01,
        "primary_qubo_bedroc": 0.8,
        "target_size": 3,
        "family": "coverage_qubo",
        "aggregation": "min_score",
        "weights": "{}",
    }
    second = dict(first)
    second["worst_seed_bedroc_delta"] = 0.02
    second["primary_qubo_bedroc"] = 0.7

    assert diagnostic_sort_key(second) < diagnostic_sort_key(first)


def test_compact_candidate_excludes_fold_level_bulk() -> None:
    row = {
        "family": "coverage_qubo",
        "target_size": 2,
        "aggregation": "mean_score",
        "weights": "{}",
        "final_subset": "R1+R2",
        "final_linear_subset": "R1+R3",
        "fold_subset_difference_count": 2,
        "primary_qubo_bedroc": 0.8,
        "primary_linear_bedroc": 0.7,
        "primary_bedroc_delta": 0.1,
        "mean_seed_bedroc_delta": 0.1,
        "worst_seed_bedroc_delta": 0.05,
        "fold_seed_fit_mean_pairwise_jaccard": 0.8,
        "final_seed_fit_mean_pairwise_jaccard": 0.7,
        "all_diagnostic_checks": True,
        "fold_subsets": "large",
    }

    compact = compact_candidate(row)

    assert compact["final_subset"] == "R1+R2"
    assert "fold_subsets" not in compact
