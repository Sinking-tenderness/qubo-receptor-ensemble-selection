from pathlib import Path

import pytest

from scripts.select_reliable_receptor_protocol import (
    file_sha256,
    load_config,
    mean_delta,
)


CONFIG_PATH = Path("configs/stage04_cdk2_reliable_protocol_selector.json")


def test_reliability_selector_is_fail_closed_and_keeps_test_external():
    config = load_config(CONFIG_PATH)
    assert config["fallback_protocol"]["method"] == "single_best"
    assert config["criteria"]["minimum_feasibility_fraction"] == 1.0
    assert config["criteria"]["minimum_positive_primary_bedroc_fraction"] == 0.8
    assert config["criteria"]["minimum_mean_primary_bedroc_delta"] == 0.02
    assert "locked test" in config["interpretation_boundary"]


def test_reliability_candidate_summaries_are_hash_pinned():
    config = load_config(CONFIG_PATH)
    for candidate in config["candidates"]:
        path = Path(candidate["summary_path"])
        assert file_sha256(path) == candidate["summary_sha256"]


def test_mean_delta_pairs_methods_within_each_repeat_seed():
    rows = [
        {
            "fold_seed": "1",
            "matrix": "primary",
            "method": "single_best",
            "bedroc_alpha_20": "0.7",
        },
        {
            "fold_seed": "1",
            "matrix": "primary",
            "method": "coverage_qubo",
            "bedroc_alpha_20": "0.8",
        },
        {
            "fold_seed": "2",
            "matrix": "primary",
            "method": "single_best",
            "bedroc_alpha_20": "0.9",
        },
        {
            "fold_seed": "2",
            "matrix": "primary",
            "method": "coverage_qubo",
            "bedroc_alpha_20": "0.85",
        },
    ]
    values, seeds = mean_delta(
        rows,
        "coverage_qubo",
        "single_best",
        "primary",
        "bedroc_alpha_20",
    )
    assert seeds == [1, 2]
    assert values == pytest.approx([0.1, -0.05])
