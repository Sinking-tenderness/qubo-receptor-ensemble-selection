import json
from pathlib import Path

import pytest

from scripts.run_repeated_development_scaffold_cv import (
    derive_gate_config,
    file_sha256,
    load_protocol,
    summary_statistics,
)


PROTOCOL_PATH = Path(
    "configs/stage04_cdk2_core_plus_one_repeated_scaffold_cv.json"
)


def test_repeated_protocol_preregisters_five_seeds_and_locked_base_config():
    protocol = load_protocol(PROTOCOL_PATH)
    base_path = Path(protocol["base_config"])
    assert len(protocol["fold_seeds"]) == 5
    assert len(set(protocol["fold_seeds"])) == 5
    assert file_sha256(base_path) == protocol["base_config_sha256"]
    base = json.loads(base_path.read_text(encoding="ascii"))
    assert base["cross_validation"]["evaluate_locked_test"] is False


def test_derived_repeat_changes_only_seed_identity_and_output_root():
    protocol = load_protocol(PROTOCOL_PATH)
    base_path = Path(protocol["base_config"])
    base = json.loads(base_path.read_text(encoding="ascii"))
    run_directory = Path("results/runs/example_repeat/fold_seed_17")
    derived = derive_gate_config(
        base,
        17,
        run_directory,
        base_path,
        file_sha256(base_path),
    )
    assert derived["cross_validation"]["fold_seed"] == 17
    assert derived["cross_validation"]["evaluate_locked_test"] is False
    assert derived["model"] == base["model"]
    assert derived["outputs"]["run_directory"] == run_directory.as_posix()
    assert all(
        str(value).startswith(run_directory.as_posix())
        for value in derived["outputs"].values()
    )


def test_summary_statistics_reports_direction_and_spread():
    result = summary_statistics([-0.1, 0.2, 0.3])
    assert result["count"] == 3
    assert result["positive_count"] == 2
    assert result["mean"] == pytest.approx(0.4 / 3.0)
    assert result["minimum"] == -0.1
    assert result["maximum"] == 0.3
