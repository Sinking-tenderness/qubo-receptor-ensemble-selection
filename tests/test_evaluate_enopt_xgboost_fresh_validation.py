import json
from pathlib import Path

import pytest

from scripts.evaluate_enopt_xgboost_fresh_validation import (
    robust_bedroc_delta,
)
from scripts.prepare_receptor import file_sha256


CONFIG = Path(
    "configs/stage05_mk14_enopt_xgboost_fresh_validation_preregistration.json"
)


def test_preregistration_pins_the_evaluator_implementation():
    config = json.loads(CONFIG.read_text(encoding="ascii"))
    implementation = config["implementation"]
    assert file_sha256(Path(implementation["path"])) == implementation["sha256"]
    assert config["primary_gate"]["must_run_first"] is True
    assert config["primary_gate"]["acceptance_result_must_remain_unchanged"] is True


def test_robust_bedroc_delta_has_explicit_left_minus_right_direction():
    left = {
        "primary": 0.9,
        "sensitivity": 0.8,
        "mean_seed": 0.7,
        "worst_seed": 0.6,
    }
    right = {
        "primary": 0.4,
        "sensitivity": 0.3,
        "mean_seed": 0.2,
        "worst_seed": 0.1,
    }

    assert robust_bedroc_delta(left, right) == pytest.approx({
        "primary": 0.5,
        "sensitivity": 0.5,
        "mean_seed": 0.5,
        "worst_seed": 0.5,
    })
