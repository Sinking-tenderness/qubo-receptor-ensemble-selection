from pathlib import Path

import pytest

from scripts.audit_stage05_mk14_method_gate import (
    assert_close,
    checked_paths,
    load_audit_config,
)


CONFIG_PATH = Path(
    "configs/stage05_mk14_development_method_gate_audit.json"
)


def test_audit_config_locks_failed_gate_evidence():
    config = load_audit_config(CONFIG_PATH)
    paths = checked_paths(config)

    assert set(paths) == {
        "execution_config",
        "summary",
        "candidate_protocol",
        "validation_metrics",
        "validation_scores",
        "exact_random_subsets",
        "exact_random_summary",
    }
    assert config["numeric_tolerance"] == 1e-12


def test_numeric_audit_is_fail_closed():
    assert_close(0.5, 0.5 + 1e-13, 1e-12, "close")
    with pytest.raises(ValueError, match="numeric audit differs"):
        assert_close(0.5, 0.6, 1e-12, "different")
