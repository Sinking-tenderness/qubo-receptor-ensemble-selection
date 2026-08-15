import json
from collections import Counter
from pathlib import Path

from scripts.experimental.unidock.build_stage58a_ppard_pilot96_input_bundle import (
    bundle_paths,
)
from scripts.experimental.unidock.prepare_stage58a_ppard_pilot96_inputs import (
    validate_source,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage58a_ppard_pilot96_unidock_input_preparation.json"


def test_stage58a_source_is_the_frozen_balanced_pilot96():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows, _, labels = validate_source(ROOT, config)
    assert len(rows) == 96
    assert labels == Counter({"active": 48, "decoy": 48})
    assert {row["pilot_selected"] for row in rows} == {"True"}
    assert {row["pilot_role"] for row in rows} == {"development_train_pilot"}
    assert Counter((row["pilot_outer_fold"], row["label"]) for row in rows) == Counter(
        {(str(fold), label): 12 for fold in range(4) for label in ("active", "decoy")}
    )


def test_stage58a_is_authorized_by_the_frozen_stage57_gate():
    summary = json.loads(
        (ROOT / "data/stage57_ppard_cognate_redocking_summary.json").read_text()
    )
    assert summary["status"] == "stage57_ppard_cognate_redocking_gate_ok"
    assert summary["passed_receptor_count"] == 29
    assert summary["minimum_passing_receptor_count"] == 24
    assert summary["pose_integrity_failure_count"] == 0
    assert summary["unresolved_warning_event_count"] == 0


def test_stage58a_bundle_excludes_protected_splits():
    paths = bundle_paths(ROOT)
    assert "data/processed/stage56_ppard_pilot96_ligand_manifest.csv" in paths
    assert "data/stage57_ppard_cognate_redocking_summary.json" in paths
    assert not any(
        marker in path.lower()
        for path in paths
        for marker in ("fresh_validation", "locked_test", "protected/")
    )


def test_stage58a_config_freezes_only_input_preparation():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["expected"] == {
        "ligand_count": 96,
        "label_counts": {"active": 48, "decoy": 48},
        "future_receptor_count": 29,
        "future_seed_count": 3,
        "future_pair_count": 8352,
        "fresh_validation_rows": 0,
        "locked_test_rows": 0,
    }
    assert config["source"]["required_row_values"] == {
        "pilot_selected": "True",
        "pilot_role": "development_train_pilot",
    }
