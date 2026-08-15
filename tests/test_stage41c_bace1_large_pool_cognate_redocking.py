from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experimental.unidock import (
    run_stage41c_bace1_large_pool_cognate_redocking as stage41c,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage41c_bace1_large_pool_cognate_redocking.json"


def test_stage41c_frozen_input_audit() -> None:
    config = json.loads(CONFIG.read_text(encoding="ascii"))
    receptors, cases, audit = stage41c.validate_inputs(ROOT, config)

    assert len(receptors) == 48
    assert len(cases) == 48
    assert audit["frozen_receptor_count"] == 49
    assert audit["expected_redocking_pair_count"] == 144
    assert audit["technical_preparation_failure_count"] == 1
    assert audit["technical_preparation_failures"] == [
        {
            "conformer_id": "BACE1_6DMI_aligned",
            "case_id": "6DMI_5T5",
            "error": "ValueError: BACE1 ModelServer ligand coordinates differ: 6DMI_5T5",
        }
    ]
    assert all(int(audit[key]) == 0 for key in (
        "ligand_labels_read",
        "benchmark_docking_scores_read",
        "fresh_validation_rows_read",
        "test_rows_read",
    ))


def test_stage41c_rejects_changed_failure_ledger() -> None:
    config = json.loads(CONFIG.read_text(encoding="ascii"))
    config["expected"]["preparation_failures"][0]["conformer_id"] = "BACE1_CHANGED"

    with pytest.raises(ValueError, match="preparation-failure ledger differs"):
        stage41c.validate_inputs(ROOT, config)


def test_stage41c_gate_summary_requires_all_three_seeds() -> None:
    rows = [
        {
            "conformer_id": "BACE1_TEST",
            "top_ranked_rmsd_angstrom": value,
            "top_ranked_pose_success": value <= 2.0,
        }
        for value in (1.0, 1.5, 3.0)
    ]
    summary = stage41c.summarize_gate(rows, ["BACE1_TEST"], 2.0, 2)

    assert summary[0]["successful_seed_count"] == 2
    assert summary[0]["median_top_ranked_rmsd_angstrom"] == 1.5
    assert summary[0]["gate_pass"] is True

    with pytest.raises(ValueError, match="result count differs"):
        stage41c.summarize_gate(rows[:2], ["BACE1_TEST"], 2.0, 2)
