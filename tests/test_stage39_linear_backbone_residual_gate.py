from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.run_stage39_linear_backbone_residual_gate import conservative_delta, subgroup_counts


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="ascii"))


def test_conservative_delta_penalizes_unstable_block_gains() -> None:
    stable = conservative_delta([0.02, 0.02, 0.02], 0.5)
    unstable = conservative_delta([0.05, 0.01, -0.02], 0.5)
    assert stable == 0.02
    assert unstable < stable


def test_subgroup_counts_separate_active_rescue_from_decoy_promotion() -> None:
    ranks = np.asarray(
        [
            [[0.2, 0.05], [0.05, 0.2], [0.2, 0.05], [0.05, 0.2]],
            [[0.2, 0.05], [0.05, 0.2], [0.2, 0.05], [0.05, 0.2]],
            [[0.2, 0.05], [0.05, 0.2], [0.2, 0.05], [0.05, 0.2]],
        ],
        dtype=float,
    )
    counts = subgroup_counts(ranks, np.asarray([1, 1, 0, 0]), (0,), (1,), 0.1)
    assert counts == {"active_rescued": 1, "active_lost": 1, "decoy_newly_promoted": 1, "decoy_removed": 1}


def test_stage39_gate_rejects_unreliable_corrections() -> None:
    result = load("stage39_linear_backbone_residual_gate_result.json")
    assert result["status"] == "stage39_linear_backbone_residual_gate_complete"
    assert result["summary"]["correction_cell_count"] == 5
    assert result["summary"]["positive_corrected_holdout_cell_count"] == 2
    assert result["summary"]["negative_corrected_holdout_cell_count"] == 3
    assert result["decision"]["gated_residual_correction_supported"] is False
    assert result["decision"]["stage40_trust_region_qubo_authorized"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False


def test_stage39_audit_passes() -> None:
    audit = load("stage39_linear_backbone_residual_gate_audit.json")
    assert audit["status"] == "stage39_linear_backbone_residual_gate_audit_ok"
    assert all(audit["checks"].values())
    assert all(audit["cell_checks"].values())
    assert audit["maximum_absolute_recalculation_difference"] <= 1e-12
