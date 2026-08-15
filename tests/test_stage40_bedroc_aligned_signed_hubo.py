from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.run_stage40_bedroc_aligned_signed_hubo import (
    class_contrast,
    early_rank_utilities,
    signed_stable,
)


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="ascii"))


def test_class_contrast_is_positive_when_actives_rank_earlier() -> None:
    ranks = np.asarray([[0.05], [0.1], [0.7], [0.8]])
    assert class_contrast(ranks, np.asarray([1, 1, 0, 0]), 20.0)[0] > 0


def test_early_rank_utility_returns_finite_subset_values() -> None:
    ranks = np.asarray(
        [
            [[0.1, 0.8], [0.2, 0.7], [0.8, 0.2], [0.9, 0.1]],
            [[0.1, 0.7], [0.3, 0.8], [0.7, 0.3], [0.8, 0.2]],
            [[0.2, 0.9], [0.1, 0.8], [0.9, 0.2], [0.8, 0.1]],
        ],
        dtype=float,
    )
    values = early_rank_utilities(ranks, np.asarray([1, 1, 0, 0]), [(0,), (1,)], 20.0)
    assert values.shape == (2,)
    assert np.all(np.isfinite(values))


def test_signed_stability_retains_repeated_negative_redundancy() -> None:
    assert signed_stable([-0.02, -0.03, -0.01], 2 / 3, 0.5, 0.001) < 0
    assert signed_stable([0.02, -0.03, 0.001], 2 / 3, 0.5, 0.001) == 0


def test_stage40_final_internal_objective_gate_fails() -> None:
    result = load("stage40_bedroc_aligned_signed_hubo_result.json")
    assert result["status"] == "stage40_bedroc_aligned_signed_hubo_complete"
    assert len(result["cells"]) == 32
    assert result["summary"]["positive_holdout_vs_legacy_cell_count"] == 10
    assert result["summary"]["positive_solver_gap_cell_count"] == 0
    assert result["summary"]["per_target"]["MK14"]["mean_train_objective_early_rank_spearman"] > 0.85
    assert result["decision"]["bedroc_aligned_objective_supported"] is False
    assert result["decision"]["small_pool_classical_difficulty_detected"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False


def test_stage40_audit_passes() -> None:
    audit = load("stage40_bedroc_aligned_signed_hubo_audit.json")
    assert audit["status"] == "stage40_bedroc_aligned_signed_hubo_audit_ok"
    assert all(audit["checks"].values())
    assert all(audit["cell_checks"].values())
    assert audit["maximum_absolute_recalculation_difference"] <= 1e-12
