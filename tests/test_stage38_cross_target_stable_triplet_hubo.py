from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.run_stage38_cross_target_stable_triplet_hubo import (
    fixed_size_strong_search,
    robust_utilities,
    stable_lcb,
)


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="ascii"))


def test_robust_utilities_returns_one_value_per_subset() -> None:
    ranks = np.asarray(
        [
            [[0.1, 0.8], [0.2, 0.7], [0.8, 0.2], [0.9, 0.1]],
            [[0.1, 0.7], [0.3, 0.8], [0.7, 0.3], [0.8, 0.2]],
            [[0.2, 0.9], [0.1, 0.8], [0.9, 0.2], [0.8, 0.1]],
        ],
        dtype=float,
    )
    values = robust_utilities(ranks, np.asarray([1, 1, 0, 0]), [(0,), (1,)], 20.0)
    assert values.shape == (2,)
    assert np.all(np.isfinite(values))


def test_stable_lcb_requires_repeated_positive_residuals() -> None:
    assert stable_lcb([0.01, 0.02, -0.001], 2 / 3, 0.5, 0.002) > 0
    assert stable_lcb([0.01, -0.001, -0.002], 2 / 3, 0.5, 0.002) == 0


def test_fixed_size_strong_search_can_use_intermediate_states() -> None:
    values = {
        (0,): 0.0,
        (1,): 0.1,
        (2,): 0.2,
        (0, 1): 0.2,
        (0, 2): 0.4,
        (1, 2): 0.3,
    }
    subset, record = fixed_size_strong_search(values, 3, 2, 2)
    assert subset == (0, 2)
    assert record["beam_target_start_count"] == 2


def test_stage38_result_retains_signal_but_fails_support_gate() -> None:
    result = load("stage38_cross_target_stable_triplet_hubo_result.json")
    assert result["status"] == "stage38_cross_target_stable_triplet_hubo_complete"
    assert len(result["cells"]) == 32
    assert result["summary"]["retained_triplet_model_count"] == 24
    assert result["summary"]["positive_solver_gap_cell_count"] == 0
    assert result["summary"]["per_target"]["MK14"]["mean_train_objective_spearman"] > 0.8
    assert result["summary"]["per_target"]["PPARG"]["mean_train_objective_spearman"] > 0.8
    assert result["decision"]["stable_triplet_objective_supported"] is False
    assert result["decision"]["stage39_quadratization_authorized"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False


def test_stage38_audit_passes() -> None:
    audit = load("stage38_cross_target_stable_triplet_hubo_audit.json")
    assert audit["status"] == "stage38_cross_target_stable_triplet_hubo_audit_ok"
    assert all(audit["checks"].values())
    assert all(audit["cell_checks"].values())
    assert audit["maximum_absolute_recalculation_difference"] <= 1e-12
