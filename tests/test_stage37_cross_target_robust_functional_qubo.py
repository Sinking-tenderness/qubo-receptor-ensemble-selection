from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.run_stage37_cross_target_robust_functional_qubo import (
    fit_rank_transform,
    objective_components,
    objective_value,
)


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / "data" / name).read_text(encoding="ascii"))


def test_train_rank_transform_uses_lower_scores_as_better() -> None:
    train = np.asarray([-9.0, -8.0, -7.0, -6.0])
    ranks = fit_rank_transform(train, np.asarray([-10.0, -8.0, -5.0]))
    assert np.allclose(ranks, [0.1, 0.4, 0.9])


def test_functional_components_distinguish_stable_active_support_and_decoy_exposure() -> None:
    favorable = np.asarray(
        [
            [[True, False], [False, True]],
            [[True, True], [False, False]],
            [[False, True], [True, False]],
        ],
        dtype=bool,
    )
    components = objective_components(favorable, np.asarray([1, 0]), (0, 1))
    assert components["active_majority_seed_coverage"] == 1.0
    assert components["active_all_seed_coverage"] == 1.0
    assert components["active_double_receptor_majority_seed_support"] == 0.0
    assert components["decoy_any_seed_exposure"] == 1.0
    config = {
        "maximum_subset_size": 6,
        "weights": {
            "active_majority_seed_coverage": 0.35,
            "active_all_seed_coverage": 0.25,
            "active_double_receptor_majority_seed_support": 0.2,
            "decoy_any_seed_exposure": 0.15,
            "receptor_cost": 0.05,
        },
    }
    assert abs(objective_value(components, 2, config) - (0.35 + 0.25 - 0.15 - 0.05 * 2 / 6)) < 1e-12


def test_stage37_result_is_complete_but_does_not_authorize_hardware() -> None:
    result = load("stage37_cross_target_robust_functional_qubo_result.json")
    assert result["status"] == "stage37_cross_target_robust_functional_qubo_complete"
    assert len(result["cells"]) == 8
    assert result["summary"]["positive_train_gap_cell_count"] == 0
    assert result["decision"]["functional_objective_supported"] is False
    assert result["decision"]["stage38_sparse_auxiliary_qubo_authorized"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False


def test_stage37_independent_audit_passes() -> None:
    audit = load("stage37_cross_target_robust_functional_qubo_audit.json")
    assert audit["status"] == "stage37_cross_target_robust_functional_qubo_audit_ok"
    assert all(audit["checks"].values())
    assert all(audit["cell_checks"].values())
    assert audit["maximum_absolute_recalculation_difference"] <= 1e-12
