from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.run_stage33_pparg_sparse_hardware_qubo import (
    build_domain_wall_qubo,
    qubo_energy,
    selected_to_domain_wall,
    sparse_objective,
)


ROOT = Path(__file__).resolve().parents[1]


def synthetic_model() -> dict:
    width = 2
    count = 16
    pair_matrix = np.zeros((count, count), dtype=float)
    pair = {(0, 2): -0.1, (1, 3): -0.2}
    for edge, value in pair.items():
        pair_matrix[edge] = value
        pair_matrix[edge[::-1]] = value
    return {
        "per_start": width,
        "groups": [tuple(range(group * width, (group + 1) * width)) for group in range(8)],
        "unary": np.linspace(0.01, 0.16, count),
        "pair": pair,
        "pair_matrix": pair_matrix,
        "k": 8,
    }


def test_domain_wall_energy_matches_negative_feasible_objective() -> None:
    model = synthetic_model()
    selected = tuple(group[0] if index % 2 == 0 else group[1] for index, group in enumerate(model["groups"]))
    qubo = build_domain_wall_qubo(model, 2.0)
    bits = selected_to_domain_wall(selected, model)
    assert abs(qubo_energy(bits, qubo) + sparse_objective(selected, model)) < 1e-12


def test_stage33_completed_with_frozen_boundaries() -> None:
    result = json.loads((ROOT / "data/stage33_pparg_sparse_hardware_qubo_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "stage33_pparg_sparse_hardware_qubo_complete"
    assert result["data_boundary"] == {
        "docking_scores_read": 0,
        "fresh_validation_rows_read": 0,
        "ligand_labels_read": 0,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
        "test_rows_read": 0,
    }
    assert result["decision"]["domain_wall_equivalence_gate_passed"] is True
    assert result["decision"]["full_pool_sparsity_gate_passed"] is True
    assert result["decision"]["quantum_advantage_claim_authorized"] is False


def test_stage33_independent_audit_passed() -> None:
    audit = json.loads((ROOT / "data/stage33_pparg_sparse_hardware_qubo_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "stage33_pparg_sparse_hardware_qubo_audit_ok"
    assert all(audit["checks"].values())
