import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage22_structural_state_coverage_qubo import build_coverage_terms, qubo_energy
from scripts.run_stage24_multiscale_coverage_qubo import (
    anneal_read,
    assignment_for_subset,
    build_multiscale_qubo,
    fast_multiscale_value,
    multiscale_components,
)


def problem():
    ids = ["A", "B", "C", "D"]
    matrix = np.array([[0.0, 0.2, 0.8, 1.0], [0.2, 0.0, 0.7, 0.9], [0.8, 0.7, 0.0, 0.3], [1.0, 0.9, 0.3, 0.0]])
    fractions = [0.25, 0.5, 1.0]
    terms = [build_coverage_terms(ids, matrix, value) for value in fractions]
    masks = [[term["coverage_masks"][value] for value in ids] for term in terms]
    return ids, matrix, terms, masks


def test_multiscale_fast_and_qubo_energies_match() -> None:
    ids, matrix, terms, masks = problem()
    weights = [1 / 3] * 3
    subset = ("A", "D")
    objective = multiscale_components(subset, ids, matrix, terms, weights, 0.15)["composite_objective"]
    assert fast_multiscale_value([0, 3], masks, weights, matrix, 0.15) == pytest.approx(objective)
    qubo = build_multiscale_qubo(ids, matrix, terms, weights, 2, 0.15, 20.0, 100.0)
    energy = qubo_energy(qubo, assignment_for_subset(subset, terms, qubo))
    assert energy == pytest.approx(-objective, abs=1e-8)


def test_multiscale_annealer_is_deterministic() -> None:
    ids, matrix, terms, masks = problem()
    weights = [1 / 3] * 3
    first = anneal_read(4, 2, masks, weights, matrix, 0.15, 500, 0.04, 0.0002, random.Random(24))
    second = anneal_read(4, 2, masks, weights, matrix, 0.15, 500, 0.04, 0.0002, random.Random(24))
    assert first == second


def test_stage24_result_and_audit() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "data/stage24_multiscale_coverage_qubo_result.json").read_text(encoding="ascii"))
    audit = json.loads((root / "data/stage24_multiscale_coverage_qubo_audit.json").read_text(encoding="ascii"))
    assert result["status"] == "stage24_multiscale_coverage_qubo_complete"
    assert result["decision"]["multiscale_qubo_gate_passed"] is False
    assert all(record["within_tolerance_batch_fraction"] == 1.0 for record in result["target_records"].values())
    assert audit["status"] == "stage24_multiscale_coverage_qubo_audit_ok"
    assert audit["coverage"]["read_rows_recomputed"] == 512
    assert audit["coverage"]["batch_rows_recomputed"] == 8
