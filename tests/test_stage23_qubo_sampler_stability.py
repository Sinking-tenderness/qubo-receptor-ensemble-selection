import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage22_structural_state_coverage_qubo import (
    build_coverage_terms,
    objective_components,
)
from scripts.run_stage23_qubo_sampler_stability import anneal_read, fast_value


def problem() -> tuple[list[str], np.ndarray]:
    ids = ["A", "B", "C", "D"]
    matrix = np.array(
        [[0.0, 0.2, 0.8, 1.0], [0.2, 0.0, 0.7, 0.9], [0.8, 0.7, 0.0, 0.3], [1.0, 0.9, 0.3, 0.0]]
    )
    return ids, matrix


def test_fast_value_matches_frozen_objective() -> None:
    ids, matrix = problem()
    terms = build_coverage_terms(ids, matrix, 0.5)
    masks = [terms["coverage_masks"][value] for value in ids]
    indices = [0, 3]
    expected = objective_components(("A", "D"), ids, matrix, terms, 0.15)[
        "composite_objective"
    ]
    assert fast_value(indices, masks, matrix, 0.15) == pytest.approx(expected)


def test_annealer_is_deterministic_for_fixed_seed() -> None:
    ids, matrix = problem()
    terms = build_coverage_terms(ids, matrix, 0.5)
    masks = [terms["coverage_masks"][value] for value in ids]
    first = anneal_read(4, 2, masks, matrix, 0.15, 500, 0.04, 0.0002, random.Random(23))
    second = anneal_read(4, 2, masks, matrix, 0.15, 500, 0.04, 0.0002, random.Random(23))
    assert first == second


def test_stage23_result_and_audit() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "data/stage23_qubo_sampler_stability_result.json").read_text(encoding="ascii"))
    audit = json.loads((root / "data/stage23_qubo_sampler_stability_audit.json").read_text(encoding="ascii"))
    assert result["status"] == "stage23_qubo_sampler_stability_complete"
    assert result["decision"]["primary_pass"] is True
    assert result["decision"]["sampler_stability_gate_passed"] is False
    assert audit["status"] == "stage23_qubo_sampler_stability_audit_ok"
    assert audit["coverage"]["read_rows_recomputed"] == 1536
    assert audit["coverage"]["batch_rows_recomputed"] == 24
