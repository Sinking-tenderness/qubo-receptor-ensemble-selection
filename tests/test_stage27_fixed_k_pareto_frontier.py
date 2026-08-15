import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage22_structural_state_coverage_qubo import build_coverage_terms, qubo_energy
from scripts.run_stage27_fixed_k_pareto_frontier import (
    assignment_for_fixed_subset,
    benefit,
    build_fixed_k_qubo,
    supported_cost_intervals,
)


def objective() -> dict[str, object]:
    return {
        "maximum_selected": 8,
        "single_coverage_weight": 0.25,
        "double_coverage_weight": 0.75,
        "pair_diversity_weight": 0.10,
        "cardinality_penalty": 20.0,
        "coverage_constraint_penalty": 100.0,
    }


def test_supported_cost_intervals_are_derived_without_grid_search() -> None:
    rows = supported_cost_intervals({0: 0.0, 1: 0.2, 2: 0.35})
    assert [row["k"] for row in rows] == [0, 1, 2]
    by_k = {row["k"]: row for row in rows}
    assert abs(float(by_k[2]["cost_upper_inclusive"]) - 0.15) < 1e-12
    assert abs(float(by_k[1]["cost_lower_inclusive"]) - 0.15) < 1e-12
    assert abs(float(by_k[1]["cost_upper_inclusive"]) - 0.2) < 1e-12
    assert by_k[0]["cost_upper_inclusive"] == "inf"


def test_fixed_k_qubo_matches_reduced_benefit() -> None:
    ids = ["A", "B", "C", "D"]
    matrix = np.array(
        [[0.0, 0.2, 0.8, 1.0], [0.2, 0.0, 0.7, 0.9], [0.8, 0.7, 0.0, 0.3], [1.0, 0.9, 0.3, 0.0]]
    )
    terms = build_coverage_terms(ids, matrix, 0.5)
    masks = [int(terms["coverage_masks"][value]) for value in ids]
    frozen = objective()
    qubo = build_fixed_k_qubo(ids, matrix, terms, 2, frozen)
    for selected in ((0, 1), (0, 2), (2, 3)):
        metrics = benefit(selected, masks, matrix, frozen)
        energy = qubo_energy(qubo, assignment_for_fixed_subset(selected, ids, terms, qubo))
        assert abs(energy + metrics["composite_objective"]) < 1e-8


def test_stage27_config_is_frozen_and_cost_free() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/stage27_fixed_k_pareto_frontier.json").read_text(encoding="ascii"))
    assert config["evidence_timing"]["stage27_frontier_outcomes_known_before_freeze"] is False
    assert config["evidence_timing"]["docking_scores_permitted"] is False
    assert config["objective"]["per_conformer_cost_in_optimization"] == 0.0
    assert config["objective"]["k_values"] == list(range(1, 9))
    assert set(config["targets"]) == {"MK14", "PPARG", "BACE1", "EGFR", "FA10"}


def test_stage27_result_and_audit_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    result_path = root / "data/stage27_fixed_k_pareto_frontier_result.json"
    audit_path = root / "data/stage27_fixed_k_pareto_frontier_audit.json"
    if not result_path.exists() or not audit_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="ascii"))
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    assert result["status"] == "stage27_fixed_k_pareto_frontier_complete"
    assert len(result["target_records"]) == 5
    assert audit["status"] == "stage27_fixed_k_pareto_frontier_audit_ok"
    assert audit["checks"]["data_boundary_zero"] is True
