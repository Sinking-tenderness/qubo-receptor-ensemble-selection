import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage22_structural_state_coverage_qubo import (
    build_coverage_terms,
    qubo_energy,
)
from scripts.run_stage26_variable_budget_consensus_qubo import (
    assignment_for_subset,
    build_consensus_qubo,
    consensus_components,
)


def frozen_objective() -> dict[str, float | int]:
    return {
        "maximum_selected": 2,
        "neighborhood_fraction": 0.5,
        "single_coverage_weight": 0.25,
        "double_coverage_weight": 0.75,
        "pair_diversity_weight": 0.10,
        "per_conformer_cost": 0.04,
        "budget_constraint_penalty": 20.0,
        "coverage_constraint_penalty": 100.0,
    }


def test_second_hit_has_larger_marginal_reward() -> None:
    masks = [0b11, 0b11]
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    objective = frozen_objective()
    empty = consensus_components((), masks, matrix, 2, 0.25, 0.75, 0.10, 0.04)
    first = consensus_components((0,), masks, matrix, 2, 0.25, 0.75, 0.10, 0.04)
    second = consensus_components((0, 1), masks, matrix, 2, 0.25, 0.75, 0.10, 0.04)
    first_gain = first["composite_objective"] - empty["composite_objective"]
    second_gain = second["composite_objective"] - first["composite_objective"]
    assert second_gain > first_gain


def test_reduced_objective_matches_full_qubo() -> None:
    ids = ["A", "B", "C", "D"]
    matrix = np.array(
        [
            [0.0, 0.2, 0.8, 1.0],
            [0.2, 0.0, 0.7, 0.9],
            [0.8, 0.7, 0.0, 0.3],
            [1.0, 0.9, 0.3, 0.0],
        ]
    )
    objective = frozen_objective()
    terms = build_coverage_terms(ids, matrix, 0.5)
    masks = [int(terms["coverage_masks"][value]) for value in ids]
    qubo = build_consensus_qubo(ids, matrix, terms, objective)
    for selected in ((), (0,), (0, 1), (0, 2)):
        metrics = consensus_components(selected, masks, matrix, 2, 0.25, 0.75, 0.10, 0.04)
        energy = qubo_energy(qubo, assignment_for_subset(selected, ids, terms, qubo))
        assert abs(energy + metrics["composite_objective"]) < 1e-8


def test_stage26_frozen_config_is_structure_only() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/stage26_variable_budget_consensus_qubo.json").read_text(encoding="ascii")
    )
    assert config["evidence_timing"]["stage26_objective_outcomes_known_before_freeze"] is False
    assert config["evidence_timing"]["docking_scores_permitted"] is False
    assert config["evidence_timing"]["ligand_labels_permitted"] is False
    assert config["objective"]["double_coverage_weight"] > config["objective"]["single_coverage_weight"]
    assert config["objective"]["maximum_selected"] == 8


def test_stage26_result_and_audit_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    result_path = root / "data/stage26_variable_budget_consensus_qubo_result.json"
    audit_path = root / "data/stage26_variable_budget_consensus_qubo_audit.json"
    if not result_path.exists() or not audit_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="ascii"))
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    assert result["status"] == "stage26_variable_budget_consensus_qubo_complete"
    assert len(result["target_records"]) == 3
    assert audit["status"] == "stage26_variable_budget_consensus_qubo_audit_ok"
    assert audit["checks"]["data_boundary_zero"] is True
