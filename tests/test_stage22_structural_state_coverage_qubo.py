import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage22_structural_state_coverage_qubo import (
    assignment_for_subset,
    build_auxiliary_qubo,
    build_coverage_terms,
    direct_greedy,
    objective_components,
    qubo_energy,
    run_restarts,
)
from scripts.diagnose_stage22_beam_baseline import beam_search


def synthetic_problem() -> tuple[list[str], np.ndarray]:
    ids = ["A", "B", "C", "D"]
    matrix = np.array(
        [
            [0.0, 0.2, 0.8, 1.0],
            [0.2, 0.0, 0.7, 0.9],
            [0.8, 0.7, 0.0, 0.3],
            [1.0, 0.9, 0.3, 0.0],
        ]
    )
    return ids, matrix


def test_auxiliary_qubo_matches_reduced_objective_for_all_fixed_k_states() -> None:
    ids, matrix = synthetic_problem()
    terms = build_coverage_terms(ids, matrix, 0.5)
    for k in (1, 2, 3):
        qubo = build_auxiliary_qubo(ids, matrix, terms, k, 0.15, 20.0, 100.0)
        for subset in itertools.combinations(ids, k):
            assignment = assignment_for_subset(subset, terms, qubo)
            energy = qubo_energy(qubo, assignment)
            objective = objective_components(
                subset, ids, matrix, terms, 0.15
            )["composite_objective"]
            assert energy == pytest.approx(-objective, abs=1e-8)


def test_coverage_solver_is_deterministic() -> None:
    ids, matrix = synthetic_problem()
    terms = build_coverage_terms(ids, matrix, 0.5)
    assert direct_greedy(ids, matrix, terms, 2, 0.15) == direct_greedy(
        ids, matrix, terms, 2, 0.15
    )
    first = run_restarts(ids, matrix, terms, 2, "A", 0.15, 8, 20260802)
    second = run_restarts(ids, matrix, terms, 2, "A", 0.15, 8, 20260802)
    assert first == second


def test_beam_baseline_is_deterministic_and_returns_fixed_cardinality() -> None:
    ids, matrix = synthetic_problem()
    terms = build_coverage_terms(ids, matrix, 0.5)
    first_subset, first_metrics = beam_search(ids, matrix, terms, 2, 0.15, 4)
    second_subset, second_metrics = beam_search(ids, matrix, terms, 2, 0.15, 4)
    assert len(first_subset) == 2
    assert first_subset == second_subset
    assert first_metrics == second_metrics


def test_stage22_config_is_structure_only_and_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/stage22_structural_state_coverage_qubo.json").read_text(
            encoding="ascii"
        )
    )
    assert config["evidence_timing"]["new_docking_jobs"] is False
    assert config["evidence_timing"]["quantum_hardware_execution"] is False
    assert config["diagnostic"]["primary_neighborhood_fraction"] == 0.10
    assert config["diagnostic"]["restart_count"] == 32
    assert config["go_no_go"]["required_common_k"] == 8
    paths = [
        value
        for target in config["targets"].values()
        for value in target["inputs"].values()
    ]
    assert not any("fresh_validation" in value or "locked_test" in value for value in paths)


def test_stage22_result_and_audit_are_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "data/stage22_structural_state_coverage_qubo_result.json").read_text(
            encoding="ascii"
        )
    )
    audit = json.loads(
        (root / "data/stage22_structural_state_coverage_qubo_audit.json").read_text(
            encoding="ascii"
        )
    )
    assert result["status"] == "stage22_structural_state_coverage_qubo_complete"
    assert result["decision"]["structural_coverage_gate_passed"] is False
    assert result["decision"]["new_docking_jobs_authorized_by_this_stage"] is False
    assert result["decision"]["quantum_hardware_authorized"] is False
    assert audit["status"] == "stage22_structural_state_coverage_qubo_audit_ok"
    assert audit["coverage"]["selection_rows_recomputed"] == 120
    assert audit["coverage"]["restart_rows_recomputed"] == 960
    assert audit["checks"]["go_no_go_decision_recomputed"] is True
    beam = json.loads(
        (root / "data/stage22_beam_baseline_diagnostic.json").read_text(
            encoding="ascii"
        )
    )
    milp = json.loads(
        (root / "data/stage22_global_milp_diagnostic.json").read_text(
            encoding="ascii"
        )
    )
    assert beam["status"] == "stage22_beam_baseline_diagnostic_complete"
    assert milp["status"] == "stage22_global_milp_diagnostic_complete"
    assert all(row["solver_status"] == 1 for row in milp["rows"])
