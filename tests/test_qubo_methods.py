import pytest

from qubo_receptor_ensemble.methods import (
    MethodRegistryError,
    check_method_capability,
    get_method_spec,
    list_method_ids,
    resolve_method_requests,
)
from qubo_receptor_ensemble.solvers import ProblemError, build_problem, solve_problem


def _rows() -> list[dict[str, object]]:
    return [
        {"ligand_id": "A1", "label": "active", "R1": -10.0, "R2": -9.0},
        {"ligand_id": "D1", "label": "decoy", "R1": -5.0, "R2": -4.0},
    ]


def test_registry_contains_historical_method_families() -> None:
    method_ids = set(list_method_ids())

    assert {
        "basic_utility",
        "bedroc20_pair_synergy",
        "rank_sensitive_pair",
        "normalized_coverage",
        "auxiliary_coverage",
        "rankbin_bedroc20",
        "structure_aware",
        "signed_hubo",
        "dirac_global",
    }.issubset(method_ids)


def test_registry_exposes_method_provenance_and_input_contract() -> None:
    spec = get_method_spec("bedroc20_pair_synergy")

    assert spec.method_id == "bedroc20_pair_synergy"
    assert spec.formulation_kind == "qubo"
    assert spec.provenance
    assert "ligand_id" in spec.required_columns
    assert spec.defaults["bedroc_alpha"] == 20.0


def test_score_matrix_method_is_ready_without_extra_artifacts() -> None:
    capability = check_method_capability(
        "basic_utility",
        _rows(),
        available_inputs=(),
    )

    assert capability.status == "ready"
    assert capability.missing == ()


def test_structural_method_reports_missing_artifacts_without_fallback() -> None:
    capability = check_method_capability(
        "structure_aware",
        _rows(),
        available_inputs=(),
    )

    assert capability.status == "unsupported_for_input"
    assert "structural_features" in capability.missing


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(MethodRegistryError, match="unknown method"):
        get_method_spec("not_a_method")


def test_bedroc20_pair_synergy_builds_fixed_cardinality_qubo() -> None:
    problem = build_problem(
        _rows(),
        {
            "type": "receptor_subset",
            "strategy": "method_registry",
            "method_id": "bedroc20_pair_synergy",
            "receptor_ids": ["R1", "R2"],
            "target_size": 2,
        },
    )

    qubo = problem.formulation["qubo"]

    assert problem.strategy == "qubo"
    assert qubo["method_id"] == "bedroc20_pair_synergy"
    assert qubo["bedroc_alpha"] == 20.0
    assert qubo["fixed_cardinality"] is True
    assert "R1__R2" in qubo["quadratic_coefficients"]


def test_pair_method_can_be_solved_by_exact_backend() -> None:
    problem = build_problem(
        _rows(),
        {
            "type": "receptor_subset",
            "strategy": "method_registry",
            "method_id": "rank_sensitive_pair",
            "receptor_ids": ["R1", "R2"],
            "target_size": 1,
        },
    )

    result = solve_problem(problem, "exact")

    assert len(result.subset) == 1
    assert result.strategy == "qubo"
    assert result.metadata["fixed_cardinality"] is True


def test_unavailable_specialized_method_is_rejected() -> None:
    with pytest.raises(ProblemError, match="unsupported for current inputs"):
        build_problem(
            _rows(),
            {
                "type": "receptor_subset",
                "strategy": "method_registry",
                "method_id": "structure_aware",
                "receptor_ids": ["R1", "R2"],
                "target_size": 1,
            },
        )


def test_compare_problem_requests_inherit_bedroc20_defaults() -> None:
    requests = resolve_method_requests(
        {
            "type": "receptor_subset",
            "mode": "compare",
            "target_size": 2,
            "utility_metric": "bedroc",
            "bedroc_alpha": 20.0,
            "methods": [
                {"id": "basic_utility"},
                {"id": "bedroc20_pair_synergy", "decoy_penalty_lambda": 1.5},
            ],
        }
    )

    assert [item["method_id"] for item in requests] == [
        "basic_utility",
        "bedroc20_pair_synergy",
    ]
    assert all(item["bedroc_alpha"] == 20.0 for item in requests)
    assert all(item["target_size"] == 2 for item in requests)
    assert requests[1]["decoy_penalty_lambda"] == 1.5
