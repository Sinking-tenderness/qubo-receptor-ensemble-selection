"""Stable solver adapter contracts for the canonical experiment pipeline."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Protocol

from .method_formulations import (
    build_auxiliary_coverage_qubo,
    build_rank_pair_qubo,
    build_rankbin_qubo,
)
from .methods import check_method_capability, get_method_spec
from .qubo import build_qubo, objective


class ProblemError(ValueError):
    """Raised when a canonical problem cannot be constructed or solved."""


@dataclass(frozen=True)
class Problem:
    """A solver-independent problem with an algorithm-specific formulation."""

    problem_type: str
    strategy: str
    receptor_ids: tuple[str, ...]
    train_rows: tuple[dict[str, object], ...]
    parameters: dict[str, object]
    formulation: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "type": self.problem_type,
            "strategy": self.strategy,
            "receptor_ids": list(self.receptor_ids),
            "train_row_count": len(self.train_rows),
            "parameters": self.parameters,
            "formulation": self.formulation,
        }


@dataclass(frozen=True)
class SolverResult:
    """Normalized output shared by all registered solver backends."""

    backend: str
    strategy: str
    subset: tuple[str, ...]
    objective: float
    coefficients: dict[str, object]
    metadata: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "strategy": self.strategy,
            "subset": list(self.subset),
            "objective": self.objective,
            "coefficients": self.coefficients,
            "metadata": self.metadata,
        }


class SolverAdapter(Protocol):
    name: str

    def solve(self, problem: Problem) -> SolverResult:
        """Solve a problem and return the canonical result contract."""


def _required_int(parameters: dict[str, object], key: str) -> int:
    value = parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProblemError(f"problem.{key} must be an integer")
    if int(value) != value:
        raise ProblemError(f"problem.{key} must be an integer")
    return int(value)


def _float_parameter(
    parameters: dict[str, object], key: str, default: float
) -> float:
    value = parameters.get(key, default)
    if isinstance(value, bool):
        raise ProblemError(f"problem.{key} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProblemError(f"problem.{key} must be numeric") from exc


def _receptor_ids(config: dict[str, object]) -> tuple[str, ...]:
    values = config.get("receptor_ids")
    if not isinstance(values, list) or not values:
        raise ProblemError("problem.receptor_ids must be a non-empty list")
    receptor_ids = tuple(str(value) for value in values)
    if len(receptor_ids) != len(set(receptor_ids)):
        raise ProblemError("problem.receptor_ids must be unique")
    return receptor_ids


def _validate_rows(
    rows: list[dict[str, object]], receptor_ids: tuple[str, ...]
) -> tuple[dict[str, object], ...]:
    if not rows:
        raise ProblemError("cannot build a problem from empty training data")
    normalized: list[dict[str, object]] = []
    seen_ligands: set[str] = set()
    for row in rows:
        ligand_id = str(row.get("ligand_id", ""))
        label = str(row.get("label", ""))
        if not ligand_id or not label:
            raise ProblemError("training rows require ligand_id and label")
        if ligand_id in seen_ligands:
            raise ProblemError(f"duplicate ligand_id in training data: {ligand_id}")
        seen_ligands.add(ligand_id)
        copied = dict(row)
        for receptor_id in receptor_ids:
            if copied.get(receptor_id, "") == "":
                raise ProblemError(
                    f"missing receptor score: {ligand_id} / {receptor_id}"
                )
            try:
                copied[receptor_id] = float(copied[receptor_id])
            except (TypeError, ValueError) as exc:
                raise ProblemError(
                    f"receptor score is not numeric: {ligand_id} / {receptor_id}"
                ) from exc
        normalized.append(copied)
    return tuple(normalized)


def build_problem(
    rows: list[dict[str, object]], problem_config: dict[str, object]
) -> Problem:
    """Build the canonical problem and its strategy-specific formulation."""
    problem_type = str(problem_config.get("type", ""))
    strategy = str(problem_config.get("strategy", ""))
    if problem_type != "receptor_subset":
        raise ProblemError(f"unsupported problem type: {problem_type}")
    receptor_ids = _receptor_ids(problem_config)
    train_rows = _validate_rows(rows, receptor_ids)
    parameters = dict(problem_config)

    method_id = str(parameters.get("method_id", ""))
    if strategy in {"qubo", "basic_qubo"} and not method_id:
        method_id = "basic_utility"
    elif strategy == "normalized_qubo" and not method_id:
        method_id = "normalized_coverage"
    elif strategy == "method_registry" and not method_id:
        raise ProblemError("problem.method_id is required for method_registry")
    parameters["method_id"] = method_id

    if strategy in {"qubo", "basic_qubo"}:
        target_size = _required_int(parameters, "target_size")
        if not 0 <= target_size <= len(receptor_ids):
            raise ProblemError("problem.target_size must be within the receptor pool")
        weights = parameters.get("weights", {})
        if not isinstance(weights, dict):
            raise ProblemError("problem.weights must be an object")
        qubo = build_qubo(
            list(train_rows),
            list(receptor_ids),
            target_size,
            _float_parameter(parameters, "redundancy_weight", weights.get("redundancy", 0.25)),
            _float_parameter(parameters, "count_weight", weights.get("count", 0.10)),
            _float_parameter(parameters, "size_weight", weights.get("size", 1.0)),
            str(parameters.get("utility_metric", "bedroc")),
            str(parameters.get("utility_normalization", "none")),
            _float_parameter(parameters, "bedroc_alpha", 20.0),
            bool(parameters.get("fixed_cardinality", False)),
        )
        qubo["method_id"] = method_id
        formulation = {"qubo": qubo}
    elif strategy == "normalized_qubo":
        from scripts.normalized_receptor_qubo import build_normalized_terms

        fraction = _float_parameter(parameters, "coverage_fraction", 0.10)
        utility_metric = str(parameters.get("utility_metric", "bedroc"))
        formulation = {
            "terms": build_normalized_terms(
                list(train_rows),
                list(receptor_ids),
                fraction,
                utility_metric,
            )
        }
    elif strategy == "method_registry":
        spec = get_method_spec(method_id)
        capability = check_method_capability(
            method_id,
            train_rows,
            parameters.get("available_inputs", ()),
        )
        if capability.status != "ready":
            raise ProblemError(
                f"method {method_id} is unsupported for current inputs: "
                f"missing {', '.join(capability.missing)}"
            )
        target_size = _required_int(parameters, "target_size")
        if not 0 <= target_size <= len(receptor_ids):
            raise ProblemError("problem.target_size must be within the receptor pool")
        if method_id == "basic_utility":
            weights = parameters.get("weights", {})
            if not isinstance(weights, dict):
                raise ProblemError("problem.weights must be an object")
            qubo = build_qubo(
                list(train_rows),
                list(receptor_ids),
                target_size,
                _float_parameter(parameters, "redundancy_weight", weights.get("redundancy", 0.25)),
                _float_parameter(parameters, "count_weight", weights.get("count", 0.10)),
                _float_parameter(parameters, "size_weight", weights.get("size", 1.0)),
                str(parameters.get("utility_metric", "bedroc")),
                str(parameters.get("utility_normalization", "none")),
                _float_parameter(parameters, "bedroc_alpha", 20.0),
                bool(parameters.get("fixed_cardinality", False)),
            )
            qubo["method_id"] = method_id
            formulation = {"qubo": qubo, "method": spec.as_dict()}
        elif method_id == "normalized_coverage":
            from scripts.normalized_receptor_qubo import build_normalized_terms

            formulation = {
                "terms": build_normalized_terms(
                    list(train_rows),
                    list(receptor_ids),
                    _float_parameter(parameters, "coverage_fraction", 0.10),
                    str(parameters.get("utility_metric", "bedroc")),
                ),
                "method": spec.as_dict(),
            }
        elif method_id in {
            "pair_utility",
            "pair_synergy",
            "bedroc20_pair_synergy",
            "rank_sensitive_pair",
        }:
            formulation = {
                "qubo": build_rank_pair_qubo(
                    list(train_rows),
                    list(receptor_ids),
                    target_size,
                    parameters,
                    method_id,
                ),
                "method": spec.as_dict(),
            }
        elif method_id == "auxiliary_coverage":
            formulation = {
                "qubo": build_auxiliary_coverage_qubo(
                    list(train_rows),
                    list(receptor_ids),
                    target_size,
                    parameters,
                ),
                "method": spec.as_dict(),
            }
        elif method_id == "rankbin_bedroc20":
            formulation = {
                "qubo": build_rankbin_qubo(
                    list(train_rows),
                    list(receptor_ids),
                    target_size,
                    parameters,
                ),
                "method": spec.as_dict(),
            }
        else:
            raise ProblemError(
                f"method {method_id} is registered but has no current builder"
            )
        strategy = "normalized_qubo" if method_id == "normalized_coverage" else "qubo"
    else:
        raise ProblemError(f"unsupported problem strategy: {strategy}")

    return Problem(
        problem_type=problem_type,
        strategy=strategy,
        receptor_ids=receptor_ids,
        train_rows=train_rows,
        parameters=parameters,
        formulation=formulation,
    )


def _basic_qubo_result(
    problem: Problem, subset: tuple[str, ...], value: float, states: int
) -> SolverResult:
    qubo = problem.formulation["qubo"]
    assert isinstance(qubo, dict)
    metadata = {
        "objective": float(value),
        "states_evaluated": states,
        "selection_rule": "minimum objective, then lexicographic subset",
        "fixed_cardinality": bool(qubo.get("fixed_cardinality", False)),
        "method_id": qubo.get("method_id", "basic_utility"),
    }
    return SolverResult(
        backend="exact",
        strategy=problem.strategy,
        subset=subset,
        objective=float(value),
        coefficients=qubo,
        metadata=metadata,
    )


class ExactSolverAdapter:
    name = "exact"

    def solve(self, problem: Problem) -> SolverResult:
        if problem.strategy == "normalized_qubo":
            from scripts.normalized_receptor_qubo import exact_select

            terms = problem.formulation["terms"]
            assert isinstance(terms, dict)
            parameters = problem.parameters
            weights = parameters.get("weights", {})
            if not isinstance(weights, dict):
                raise ProblemError("problem.weights must be an object")
            target_size = _required_int(parameters, "target_size")
            subset, value, coefficients = exact_select(
                terms,
                list(problem.receptor_ids),
                target_size,
                {str(key): float(item) for key, item in weights.items()},
                _float_parameter(parameters, "size_penalty", 1.0),
                tuple(str(item) for item in parameters.get("required_receptors", [])),
            )
            search = coefficients.get("exact_search", {})
            states = int(search.get("states_evaluated", 0)) if isinstance(search, dict) else 0
            return SolverResult(
                backend=self.name,
                strategy=problem.strategy,
                subset=tuple(subset),
                objective=float(value),
                coefficients=coefficients,
                metadata={
                    "objective": float(value),
                    "states_evaluated": states,
                    "selection_rule": "minimum coefficient energy, then lexicographic subset",
                    "search": search,
                },
            )

        qubo = problem.formulation["qubo"]
        assert isinstance(qubo, dict)
        target_size = _required_int(problem.parameters, "target_size")
        if bool(qubo.get("fixed_cardinality", False)):
            candidates = [
                (subset, objective(subset, qubo))
                for subset in itertools.combinations(
                    problem.receptor_ids, target_size
                )
            ]
        else:
            candidates = [
                (subset, objective(subset, qubo))
                for size in range(len(problem.receptor_ids) + 1)
                for subset in itertools.combinations(problem.receptor_ids, size)
            ]
        subset, value = min(candidates, key=lambda item: (item[1], item[0]))
        if len(subset) != target_size:
            raise ProblemError(
                "problem size penalty did not enforce the requested target size"
            )
        return _basic_qubo_result(problem, subset, value, len(candidates))


class GreedySolverAdapter:
    name = "greedy"

    def solve(self, problem: Problem) -> SolverResult:
        if problem.strategy not in {"qubo", "basic_qubo"}:
            raise ProblemError(
                "greedy backend currently supports only the basic QUBO strategy"
            )
        qubo = problem.formulation["qubo"]
        assert isinstance(qubo, dict)
        target_size = _required_int(problem.parameters, "target_size")
        selected: tuple[str, ...] = ()
        states = 0
        while len(selected) < target_size:
            candidates = []
            for receptor_id in problem.receptor_ids:
                if receptor_id in selected:
                    continue
                candidate = tuple(sorted((*selected, receptor_id)))
                candidates.append((candidate, objective(candidate, qubo)))
            if not candidates:
                raise ProblemError("greedy solver ran out of receptor candidates")
            states += len(candidates)
            selected, _ = min(candidates, key=lambda item: (item[1], item[0]))
        value = objective(selected, qubo)
        return SolverResult(
            backend=self.name,
            strategy=problem.strategy,
            subset=selected,
            objective=float(value),
            coefficients=qubo,
            metadata={
                "objective": float(value),
                "states_evaluated": states,
                "selection_rule": "greedy minimum objective, then lexicographic subset",
                "fixed_cardinality": bool(qubo.get("fixed_cardinality", False)),
                "method_id": qubo.get("method_id", "basic_utility"),
            },
        )


_ADAPTERS: dict[str, SolverAdapter] = {
    "exact": ExactSolverAdapter(),
    "greedy": GreedySolverAdapter(),
}


def solve_problem(problem: Problem, backend: str) -> SolverResult:
    """Dispatch a problem through the registered solver adapter."""
    adapter = _ADAPTERS.get(backend)
    if adapter is None:
        raise ProblemError(f"no solver adapter is registered for backend: {backend}")
    method_id = str(problem.parameters.get("method_id", ""))
    if method_id:
        spec = get_method_spec(method_id)
        if backend not in spec.supported_backends:
            raise ProblemError(
                f"method {method_id} does not support solver backend: {backend}"
            )
    return adapter.solve(problem)
