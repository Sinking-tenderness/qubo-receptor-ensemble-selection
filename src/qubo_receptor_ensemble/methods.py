"""Registry and input contracts for historical receptor-selection methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


class MethodRegistryError(ValueError):
    """Raised when a configured method is not registered."""


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    provenance: str
    formulation_kind: str
    builder: str
    required_columns: tuple[str, ...] = ("ligand_id", "label")
    required_inputs: tuple[str, ...] = ()
    supported_backends: tuple[str, ...] = ("exact", "greedy")
    defaults: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "provenance": self.provenance,
            "formulation_kind": self.formulation_kind,
            "builder": self.builder,
            "required_columns": list(self.required_columns),
            "required_inputs": list(self.required_inputs),
            "supported_backends": list(self.supported_backends),
            "defaults": dict(self.defaults or {}),
        }


@dataclass(frozen=True)
class MethodCapability:
    method_id: str
    status: str
    missing: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "status": self.status,
            "missing": list(self.missing),
        }


_BEDROC_DEFAULTS = {
    "utility_metric": "bedroc",
    "bedroc_alpha": 20.0,
}


def _spec(
    method_id: str,
    provenance: str,
    formulation_kind: str,
    builder: str,
    *,
    required_inputs: tuple[str, ...] = (),
    supported_backends: tuple[str, ...] = ("exact", "greedy"),
    defaults: Mapping[str, object] | None = None,
) -> MethodSpec:
    return MethodSpec(
        method_id=method_id,
        provenance=provenance,
        formulation_kind=formulation_kind,
        builder=builder,
        required_inputs=required_inputs,
        supported_backends=supported_backends,
        defaults={**_BEDROC_DEFAULTS, **dict(defaults or {})},
    )


_METHODS: dict[str, MethodSpec] = {
    "basic_utility": _spec(
        "basic_utility",
        "current canonical qubo.py; historical Stage02 baseline",
        "qubo",
        "basic_utility",
    ),
    "pair_utility": _spec(
        "pair_utility",
        "historical pair ensemble utility QUBO",
        "qubo",
        "pair_utility",
    ),
    "pair_synergy": _spec(
        "pair_synergy",
        "historical marginal pair-synergy QUBO",
        "qubo",
        "pair_synergy",
    ),
    "bedroc20_pair_synergy": _spec(
        "bedroc20_pair_synergy",
        "Stage99 robust two-support BEDROC20 pair QUBO",
        "qubo",
        "bedroc20_pair_synergy",
    ),
    "rank_sensitive_pair": _spec(
        "rank_sensitive_pair",
        "Stage42f/Stage60 BEDROC20 rank-sensitive pair complementarity",
        "qubo",
        "rank_sensitive_pair",
    ),
    "uncertainty_shrunk": _spec(
        "uncertainty_shrunk",
        "Stage05 and Stage64 uncertainty-shrunk QUBO",
        "qubo",
        "uncertainty_shrunk",
        required_inputs=("seed_matrices",),
    ),
    "stability_qubo": _spec(
        "stability_qubo",
        "historical cross-seed stability QUBO",
        "qubo",
        "stability_qubo",
        required_inputs=("seed_matrices",),
    ),
    "normalized_coverage": _spec(
        "normalized_coverage",
        "current normalized receptor coverage formulation",
        "qubo",
        "normalized_coverage",
    ),
    "auxiliary_coverage": _spec(
        "auxiliary_coverage",
        "Stage19h/Stage66 active coverage and decoy exposure QUBO",
        "qubo_auxiliary",
        "auxiliary_coverage",
    ),
    "rankbin_bedroc20": _spec(
        "rankbin_bedroc20",
        "Stage67 BEDROC20 rank-bin auxiliary-variable QUBO",
        "qubo_auxiliary",
        "rankbin_bedroc20",
        defaults={"bin_count": 32},
    ),
    "structure_aware": _spec(
        "structure_aware",
        "Stage21 structure-aware conformation-pool QUBO",
        "qubo",
        "structure_aware",
        required_inputs=("structural_features",),
    ),
    "structural_state_coverage": _spec(
        "structural_state_coverage",
        "Stage22/Stage23 structural-state coverage QUBO",
        "qubo_auxiliary",
        "structural_state_coverage",
        required_inputs=("structural_states",),
    ),
    "multiscale_coverage": _spec(
        "multiscale_coverage",
        "Stage24 multiscale structural coverage QUBO",
        "qubo_auxiliary",
        "multiscale_coverage",
        required_inputs=("structural_states",),
    ),
    "group_balanced_state": _spec(
        "group_balanced_state",
        "Stage30 multiple-choice group-balanced state QUBO",
        "qubo_auxiliary",
        "group_balanced_state",
        required_inputs=("frame_groups", "structural_states"),
    ),
    "quality_plateau_portfolio": _spec(
        "quality_plateau_portfolio",
        "Stage68 quality-plateau functional-diversity portfolio QUBO",
        "qubo_auxiliary",
        "quality_plateau_portfolio",
        required_inputs=("seed_matrices", "uncertainty_intervals"),
    ),
    "signed_hubo": _spec(
        "signed_hubo",
        "Stage40 BEDROC-aligned signed Mobius HUBO",
        "qubo_encoded_hubo",
        "signed_hubo",
        supported_backends=("exact",),
    ),
    "constraint_aware": _spec(
        "constraint_aware",
        "Stage70 constraint-aware exact-penalty encoding",
        "qubo_encoded_hubo",
        "constraint_aware",
        required_inputs=("constraint_spec",),
    ),
    "quality_shell": _spec(
        "quality_shell",
        "Stage83 quality-shell scalarized QUBO",
        "qubo_encoded_hubo",
        "quality_shell",
        required_inputs=("constraint_spec",),
    ),
    "constraint_native_cqm": _spec(
        "constraint_native_cqm",
        "Stage72 constraint-native CQM formulation",
        "cqm",
        "constraint_native_cqm",
        required_inputs=("constraint_spec",),
        supported_backends=("cqm",),
    ),
    "dirac_global": _spec(
        "dirac_global",
        "Stage81 Dirac-compatible global variable-k encoding",
        "qubo_encoded_hubo",
        "dirac_global",
        required_inputs=("dirac_backend",),
        supported_backends=("dirac",),
    ),
    "lagrangian_fixed_k": _spec(
        "lagrangian_fixed_k",
        "Stage82 Lagrangian fixed-k Dirac encoding",
        "qubo_encoded_hubo",
        "lagrangian_fixed_k",
        required_inputs=("dirac_backend",),
        supported_backends=("dirac",),
    ),
}


def list_method_ids() -> tuple[str, ...]:
    """Return stable method IDs in deterministic order."""
    return tuple(sorted(_METHODS))


def get_method_spec(method_id: str) -> MethodSpec:
    """Return a registered method or raise a configuration error."""
    normalized = str(method_id).strip()
    try:
        return _METHODS[normalized]
    except KeyError as exc:
        raise MethodRegistryError(f"unknown method: {normalized}") from exc


def check_method_capability(
    method_id: str,
    rows: Iterable[Mapping[str, object]],
    available_inputs: Iterable[str] = (),
) -> MethodCapability:
    """Check row fields and external artifacts without building a formulation."""
    spec = get_method_spec(method_id)
    row_list = list(rows)
    row_fields: set[str] = set()
    for row in row_list:
        row_fields.update(str(key) for key in row)
    available = {str(value) for value in available_inputs}
    missing = sorted(
        set(spec.required_columns).difference(row_fields)
        | set(spec.required_inputs).difference(available)
    )
    return MethodCapability(
        method_id=spec.method_id,
        status="ready" if not missing else "unsupported_for_input",
        missing=tuple(missing),
    )


def resolve_method_requests(problem_config: Mapping[str, object]) -> list[dict[str, object]]:
    """Expand one problem block into normalized single-method requests."""
    base = {
        str(key): value
        for key, value in problem_config.items()
        if key not in {"mode", "methods", "strategy", "method_id", "id"}
    }
    base.setdefault("utility_metric", "bedroc")
    base.setdefault("bedroc_alpha", 20.0)
    configured = problem_config.get("methods")
    if configured is None:
        method_id = str(
            problem_config.get(
                "method_id",
                "normalized_coverage"
                if problem_config.get("strategy") == "normalized_qubo"
                else "basic_utility",
            )
        )
        return [{**base, "method_id": method_id, "strategy": "method_registry"}]
    if not isinstance(configured, list) or not configured:
        raise MethodRegistryError("problem.methods must be a non-empty list")
    requests: list[dict[str, object]] = []
    for index, item in enumerate(configured):
        if isinstance(item, str):
            item = {"id": item}
        if not isinstance(item, dict):
            raise MethodRegistryError(f"problem.methods[{index}] must be an object")
        method_id = str(item.get("id", item.get("method_id", ""))).strip()
        if not method_id:
            raise MethodRegistryError(f"problem.methods[{index}] requires id")
        get_method_spec(method_id)
        request = {**base, **item}
        request.pop("id", None)
        request.pop("method_id", None)
        request["method_id"] = method_id
        request["strategy"] = "method_registry"
        requests.append(request)
    return requests
