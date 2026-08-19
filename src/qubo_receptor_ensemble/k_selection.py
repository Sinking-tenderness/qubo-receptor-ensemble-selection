"""Pluggable policies for selecting the number of receptors/conformations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .solvers import SolverResult


class KSelectionError(ValueError):
    """Raised when candidate k values cannot be compared safely."""


@dataclass(frozen=True)
class KCandidate:
    """One solved candidate k and the metrics visible to its policy."""

    k: int
    result: SolverResult
    metrics_by_split: dict[str, dict[str, object]]


@dataclass(frozen=True)
class KSelectionDecision:
    """Auditable output of a k-selection policy."""

    policy: str
    selected_k: int
    candidate_scores: dict[str, float]
    rationale: str

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "selected_k": self.selected_k,
            "candidate_scores": self.candidate_scores,
            "rationale": self.rationale,
        }


class KSelectionPolicy(Protocol):
    name: str

    def choose(
        self, candidates: list[KCandidate], config: dict[str, object]
    ) -> KSelectionDecision:
        """Choose one candidate without accessing data outside the supplied metrics."""


class BestMetricKPolicy:
    """Choose the highest configured metric, breaking ties toward smaller k."""

    name = "best_metric"

    def choose(
        self, candidates: list[KCandidate], config: dict[str, object]
    ) -> KSelectionDecision:
        if not candidates:
            raise KSelectionError("cannot select k from an empty candidate list")
        selection_split = str(config.get("selection_split", "validation"))
        selection_metric = config.get("selection_metric")
        if selection_metric is None:
            utility_metric = str(config.get("utility_metric", "bedroc"))
            if utility_metric == "bedroc":
                alpha = float(config.get("bedroc_alpha", 20.0))
                selection_metric = f"bedroc_alpha_{alpha:g}"
            else:
                selection_metric = utility_metric
        selection_metric = str(selection_metric)
        scores: dict[str, float] = {}
        for candidate in candidates:
            split_metrics = candidate.metrics_by_split.get(selection_split)
            if split_metrics is None:
                raise KSelectionError(
                    f"selection split is absent for k={candidate.k}: {selection_split}"
                )
            metrics = split_metrics.get("all_metrics", split_metrics)
            if not isinstance(metrics, dict) or selection_metric not in metrics:
                raise KSelectionError(
                    f"selection metric is absent for k={candidate.k}: {selection_metric}"
                )
            value = float(metrics[selection_metric])
            if not math.isfinite(value):
                raise KSelectionError(
                    f"selection metric is nonfinite for k={candidate.k}: {selection_metric}"
                )
            scores[str(candidate.k)] = value

        selected = max(
            candidates,
            key=lambda candidate: (scores[str(candidate.k)], -candidate.k),
        )
        return KSelectionDecision(
            policy=self.name,
            selected_k=selected.k,
            candidate_scores=scores,
            rationale=(
                f"maximized {selection_metric} on {selection_split}; "
                "ties selected the smaller k"
            ),
        )


_POLICIES: dict[str, KSelectionPolicy] = {
    BestMetricKPolicy.name: BestMetricKPolicy(),
}


def register_k_selection_policy(name: str, policy: KSelectionPolicy) -> None:
    """Register or replace a named experiment-specific k-selection policy."""
    normalized = str(name).strip()
    if not normalized:
        raise KSelectionError("k-selection policy name must not be empty")
    _POLICIES[normalized] = policy


def choose_k(
    candidates: list[KCandidate], policy_config: dict[str, object]
) -> KSelectionDecision:
    """Dispatch k selection through the configured policy registry."""
    selector = str(policy_config.get("selector", BestMetricKPolicy.name))
    policy = _POLICIES.get(selector)
    if policy is None:
        raise KSelectionError(f"no k-selection policy is registered: {selector}")
    decision = policy.choose(candidates, policy_config)
    candidate_ks = {candidate.k for candidate in candidates}
    if decision.selected_k not in candidate_ks:
        raise KSelectionError(
            f"k-selection policy chose an unavailable candidate: {decision.selected_k}"
        )
    return decision
