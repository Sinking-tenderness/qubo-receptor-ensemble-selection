"""Auditable OOF selection of receptor/conformation cardinality."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping, Sequence

from .screening import bedroc, enrichment_factor, roc_auc_pairwise


SUPPORTED_ADAPTIVE_METRICS = {"roc_auc", "bedroc", "ef5"}
SUPPORTED_ADAPTIVE_AGGREGATIONS = {"min_score", "mean_score"}
ProgressCallback = Callable[[str, Mapping[str, object]], None]


class AdaptiveCardinalityError(ValueError):
    """Raised when adaptive-cardinality evidence is invalid."""


@dataclass(frozen=True)
class MarginalObservation:
    """One paired marginal observation for a scaffold group."""

    scaffold_id: str
    paired_bedroc_gain: float
    active_rescue_top1: float
    active_rescue_top5: float
    decoy_rescue_top1: float
    decoy_rescue_top5: float


@dataclass(frozen=True)
class TransitionEvidence:
    """Evidence for one adjacent cardinality transition."""

    from_k: int
    to_k: int
    observations: tuple[MarginalObservation, ...]
    bootstrap_samples: tuple[float, ...] | None = None
    mean_rescue_contrast_override: float | None = None
    mean_gain_override: float | None = None
    utility_metric: str = "bedroc"


@dataclass(frozen=True)
class AdaptiveCardinalityDecision:
    """Auditable output of the risk-adjusted OOF cardinality policy."""

    policy: str
    selected_k: int
    need_multi_conformation: bool
    transitions: tuple[dict[str, object], ...]
    uses_outer_labels: bool = False
    metric: str = "bedroc"
    aggregation: str = "mean_score"
    evaluated_candidates: tuple[int, ...] = ()
    stop_reason: str = "candidate_limit"

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "metric": self.metric,
            "aggregation": self.aggregation,
            "selected_k": self.selected_k,
            "need_multi_conformation": self.need_multi_conformation,
            "transitions": [dict(item) for item in self.transitions],
            "uses_outer_labels": self.uses_outer_labels,
            "evaluated_candidates": list(self.evaluated_candidates),
            "stop_reason": self.stop_reason,
        }


def _finite(value: float, name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise AdaptiveCardinalityError(f"{name} must be numeric") from exc
    if not math.isfinite(normalized):
        raise AdaptiveCardinalityError(f"{name} must be finite")
    return normalized


def _validate_utility_metric(value: object) -> str:
    metric = str(value or "bedroc").strip().lower()
    if metric not in SUPPORTED_ADAPTIVE_METRICS:
        raise AdaptiveCardinalityError(
            "utility_metric must be roc_auc, bedroc, or ef5"
        )
    return metric


def _metric_gain_key(utility_metric: str) -> str:
    return f"mean_paired_{utility_metric}_gain"


def _validate_aggregation(value: object) -> str:
    aggregation = str(value or "mean_score").strip().lower()
    if aggregation not in SUPPORTED_ADAPTIVE_AGGREGATIONS:
        raise AdaptiveCardinalityError(
            "aggregation must be min_score or mean_score"
        )
    return aggregation


def _validate_observation(observation: MarginalObservation) -> None:
    if not str(observation.scaffold_id).strip():
        raise AdaptiveCardinalityError("scaffold_id must be non-empty")
    _finite(observation.paired_bedroc_gain, "paired_bedroc_gain")
    for name in (
        "active_rescue_top1",
        "active_rescue_top5",
        "decoy_rescue_top1",
        "decoy_rescue_top5",
    ):
        _finite(getattr(observation, name), name)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise AdaptiveCardinalityError("bootstrap produced no samples")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _bootstrap_transition(
    observations: Iterable[MarginalObservation],
    *,
    bootstrap_iterations: int,
    lower_quantile: float,
    random_seed: int,
    bootstrap_samples: Iterable[float] | None = None,
    mean_gain_override: float | None = None,
    mean_rescue_contrast_override: float | None = None,
    utility_metric: str = "bedroc",
) -> dict[str, float | int]:
    utility_metric = _validate_utility_metric(utility_metric)
    rows = tuple(observations)
    if not rows:
        raise AdaptiveCardinalityError("transition requires at least one observation")
    for observation in rows:
        _validate_observation(observation)

    if bootstrap_samples is not None:
        samples = [_finite(value, "bootstrap sample") for value in bootstrap_samples]
        if not samples:
            raise AdaptiveCardinalityError("bootstrap produced no samples")
        mean_gain = (
            _finite(mean_gain_override, "mean_paired_gain")
            if mean_gain_override is not None
            else sum(samples) / len(samples)
        )
        mean_rescue = (
            _finite(mean_rescue_contrast_override, "mean_rescue_contrast")
            if mean_rescue_contrast_override is not None
            else sum(
                (
                    float(row.active_rescue_top1) - float(row.decoy_rescue_top1)
                    + float(row.active_rescue_top5)
                    - float(row.decoy_rescue_top5)
                )
                / 2.0
                for row in rows
            )
            / len(rows)
        )
        return {
            "observation_count": len(rows),
            "scaffold_count": len({str(row.scaffold_id) for row in rows}),
            _metric_gain_key(utility_metric): mean_gain,
            "bootstrap_lcb": _quantile(samples, lower_quantile),
            "bootstrap_positive_probability": sum(value > 0 for value in samples)
            / len(samples),
            "mean_rescue_contrast": mean_rescue,
        }

    grouped: dict[str, list[MarginalObservation]] = {}
    for observation in rows:
        grouped.setdefault(str(observation.scaffold_id), []).append(observation)
    scaffold_means: dict[str, tuple[float, float]] = {}
    for scaffold_id, group in grouped.items():
        scaffold_means[scaffold_id] = (
            sum(float(row.paired_bedroc_gain) for row in group) / len(group),
            sum(
                (
                    float(row.active_rescue_top1) - float(row.decoy_rescue_top1)
                    + float(row.active_rescue_top5)
                    - float(row.decoy_rescue_top5)
                )
                / 2.0
                for row in group
            )
            / len(group),
        )

    scaffold_ids = tuple(sorted(scaffold_means))
    rng = random.Random(random_seed)
    bootstrap_gains: list[float] = []
    for _ in range(bootstrap_iterations):
        sampled = [rng.choice(scaffold_ids) for _ in scaffold_ids]
        bootstrap_gains.append(
            sum(scaffold_means[scaffold_id][0] for scaffold_id in sampled)
            / len(sampled)
        )

    mean_gain = sum(scaffold_means[scaffold_id][0] for scaffold_id in scaffold_ids) / len(
        scaffold_ids
    )
    mean_rescue = sum(scaffold_means[scaffold_id][1] for scaffold_id in scaffold_ids) / len(
        scaffold_ids
    )
    return {
        "observation_count": len(rows),
        "scaffold_count": len(scaffold_ids),
        _metric_gain_key(utility_metric): mean_gain,
        "bootstrap_lcb": _quantile(bootstrap_gains, lower_quantile),
        "bootstrap_positive_probability": sum(value > 0 for value in bootstrap_gains)
        / len(bootstrap_gains),
        "mean_rescue_contrast": mean_rescue,
    }


def _validate_rows(
    rows: Sequence[Mapping[str, object]], receptor_ids: Sequence[str], scaffold_field: str
) -> list[dict[str, object]]:
    if not rows:
        raise AdaptiveCardinalityError("adaptive cardinality requires non-empty rows")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    receptor_names = tuple(str(value) for value in receptor_ids)
    if not receptor_names or len(set(receptor_names)) != len(receptor_names):
        raise AdaptiveCardinalityError("receptor_ids must be non-empty and unique")
    for source in rows:
        row = {str(key): value for key, value in source.items()}
        ligand_id = str(row.get("ligand_id", ""))
        label = str(row.get("label", ""))
        scaffold = str(row.get(scaffold_field, ""))
        if not ligand_id or ligand_id in seen:
            raise AdaptiveCardinalityError("ligand_id values must be non-empty and unique")
        if label not in {"active", "decoy"}:
            raise AdaptiveCardinalityError("label must be active or decoy")
        if not scaffold:
            raise AdaptiveCardinalityError(f"{scaffold_field} must be non-empty")
        for receptor_id in receptor_names:
            if receptor_id not in row:
                raise AdaptiveCardinalityError(f"missing receptor score: {receptor_id}")
            _finite(row[receptor_id], f"score for {receptor_id}")
        seen.add(ligand_id)
        normalized.append(row)
    if {str(row["label"]) for row in normalized} != {"active", "decoy"}:
        raise AdaptiveCardinalityError("adaptive rows must contain active and decoy labels")
    return normalized


def _inner_fold_assignments(
    rows: Sequence[Mapping[str, object]], scaffold_field: str, fold_count: int
) -> dict[str, int]:
    if isinstance(fold_count, bool) or fold_count < 2:
        raise AdaptiveCardinalityError("inner_fold_count must be at least two")
    scaffolds = sorted({str(row[scaffold_field]) for row in rows})
    if len(scaffolds) < fold_count:
        raise AdaptiveCardinalityError("inner_fold_count exceeds scaffold group count")
    return {scaffold: index % fold_count for index, scaffold in enumerate(scaffolds)}


def _score_records(
    rows: Sequence[Mapping[str, object]],
    subset: Sequence[str],
    aggregation: str = "min_score",
) -> dict[str, dict[str, object]]:
    aggregation = _validate_aggregation(aggregation)
    if not subset or len(set(subset)) != len(subset):
        raise AdaptiveCardinalityError("solver returned an invalid receptor subset")
    records: dict[str, dict[str, object]] = {}
    for row in rows:
        ligand_id = str(row["ligand_id"])
        scores = [float(row[receptor_id]) for receptor_id in subset]
        records[ligand_id] = {
            "score": min(scores)
            if aggregation == "min_score"
            else sum(scores) / len(scores),
            "label": str(row["label"]),
        }
    return records


def _sampled_metric(
    records: Mapping[str, Mapping[str, object]],
    grouped_ids: Mapping[str, Sequence[str]],
    sampled_groups: Sequence[str],
    *,
    utility_metric: str,
    alpha: float,
) -> float:
    ranked_values: list[tuple[float, int, str, int]] = []
    for draw_index, group_id in enumerate(sampled_groups):
        for ligand_id in grouped_ids[group_id]:
            record = records[ligand_id]
            ranked_values.append(
                (
                    float(record["score"]),
                    draw_index,
                    ligand_id,
                    int(str(record["label"]) == "active"),
                )
            )
    ranked_values.sort(key=lambda value: (value[0], value[1], value[2]))
    labels = [binary_label for _, _, _, binary_label in ranked_values]
    if utility_metric == "roc_auc":
        return float(
            roc_auc_pairwise(labels, [-score for score, _, _, _ in ranked_values])
        )
    ranked = [{"binary_label": binary_label} for binary_label in labels]
    if utility_metric == "bedroc":
        return float(bedroc(ranked, alpha))
    return float(enrichment_factor(ranked, 0.05)["ef"])


def _shared_candidate_bootstrap(
    records_by_candidate: Mapping[int, Mapping[str, Mapping[str, object]]],
    group_by_ligand: Mapping[str, str],
    *,
    replicates: int,
    seed: int,
    utility_metric: str = "bedroc",
    alpha: float,
) -> tuple[dict[int, tuple[float, ...]], dict[int, float]]:
    """Bootstrap absolute candidate metrics using one shared draw stream."""

    utility_metric = _validate_utility_metric(utility_metric)
    if replicates <= 0:
        raise AdaptiveCardinalityError("bootstrap_iterations must be positive")
    candidate_ids = tuple(sorted(records_by_candidate))
    if not candidate_ids:
        raise AdaptiveCardinalityError("candidate bootstrap requires candidates")
    ligand_ids = set(records_by_candidate[candidate_ids[0]])
    if any(set(records_by_candidate[candidate]) != ligand_ids for candidate in candidate_ids):
        raise AdaptiveCardinalityError("bootstrap candidates contain different ligand IDs")
    if set(group_by_ligand) != ligand_ids:
        raise AdaptiveCardinalityError("bootstrap group map differs from score records")
    grouped_ids: dict[str, list[str]] = {}
    for ligand_id, group_id in group_by_ligand.items():
        grouped_ids.setdefault(str(group_id), []).append(ligand_id)
    for group in grouped_ids.values():
        group.sort()
    group_ids = sorted(grouped_ids)
    samples: dict[int, list[float]] = {candidate: [] for candidate in candidate_ids}

    rng = random.Random(seed)
    attempts = 0
    while len(next(iter(samples.values()))) < replicates:
        attempts += 1
        if attempts > replicates * 2:
            raise AdaptiveCardinalityError(
                "too many bootstrap samples lacked both labels"
            )
        sampled_groups = rng.choices(group_ids, k=len(group_ids))
        values = {
            candidate: _sampled_metric(
                records_by_candidate[candidate],
                grouped_ids,
                sampled_groups,
                utility_metric=utility_metric,
                alpha=alpha,
            )
            for candidate in candidate_ids
        }
        if not all(math.isfinite(value) for value in values.values()):
            continue
        for candidate in candidate_ids:
            samples[candidate].append(values[candidate])
    mean_metrics = {
        candidate: _sampled_metric(
            records_by_candidate[candidate],
            grouped_ids,
            group_ids,
            utility_metric=utility_metric,
            alpha=alpha,
        )
        for candidate in candidate_ids
    }
    return (
        {candidate: tuple(values) for candidate, values in samples.items()},
        mean_metrics,
    )


def _rank_rescue_contrast(
    previous: Mapping[str, Mapping[str, object]],
    current: Mapping[str, Mapping[str, object]],
    fractions: Sequence[float],
) -> float:
    if set(previous) != set(current):
        raise AdaptiveCardinalityError("rescue methods contain different ligand IDs")
    ranks: dict[str, dict[str, int]] = {}
    for method, records in (("previous", previous), ("current", current)):
        ordered = sorted(
            records.items(), key=lambda item: (float(item[1]["score"]), item[0])
        )
        ranks[method] = {ligand_id: index for index, (ligand_id, _) in enumerate(ordered, 1)}
    contrasts: list[float] = []
    for fraction in fractions:
        if not 0.0 < float(fraction) <= 1.0:
            raise AdaptiveCardinalityError("rescue fractions must be in (0, 1]")
        cutoff = max(1, math.ceil(len(current) * float(fraction)))
        class_fractions: dict[str, float] = {}
        for label in ("active", "decoy"):
            members = [
                ligand_id
                for ligand_id, row in current.items()
                if str(row["label"]) == label
            ]
            rescued = sum(
                ranks["previous"][ligand_id] > cutoff
                and ranks["current"][ligand_id] <= cutoff
                for ligand_id in members
            )
            class_fractions[label] = rescued / len(members) if members else math.nan
        if not all(math.isfinite(value) for value in class_fractions.values()):
            raise AdaptiveCardinalityError("rescue contrast requires both labels")
        contrasts.append(class_fractions["active"] - class_fractions["decoy"])
    return sum(contrasts) / len(contrasts)


def estimate_adaptive_cardinality(
    rows: Sequence[Mapping[str, object]],
    receptor_ids: Sequence[str],
    *,
    problem_config: Mapping[str, object] | None = None,
    solve_subset: Callable[[list[dict[str, object]], int], Sequence[str]] | None = None,
    solver_backend: str = "exact",
    candidate_ks: Sequence[int] | None = None,
    scaffold_field: str = "scaffold_smiles",
    inner_fold_count: int = 3,
    bootstrap_iterations: int = 1000,
    lower_quantile: float = 0.05,
    minimum_effect: float = 0.0,
    required_probability: float = 0.5,
    cost_per_receptor: float = 0.0,
    selection_tie_tolerance: float = 0.0,
    require_rescue_contrast: bool = False,
    rescue_fractions: Sequence[float] = (0.01, 0.05),
    bedroc_alpha: float = 20.0,
    random_seed: int = 0,
    progress: ProgressCallback | None = None,
    aggregation: str = "mean_score",
) -> AdaptiveCardinalityDecision:
    """Estimate cardinality from inner-fold predictions without outer labels."""

    normalized = _validate_rows(rows, receptor_ids, scaffold_field)
    if candidate_ks is not None and any(
        isinstance(value, bool) or not isinstance(value, int) for value in candidate_ks
    ):
        raise AdaptiveCardinalityError("candidate_ks must contain integers")
    candidates = (
        tuple(range(1, len(receptor_ids) + 1))
        if candidate_ks is None
        else tuple(candidate_ks)
    )
    if (
        not candidates
        or candidates[0] != 1
        or tuple(sorted(set(candidates))) != candidates
        or candidates != tuple(range(1, candidates[-1] + 1))
    ):
        raise AdaptiveCardinalityError(
            "candidate_ks must be consecutive, unique, and start at 1"
        )
    if any(value <= 0 for value in candidates):
        raise AdaptiveCardinalityError("candidate_ks must contain positive integers")
    if any(value > len(receptor_ids) for value in candidates):
        raise AdaptiveCardinalityError("candidate_ks must fit the receptor pool")
    if problem_config is None:
        problem_config = {
            "type": "receptor_subset",
            "strategy": "qubo",
            "weights": {"redundancy": 0.25, "count": 0.1, "size": 1.0},
            "utility_metric": "bedroc",
            "bedroc_alpha": bedroc_alpha,
        }
    base_problem_config = dict(problem_config)
    utility_metric = _validate_utility_metric(
        base_problem_config.get("utility_metric", "bedroc")
    )
    aggregation = _validate_aggregation(aggregation)
    receptor_names = tuple(str(value) for value in receptor_ids)
    if progress is not None:
        progress(
            "adaptive_started",
            {
                "metric": utility_metric,
                "aggregation": aggregation,
                "candidates": list(candidates),
            },
        )

    if solve_subset is None:
        def solve_subset(train_rows: list[dict[str, object]], k: int) -> Sequence[str]:
            from .solvers import build_problem, solve_problem

            candidate_config = dict(base_problem_config)
            candidate_config.pop("k_policy", None)
            candidate_config["receptor_ids"] = list(receptor_names)
            candidate_config["target_size"] = k
            problem = build_problem(train_rows, candidate_config)
            return solve_problem(problem, solver_backend).subset

    assignments = _inner_fold_assignments(normalized, scaffold_field, inner_fold_count)
    folds: list[tuple[list[dict[str, object]], list[dict[str, object]]]] = []
    for inner_fold in range(inner_fold_count):
        if progress is not None:
            progress(
                "inner_fold_started",
                {"fold": inner_fold + 1, "fold_count": inner_fold_count},
            )
        train_rows = [
            row
            for row in normalized
            if assignments[str(row[scaffold_field])] != inner_fold
        ]
        validation_rows = [
            row
            for row in normalized
            if assignments[str(row[scaffold_field])] == inner_fold
        ]
        if not train_rows or not validation_rows:
            raise AdaptiveCardinalityError("inner fold is empty")
        folds.append((train_rows, validation_rows))

    group_by_ligand = {
        str(row["ligand_id"]): str(row[scaffold_field]) for row in normalized
    }
    previous_k = candidates[0]
    previous_records: dict[str, dict[str, object]] = {}
    for fold_index, (train_rows, validation_rows) in enumerate(folds):
        subset = tuple(str(value) for value in solve_subset(train_rows, previous_k))
        previous_records.update(_score_records(validation_rows, subset, aggregation))
        if progress is not None:
            progress(
                "candidate_completed",
                {
                    "fold": fold_index + 1,
                    "fold_count": inner_fold_count,
                    "candidate_k": previous_k,
                },
            )

    transitions: list[TransitionEvidence] = []
    evaluated_candidates = [previous_k]
    uncertain_confirmation_used = False
    stop_reason = "candidate_limit"
    for current_k in candidates[1:]:
        current_records: dict[str, dict[str, object]] = {}
        for fold_index, (train_rows, validation_rows) in enumerate(folds):
            subset = tuple(str(value) for value in solve_subset(train_rows, current_k))
            current_records.update(
                _score_records(validation_rows, subset, aggregation)
            )
            if progress is not None:
                progress(
                    "candidate_completed",
                    {
                        "fold": fold_index + 1,
                        "fold_count": inner_fold_count,
                        "candidate_k": current_k,
                    },
                )

        candidate_samples, candidate_mean_metrics = _shared_candidate_bootstrap(
            {previous_k: previous_records, current_k: current_records},
            group_by_ligand,
            replicates=bootstrap_iterations,
            seed=random_seed + previous_k,
            utility_metric=utility_metric,
            alpha=bedroc_alpha,
        )
        samples = tuple(
            current_value - previous_value
            for current_value, previous_value in zip(
                candidate_samples[current_k], candidate_samples[previous_k]
            )
        )
        rescue = _rank_rescue_contrast(
            previous_records, current_records, rescue_fractions
        )
        observations = tuple(
            _observation_from_group(
                scaffold_id,
                normalized,
                group_by_ligand,
                samples,
                rescue,
            )
            for scaffold_id in sorted(set(group_by_ligand.values()))
        )
        transitions.append(
            TransitionEvidence(
                from_k=previous_k,
                to_k=current_k,
                observations=observations,
                bootstrap_samples=samples,
                mean_gain_override=(
                    candidate_mean_metrics[current_k]
                    - candidate_mean_metrics[previous_k]
                ),
                mean_rescue_contrast_override=rescue,
                utility_metric=utility_metric,
            )
        )
        evaluated_candidates.append(current_k)
        interim_decision = select_adaptive_k(
            transitions,
            bootstrap_iterations=bootstrap_iterations,
            lower_quantile=lower_quantile,
            minimum_effect=minimum_effect,
            required_probability=required_probability,
            cost_per_receptor=cost_per_receptor,
            selection_tie_tolerance=selection_tie_tolerance,
            require_rescue_contrast=require_rescue_contrast,
            random_seed=random_seed,
            utility_metric=utility_metric,
            aggregation=aggregation,
        )
        marginal_state = str(interim_decision.transitions[-1]["marginal_state"])
        if marginal_state == "harmful":
            stop_reason = "harmful_transition"
            break
        if marginal_state == "uncertain":
            if uncertain_confirmation_used:
                stop_reason = "uncertain_confirmation"
                break
            uncertain_confirmation_used = True
        elif uncertain_confirmation_used:
            stop_reason = "uncertain_confirmation"
            break
        previous_k = current_k
        previous_records = current_records

    decision = select_adaptive_k(
        transitions,
        bootstrap_iterations=bootstrap_iterations,
        lower_quantile=lower_quantile,
        minimum_effect=minimum_effect,
        required_probability=required_probability,
        cost_per_receptor=cost_per_receptor,
        selection_tie_tolerance=selection_tie_tolerance,
        require_rescue_contrast=require_rescue_contrast,
        random_seed=random_seed,
        utility_metric=utility_metric,
        aggregation=aggregation,
        progress=progress,
    )
    decision = replace(
        decision,
        evaluated_candidates=tuple(evaluated_candidates),
        stop_reason=stop_reason,
    )
    if progress is not None:
        progress(
            "adaptive_completed",
            {
                "metric": utility_metric,
                "aggregation": aggregation,
                "selected_k": decision.selected_k,
                "evaluated_candidates": list(decision.evaluated_candidates),
                "stop_reason": decision.stop_reason,
            },
        )
    return decision


def _observation_from_group(
    scaffold_id: str,
    rows: Sequence[Mapping[str, object]],
    group_by_ligand: Mapping[str, str],
    bootstrap_samples: Sequence[float],
    rescue: float,
) -> MarginalObservation:
    del rows, group_by_ligand, bootstrap_samples
    return MarginalObservation(
        scaffold_id=scaffold_id,
        paired_bedroc_gain=0.0,
        active_rescue_top1=rescue,
        active_rescue_top5=rescue,
        decoy_rescue_top1=0.0,
        decoy_rescue_top5=0.0,
    )


def select_adaptive_k(
    transitions: Iterable[TransitionEvidence],
    *,
    bootstrap_iterations: int = 1000,
    lower_quantile: float = 0.05,
    minimum_effect: float = 0.0,
    required_probability: float = 0.5,
    cost_per_receptor: float = 0.0,
    selection_tie_tolerance: float = 0.0,
    require_rescue_contrast: bool = False,
    random_seed: int = 0,
    utility_metric: str | None = None,
    aggregation: str = "mean_score",
    progress: ProgressCallback | None = None,
) -> AdaptiveCardinalityDecision:
    """Select cardinality from adjacent gains with a supported-path gate.

    Each transition is judged against the immediately preceding cardinality.
    Candidate utility is accumulated along the transition path, but a
    non-supported transition permanently blocks larger candidates.
    The bootstrap lower confidence bound remains an audit statistic; the
    configured probability and practical-effect rules control support.
    """

    if isinstance(bootstrap_iterations, bool) or bootstrap_iterations <= 0:
        raise AdaptiveCardinalityError("bootstrap_iterations must be positive")
    if not 0.0 <= lower_quantile <= 1.0:
        raise AdaptiveCardinalityError("lower_quantile must be between 0 and 1")
    minimum_effect = _finite(minimum_effect, "minimum_effect")
    required_probability = _finite(required_probability, "required_probability")
    if not 0.0 <= required_probability <= 1.0:
        raise AdaptiveCardinalityError("required_probability must be between 0 and 1")
    cost_per_receptor = _finite(cost_per_receptor, "cost_per_receptor")
    if cost_per_receptor < 0.0:
        raise AdaptiveCardinalityError("cost_per_receptor must be non-negative")
    selection_tie_tolerance = _finite(
        selection_tie_tolerance, "selection_tie_tolerance"
    )
    if selection_tie_tolerance < 0.0:
        raise AdaptiveCardinalityError(
            "selection_tie_tolerance must be non-negative"
        )
    if not isinstance(require_rescue_contrast, bool):
        raise AdaptiveCardinalityError("require_rescue_contrast must be boolean")

    ordered = sorted(
        list(transitions), key=lambda transition: (transition.to_k, transition.from_k)
    )
    transition_metrics = {
        _validate_utility_metric(transition.utility_metric) for transition in ordered
    }
    if utility_metric is None:
        utility_metric = next(iter(transition_metrics), "bedroc")
    utility_metric = _validate_utility_metric(utility_metric)
    aggregation = _validate_aggregation(aggregation)
    if transition_metrics and transition_metrics != {utility_metric}:
        raise AdaptiveCardinalityError(
            "all transitions must use the configured utility_metric"
        )
    selected_k = 1
    selected_score = 0.0
    seen_pairs: set[tuple[int, int]] = set()
    diagnostics: list[dict[str, object]] = []
    cumulative_samples_by_k: dict[int, tuple[float, ...]] = {1: tuple()}
    cumulative_mean_by_k: dict[int, float] = {1: 0.0}
    path_blocked = False
    for transition in ordered:
        pair = (transition.from_k, transition.to_k)
        if transition.from_k < 1 or transition.to_k <= transition.from_k:
            raise AdaptiveCardinalityError(
                "transitions must use positive increasing from_k and to_k"
            )
        if pair in seen_pairs:
            raise AdaptiveCardinalityError("transitions must be unique")
        seen_pairs.add(pair)
        stats = _bootstrap_transition(
            transition.observations,
            bootstrap_iterations=bootstrap_iterations,
            lower_quantile=lower_quantile,
            random_seed=random_seed + transition.from_k,
            bootstrap_samples=transition.bootstrap_samples,
            mean_gain_override=transition.mean_gain_override,
            mean_rescue_contrast_override=transition.mean_rescue_contrast_override,
            utility_metric=utility_metric,
        )
        metric_key = _metric_gain_key(utility_metric)
        mean_gain = float(stats[metric_key])
        candidate_cost = cost_per_receptor * (transition.to_k - transition.from_k)
        risk_adjusted_gain = mean_gain - candidate_cost
        if transition.bootstrap_samples is not None:
            net_samples = tuple(
                _finite(value, "bootstrap sample") - candidate_cost
                for value in transition.bootstrap_samples
            )
        else:
            net_samples = tuple()
        if transition.bootstrap_samples is not None:
            risk_positive_probability = sum(
                value > minimum_effect for value in net_samples
            ) / len(net_samples)
        else:
            risk_positive_probability = float(risk_adjusted_gain > minimum_effect)
        risk_negative_probability = (
            sum(value < -minimum_effect for value in net_samples) / len(net_samples)
            if net_samples
            else float(risk_adjusted_gain < -minimum_effect)
        )
        rescue_supported = float(stats["mean_rescue_contrast"]) > 0.0
        marginal_passed = bool(
            risk_adjusted_gain > minimum_effect
            and risk_positive_probability >= required_probability
            and (not require_rescue_contrast or rescue_supported)
        )
        if marginal_passed:
            marginal_state = "supported"
        elif risk_negative_probability >= required_probability:
            marginal_state = "harmful"
        else:
            marginal_state = "uncertain"

        path_available = (
            not path_blocked and transition.from_k in cumulative_mean_by_k
        )
        cumulative_samples: tuple[float, ...] = tuple()
        cumulative_mean = math.nan
        cumulative_positive_probability = 0.0
        cumulative_passed = False
        cumulative_lcb = math.nan
        cumulative_minimum_effect = minimum_effect * (transition.to_k - 1)
        if path_available:
            previous_samples = cumulative_samples_by_k[transition.from_k]
            previous_mean = cumulative_mean_by_k[transition.from_k]
            if previous_samples and len(previous_samples) != len(net_samples):
                raise AdaptiveCardinalityError(
                    "adjacent transitions must use the same bootstrap sample count"
                )
            if not previous_samples:
                cumulative_samples = net_samples
            elif not net_samples:
                cumulative_samples = previous_samples
            else:
                cumulative_samples = tuple(
                    previous_value + current_value
                    for previous_value, current_value in zip(
                        previous_samples, net_samples
                    )
                )
            cumulative_mean = previous_mean + risk_adjusted_gain
            cumulative_positive_probability = (
                sum(value > cumulative_minimum_effect for value in cumulative_samples)
                / len(cumulative_samples)
                if cumulative_samples
                else float(cumulative_mean > cumulative_minimum_effect)
            )
            cumulative_lcb = (
                _quantile(list(cumulative_samples), lower_quantile)
                if cumulative_samples
                else cumulative_mean
            )
            cumulative_passed = bool(
                cumulative_mean > cumulative_minimum_effect
                and cumulative_positive_probability >= required_probability
                and (not require_rescue_contrast or rescue_supported)
            )
            cumulative_samples_by_k[transition.to_k] = cumulative_samples
            cumulative_mean_by_k[transition.to_k] = cumulative_mean

        candidate_score = cumulative_mean
        candidate_passed = bool(
            path_available and marginal_passed and cumulative_passed
        )
        diagnostics.append(
            {
                "transition": f"{transition.from_k}->{transition.to_k}",
                "from_k": transition.from_k,
                "to_k": transition.to_k,
                "metric": utility_metric,
                **stats,
                "risk_adjusted_gain": risk_adjusted_gain,
                "risk_adjusted_lcb": _quantile(list(net_samples), lower_quantile)
                if net_samples
                else risk_adjusted_gain,
                "risk_positive_probability": risk_positive_probability,
                "risk_negative_probability": risk_negative_probability,
                "cost_per_receptor": cost_per_receptor,
                "bootstrap_iterations": bootstrap_iterations,
                "lower_quantile": lower_quantile,
                "minimum_effect": minimum_effect,
                "required_probability": required_probability,
                "selection_tie_tolerance": selection_tie_tolerance,
                "require_rescue_contrast": require_rescue_contrast,
                "eligible_for_selection": path_available and marginal_passed,
                "marginal_state": marginal_state,
                "passed": marginal_passed,
                "candidate_passed": candidate_passed,
                "cumulative_risk_adjusted_gain": candidate_score,
                "cumulative_minimum_effect": cumulative_minimum_effect,
                "cumulative_bootstrap_lcb": cumulative_lcb,
                "cumulative_positive_probability": cumulative_positive_probability,
            }
        )
        if progress is not None:
            progress(
                "transition_evaluated",
                {
                    "from_k": transition.from_k,
                    "to_k": transition.to_k,
                    "metric": utility_metric,
                    "aggregation": aggregation,
                    "passed": marginal_passed,
                    "marginal_state": marginal_state,
                    "candidate_passed": candidate_passed,
                },
            )
        if candidate_passed:
            score = candidate_score
            if score > selected_score + selection_tie_tolerance or (
                abs(score - selected_score) <= selection_tie_tolerance
                and transition.to_k < selected_k
            ):
                selected_k = transition.to_k
                selected_score = score
        if not marginal_passed:
            path_blocked = True

    return AdaptiveCardinalityDecision(
        policy="risk_adjusted_oof",
        metric=utility_metric,
        aggregation=aggregation,
        selected_k=selected_k,
        need_multi_conformation=selected_k > 1,
        transitions=tuple(diagnostics),
    )
