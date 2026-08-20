"""Auditable sequential selection of receptor/conformation cardinality."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from .screening import bedroc


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
    """Evidence for one sequential ``k-1 -> k`` transition."""

    from_k: int
    to_k: int
    observations: tuple[MarginalObservation, ...]
    bootstrap_samples: tuple[float, ...] | None = None
    mean_rescue_contrast_override: float | None = None


@dataclass(frozen=True)
class AdaptiveCardinalityDecision:
    """Auditable output of the bootstrap-LCB cardinality policy."""

    policy: str
    selected_k: int
    need_multi_conformation: bool
    transitions: tuple[dict[str, object], ...]
    uses_outer_labels: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "selected_k": self.selected_k,
            "need_multi_conformation": self.need_multi_conformation,
            "transitions": [dict(item) for item in self.transitions],
            "uses_outer_labels": self.uses_outer_labels,
        }


def _finite(value: float, name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise AdaptiveCardinalityError(f"{name} must be numeric") from exc
    if not math.isfinite(normalized):
        raise AdaptiveCardinalityError(f"{name} must be finite")
    return normalized


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
    mean_rescue_contrast_override: float | None = None,
) -> dict[str, float | int]:
    rows = tuple(observations)
    if not rows:
        raise AdaptiveCardinalityError("transition requires at least one observation")
    for observation in rows:
        _validate_observation(observation)

    if bootstrap_samples is not None:
        samples = [_finite(value, "bootstrap sample") for value in bootstrap_samples]
        if not samples:
            raise AdaptiveCardinalityError("bootstrap produced no samples")
        mean_gain = sum(samples) / len(samples)
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
            "mean_paired_bedroc_gain": mean_gain,
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
        "mean_paired_bedroc_gain": mean_gain,
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
    rows: Sequence[Mapping[str, object]], subset: Sequence[str]
) -> dict[str, dict[str, object]]:
    if not subset or len(set(subset)) != len(subset):
        raise AdaptiveCardinalityError("solver returned an invalid receptor subset")
    records: dict[str, dict[str, object]] = {}
    for row in rows:
        ligand_id = str(row["ligand_id"])
        records[ligand_id] = {
            "score": min(float(row[receptor_id]) for receptor_id in subset),
            "label": str(row["label"]),
        }
    return records


def _sampled_bedroc(
    records: Mapping[str, Mapping[str, object]],
    grouped_ids: Mapping[str, Sequence[str]],
    sampled_groups: Sequence[str],
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
    return float(
        bedroc(
            [{"binary_label": binary_label} for _, _, _, binary_label in ranked_values],
            alpha,
        )
    )


def _paired_group_bootstrap(
    records_by_method: Mapping[str, Mapping[str, Mapping[str, object]]],
    group_by_ligand: Mapping[str, str],
    *,
    replicates: int,
    seed: int,
    alpha: float,
) -> tuple[float, ...]:
    if replicates <= 0:
        raise AdaptiveCardinalityError("bootstrap_iterations must be positive")
    method_ids = list(records_by_method)
    if len(method_ids) != 2:
        raise AdaptiveCardinalityError("paired bootstrap requires two methods")
    ligand_ids = set(records_by_method[method_ids[0]])
    if any(set(records_by_method[method]) != ligand_ids for method in method_ids):
        raise AdaptiveCardinalityError("bootstrap methods contain different ligand IDs")
    if set(group_by_ligand) != ligand_ids:
        raise AdaptiveCardinalityError("bootstrap group map differs from score records")
    grouped_ids: dict[str, list[str]] = {}
    for ligand_id, group_id in group_by_ligand.items():
        grouped_ids.setdefault(str(group_id), []).append(ligand_id)
    for group in grouped_ids.values():
        group.sort()
    group_ids = sorted(grouped_ids)
    rng = random.Random(seed)
    samples: list[float] = []
    attempts = 0
    while len(samples) < replicates:
        attempts += 1
        if attempts > replicates * 2:
            raise AdaptiveCardinalityError(
                "too many bootstrap samples lacked both labels"
            )
        sampled_groups = rng.choices(group_ids, k=len(group_ids))
        values = [
            _sampled_bedroc(records_by_method[method_id], grouped_ids, sampled_groups, alpha)
            for method_id in method_ids
        ]
        if not all(math.isfinite(value) for value in values):
            continue
        samples.append(values[1] - values[0])
    return tuple(samples)


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
    candidate_ks: Sequence[int] = (1, 2, 3),
    scaffold_field: str = "scaffold_smiles",
    inner_fold_count: int = 3,
    bootstrap_iterations: int = 1000,
    lower_quantile: float = 0.025,
    rescue_fractions: Sequence[float] = (0.01, 0.05),
    bedroc_alpha: float = 20.0,
    random_seed: int = 0,
) -> AdaptiveCardinalityDecision:
    """Estimate cardinality from inner-fold predictions without outer labels."""

    normalized = _validate_rows(rows, receptor_ids, scaffold_field)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in candidate_ks):
        raise AdaptiveCardinalityError("candidate_ks must contain integers")
    candidates = tuple(candidate_ks)
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
    receptor_names = tuple(str(value) for value in receptor_ids)

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
    records_by_method: dict[str, dict[str, dict[str, object]]] = {
        str(k): {} for k in candidates
    }
    for inner_fold in range(inner_fold_count):
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
        for k in candidates:
            subset = tuple(str(value) for value in solve_subset(train_rows, k))
            records_by_method[str(k)].update(
                _score_records(validation_rows, subset)
            )

    group_by_ligand = {
        str(row["ligand_id"]): str(row[scaffold_field]) for row in normalized
    }
    transitions: list[TransitionEvidence] = []
    for index in range(1, len(candidates)):
        previous_k = candidates[index - 1]
        current_k = candidates[index]
        previous_records = records_by_method[str(previous_k)]
        current_records = records_by_method[str(current_k)]
        samples = _paired_group_bootstrap(
            {"previous": previous_records, "current": current_records},
            group_by_ligand,
            replicates=bootstrap_iterations,
            seed=random_seed + previous_k,
            alpha=bedroc_alpha,
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
                mean_rescue_contrast_override=rescue,
            )
        )
    return select_adaptive_k(
        transitions,
        bootstrap_iterations=bootstrap_iterations,
        lower_quantile=lower_quantile,
        random_seed=random_seed,
    )


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
    lower_quantile: float = 0.025,
    random_seed: int = 0,
) -> AdaptiveCardinalityDecision:
    """Select the smallest cardinality passing sequential functional gates.

    A transition passes only when its scaffold-bootstrap lower confidence bound
    and mean active-minus-decoy rescue contrast are both strictly positive.
    The first failed transition stops the scan, so later evidence cannot rescue
    an earlier failed addition.
    """

    if isinstance(bootstrap_iterations, bool) or bootstrap_iterations <= 0:
        raise AdaptiveCardinalityError("bootstrap_iterations must be positive")
    if not 0.0 <= lower_quantile <= 1.0:
        raise AdaptiveCardinalityError("lower_quantile must be between 0 and 1")

    ordered = list(transitions)
    selected_k = 1
    expected_from = 1
    diagnostics: list[dict[str, object]] = []
    for transition in ordered:
        if transition.from_k != expected_from or transition.to_k <= transition.from_k:
            raise AdaptiveCardinalityError(
                "transitions must start at 1 and increase sequentially"
            )
        stats = _bootstrap_transition(
            transition.observations,
            bootstrap_iterations=bootstrap_iterations,
            lower_quantile=lower_quantile,
            random_seed=random_seed + transition.from_k,
            bootstrap_samples=transition.bootstrap_samples,
            mean_rescue_contrast_override=transition.mean_rescue_contrast_override,
        )
        passed = bool(
            float(stats["bootstrap_lcb"]) > 0.0
            and float(stats["mean_rescue_contrast"]) > 0.0
        )
        diagnostics.append(
            {
                "transition": f"{transition.from_k}->{transition.to_k}",
                "from_k": transition.from_k,
                "to_k": transition.to_k,
                **stats,
                "bootstrap_iterations": bootstrap_iterations,
                "lower_quantile": lower_quantile,
                "passed": passed,
            }
        )
        if not passed:
            break
        selected_k = transition.to_k
        expected_from = transition.to_k

    return AdaptiveCardinalityDecision(
        policy="mechanistic_bootstrap_lcb",
        selected_k=selected_k,
        need_multi_conformation=selected_k > 1,
        transitions=tuple(diagnostics),
    )
