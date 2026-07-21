"""Build a scale-normalized receptor-subset QUBO for exact small-pool solves."""

from __future__ import annotations

import itertools
import math

from scipy.stats import spearmanr

try:
    from .compare_receptor_screening import ranked_metrics_with_ids
    from .solve_discriminative_coverage_qubo import top_label_ids
except ImportError:
    from compare_receptor_screening import ranked_metrics_with_ids
    from solve_discriminative_coverage_qubo import top_label_ids


def minmax_terms(values: dict[str, float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot normalize an empty term map")
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("QUBO term map contains a nonfinite value")
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum == minimum:
        return {key: 0.0 for key in values}
    return {
        key: (float(value) - minimum) / (maximum - minimum)
        for key, value in values.items()
    }


def build_normalized_terms(
    rows: list[dict[str, object]],
    receptor_ids: list[str],
    fraction: float,
    utility_metric: str,
) -> dict[str, object]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("coverage fraction must be in (0, 1]")
    if utility_metric not in {"roc_auc", "bedroc", "ef5"}:
        raise ValueError(f"unsupported utility metric: {utility_metric}")
    active_total = sum(row["label"] == "active" for row in rows)
    decoy_total = sum(row["label"] == "decoy" for row in rows)
    if not active_total or not decoy_total:
        raise ValueError("QUBO terms require both active and decoy rows")

    metric_key = {
        "roc_auc": "roc_auc",
        "bedroc": "bedroc_alpha_20",
        "ef5": "EF5%",
    }[utility_metric]
    utility_raw: dict[str, float] = {}
    score_columns: dict[str, list[float]] = {}
    active_sets: dict[str, set[str]] = {}
    decoy_sets: dict[str, set[str]] = {}
    for receptor_id in receptor_ids:
        data = {
            str(row["ligand_id"]): {
                "label": row["label"],
                "score": float(row[receptor_id]),
            }
            for row in rows
        }
        utility_raw[receptor_id] = float(
            ranked_metrics_with_ids(data)[metric_key]
        )
        score_columns[receptor_id] = [
            float(row[receptor_id]) for row in rows
        ]
        active_sets[receptor_id] = top_label_ids(
            rows, receptor_id, fraction, "active"
        )
        decoy_sets[receptor_id] = top_label_ids(
            rows, receptor_id, fraction, "decoy"
        )

    active_coverage_raw = {
        receptor_id: len(active_sets[receptor_id]) / active_total
        for receptor_id in receptor_ids
    }
    decoy_exposure_raw = {
        receptor_id: len(decoy_sets[receptor_id]) / decoy_total
        for receptor_id in receptor_ids
    }
    redundancy_raw: dict[str, float] = {}
    active_overlap_raw: dict[str, float] = {}
    pair_ensemble_utility_mean_raw: dict[str, float] = {}
    pair_ensemble_utility_min_raw: dict[str, float] = {}
    for first, second in itertools.combinations(receptor_ids, 2):
        key = f"{first}__{second}"
        correlation = float(
            spearmanr(score_columns[first], score_columns[second]).statistic
        )
        if not math.isfinite(correlation):
            raise ValueError(f"nonfinite receptor correlation: {key}")
        redundancy_raw[key] = max(0.0, correlation)
        active_overlap_raw[key] = (
            len(active_sets[first] & active_sets[second]) / active_total
        )
        pair_mean_data = {
            str(row["ligand_id"]): {
                "label": row["label"],
                "score": (float(row[first]) + float(row[second])) / 2.0,
            }
            for row in rows
        }
        pair_min_data = {
            str(row["ligand_id"]): {
                "label": row["label"],
                "score": min(float(row[first]), float(row[second])),
            }
            for row in rows
        }
        pair_ensemble_utility_mean_raw[key] = float(
            ranked_metrics_with_ids(pair_mean_data)["bedroc_alpha_20"]
        )
        pair_ensemble_utility_min_raw[key] = float(
            ranked_metrics_with_ids(pair_min_data)["bedroc_alpha_20"]
        )

    raw = {
        "utility": utility_raw,
        "active_coverage": active_coverage_raw,
        "decoy_exposure": decoy_exposure_raw,
        "redundancy": redundancy_raw,
        "active_overlap": active_overlap_raw,
        "pair_ensemble_utility": pair_ensemble_utility_mean_raw,
        "pair_ensemble_utility_mean_score": pair_ensemble_utility_mean_raw,
        "pair_ensemble_utility_min_score": pair_ensemble_utility_min_raw,
    }
    normalized = {
        name: minmax_terms(values) for name, values in raw.items()
    }
    return {
        "utility_metric": utility_metric,
        "coverage_fraction": fraction,
        "active_ids": {
            receptor_id: sorted(values)
            for receptor_id, values in active_sets.items()
        },
        "decoy_ids": {
            receptor_id: sorted(values)
            for receptor_id, values in decoy_sets.items()
        },
        "raw": raw,
        "normalized": normalized,
    }


def build_coefficients(
    terms: dict[str, object],
    receptor_ids: list[str],
    target_size: int,
    weights: dict[str, float],
    size_penalty: float,
) -> dict[str, object]:
    if not 1 <= target_size <= len(receptor_ids):
        raise ValueError("target size must be within the receptor pool")
    if size_penalty <= 0.0:
        raise ValueError("size penalty must be positive")
    required_weights = {
        "active_coverage",
        "decoy_exposure",
        "active_overlap",
        "redundancy",
    }
    optional_weights = {"ensemble_pair_utility", "stability"}
    if not required_weights.issubset(weights) or not set(weights).issubset(
        required_weights | optional_weights
    ):
        raise ValueError("QUBO weights do not match the required terms")
    if any(float(value) < 0.0 for value in weights.values()):
        raise ValueError("QUBO weights must be nonnegative")
    normalized = terms["normalized"]
    assert isinstance(normalized, dict)
    utility = normalized["utility"]
    active_coverage = normalized["active_coverage"]
    decoy_exposure = normalized["decoy_exposure"]
    active_overlap = normalized["active_overlap"]
    redundancy = normalized["redundancy"]
    pair_ensemble_utility = normalized.get("pair_ensemble_utility", {})
    pair_utility_weight = float(weights.get("ensemble_pair_utility", 0.0))
    stability = normalized.get("stability", {})
    stability_weight = float(weights.get("stability", 0.0))

    linear = {
        receptor_id: (
            -float(utility[receptor_id])
            - float(weights["active_coverage"])
            * float(active_coverage[receptor_id])
            + float(weights["decoy_exposure"])
            * float(decoy_exposure[receptor_id])
            - stability_weight * float(stability.get(receptor_id, 0.0))
            + size_penalty * (1 - 2 * target_size)
        )
        for receptor_id in receptor_ids
    }
    quadratic = {
        key: (
            float(weights["active_overlap"]) * float(active_overlap[key])
            + float(weights["redundancy"]) * float(redundancy[key])
            - pair_utility_weight
            * float(pair_ensemble_utility.get(key, 0.0))
            + 2.0 * size_penalty
        )
        for key in active_overlap
    }
    return {
        "constant": size_penalty * target_size**2,
        "linear": linear,
        "quadratic": quadratic,
        "target_size": target_size,
        "weights": {key: float(value) for key, value in weights.items()},
        "size_penalty": float(size_penalty),
        "convention": (
            "Q(x)=constant+sum_i linear[i]*x_i+"
            "sum_i<j quadratic[i__j]*x_i*x_j"
        ),
    }


def coefficient_energy(
    subset: tuple[str, ...], coefficients: dict[str, object]
) -> float:
    value = float(coefficients["constant"])
    value += sum(
        float(coefficients["linear"][receptor_id])
        for receptor_id in subset
    )
    quadratic = coefficients["quadratic"]
    for first, second in itertools.combinations(subset, 2):
        key = f"{first}__{second}"
        if key not in quadratic:
            key = f"{second}__{first}"
        value += float(quadratic[key])
    return float(value)


def exact_select(
    terms: dict[str, object],
    receptor_ids: list[str],
    target_size: int,
    weights: dict[str, float],
    size_penalty: float,
    required_receptors: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], float, dict[str, object]]:
    coefficients = build_coefficients(
        terms, receptor_ids, target_size, weights, size_penalty
    )

    required = tuple(sorted(set(required_receptors)))
    if any(receptor_id not in receptor_ids for receptor_id in required):
        raise ValueError("required receptor is absent from the receptor pool")
    if len(required) > target_size:
        raise ValueError("required receptor count exceeds target size")
    if required:
        target_candidates = [
            (subset, coefficient_energy(subset, coefficients))
            for subset in itertools.combinations(receptor_ids, target_size)
            if set(required).issubset(subset)
        ]
        if not target_candidates:
            raise ValueError("no target-size subset satisfies required receptors")
        subset, energy = min(
            target_candidates, key=lambda item: (item[1], item[0])
        )
        coefficients["exact_search"] = {
            "method": "required_receptor_constrained_cardinality_enumeration",
            "required_receptors": list(required),
            "states_evaluated": len(target_candidates),
            "full_state_count": 2 ** len(receptor_ids),
        }
        return subset, float(energy), coefficients

    # The cardinality penalty is zero at the requested size.  Enumerate that
    # small slice first, then certify it against every other cardinality with
    # a conservative lower bound.  Fall back to all 2**n states when the
    # bound is inconclusive, preserving the exact QUBO semantics.
    target_candidates = [
        (subset, coefficient_energy(subset, coefficients))
        for subset in itertools.combinations(receptor_ids, target_size)
    ]
    target_subset, target_energy = min(
        target_candidates, key=lambda item: (item[1], item[0])
    )
    penalty_linear = size_penalty * (1 - 2 * target_size)
    objective_linear = sorted(
        float(coefficients["linear"][receptor_id]) - penalty_linear
        for receptor_id in receptor_ids
    )
    objective_quadratic = sorted(
        float(value) - 2.0 * size_penalty
        for value in coefficients["quadratic"].values()
    )
    lower_bounds = {
        size: (
            size_penalty * (size - target_size) ** 2
            + sum(objective_linear[:size])
            + sum(objective_quadratic[: size * (size - 1) // 2])
        )
        for size in range(len(receptor_ids) + 1)
        if size != target_size
    }
    if all(target_energy < bound for bound in lower_bounds.values()):
        coefficients["exact_search"] = {
            "method": "target_cardinality_with_global_lower_bound_certificate",
            "states_evaluated": len(target_candidates),
            "full_state_count": 2 ** len(receptor_ids),
            "minimum_outside_cardinality_lower_bound": min(
                lower_bounds.values()
            ),
        }
        return target_subset, float(target_energy), coefficients

    candidates = [
        (subset, coefficient_energy(subset, coefficients))
        for size in range(len(receptor_ids) + 1)
        for subset in itertools.combinations(receptor_ids, size)
    ]
    subset, energy = min(candidates, key=lambda item: (item[1], item[0]))
    if len(subset) != target_size:
        raise ValueError(
            "size penalty failed to enforce the requested receptor budget"
        )
    coefficients["exact_search"] = {
        "method": "full_state_enumeration_after_inconclusive_bound",
        "states_evaluated": len(candidates),
        "full_state_count": 2 ** len(receptor_ids),
        "minimum_outside_cardinality_lower_bound": min(
            lower_bounds.values()
        ),
    }
    return subset, float(energy), coefficients
