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

    raw = {
        "utility": utility_raw,
        "active_coverage": active_coverage_raw,
        "decoy_exposure": decoy_exposure_raw,
        "redundancy": redundancy_raw,
        "active_overlap": active_overlap_raw,
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
    if set(weights) != required_weights:
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

    linear = {
        receptor_id: (
            -float(utility[receptor_id])
            - float(weights["active_coverage"])
            * float(active_coverage[receptor_id])
            + float(weights["decoy_exposure"])
            * float(decoy_exposure[receptor_id])
            + size_penalty * (1 - 2 * target_size)
        )
        for receptor_id in receptor_ids
    }
    quadratic = {
        key: (
            float(weights["active_overlap"]) * float(active_overlap[key])
            + float(weights["redundancy"]) * float(redundancy[key])
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
    selected = set(subset)
    value = float(coefficients["constant"])
    value += sum(
        float(coefficients["linear"][receptor_id])
        for receptor_id in selected
    )
    value += sum(
        float(coefficient)
        for key, coefficient in coefficients["quadratic"].items()
        if all(receptor_id in selected for receptor_id in key.split("__"))
    )
    return float(value)


def exact_select(
    terms: dict[str, object],
    receptor_ids: list[str],
    target_size: int,
    weights: dict[str, float],
    size_penalty: float,
) -> tuple[tuple[str, ...], float, dict[str, object]]:
    coefficients = build_coefficients(
        terms, receptor_ids, target_size, weights, size_penalty
    )
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
    return subset, float(energy), coefficients
