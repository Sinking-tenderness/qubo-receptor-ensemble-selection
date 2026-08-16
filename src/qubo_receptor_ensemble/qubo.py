"""QUBO receptor-subset formulation.

Consolidated from ``scripts/solve_qubo_receptor_subset.py``; behavior is
identical to the original prototype used across the project.
"""

from __future__ import annotations

import itertools

from scipy.stats import spearmanr

from .screening import ranked_metrics_with_ids


def train_data(rows: list[dict[str, str]], receptor_id: str) -> dict[str, dict[str, object]]:
    return {
        row["ligand_id"]: {"label": row["label"], "score": float(row[receptor_id])}
        for row in rows
    }


def build_qubo(
    rows: list[dict[str, str]],
    receptor_ids: list[str],
    target_size: int,
    redundancy_weight: float,
    count_weight: float,
    size_weight: float,
    utility_metric: str = "roc_auc",
    utility_normalization: str = "none",
) -> dict[str, object]:
    utilities: dict[str, float] = {}
    train_scores: dict[str, list[float]] = {}
    for receptor_id in receptor_ids:
        data = train_data(rows, receptor_id)
        metrics = ranked_metrics_with_ids(data)
        metric_key = {
            "roc_auc": "roc_auc",
            "bedroc": "bedroc_alpha_20",
            "ef5": "EF5%",
        }[utility_metric]
        utilities[receptor_id] = float(metrics[metric_key])
        train_scores[receptor_id] = [float(row[receptor_id]) for row in rows]

    if utility_normalization == "minmax":
        minimum = min(utilities.values())
        maximum = max(utilities.values())
        if maximum == minimum:
            utilities = {receptor_id: 0.5 for receptor_id in utilities}
        else:
            utilities = {
                receptor_id: (value - minimum) / (maximum - minimum)
                for receptor_id, value in utilities.items()
            }
    elif utility_normalization != "none":
        raise ValueError(f"unsupported utility normalization: {utility_normalization}")

    redundancy: dict[str, float] = {}
    for first, second in itertools.combinations(receptor_ids, 2):
        value = float(spearmanr(train_scores[first], train_scores[second]).statistic)
        redundancy[f"{first}__{second}"] = max(0.0, value)

    linear = {
        receptor_id: -utilities[receptor_id]
        + count_weight
        + size_weight * (1 - 2 * target_size)
        for receptor_id in receptor_ids
    }
    quadratic = {
        key: redundancy[key] * redundancy_weight + 2 * size_weight
        for key in redundancy
    }

    return {
        "target_size": target_size,
        "utility_metric": utility_metric,
        "utility_normalization": utility_normalization,
        "weights": {
            "redundancy": redundancy_weight,
            "count": count_weight,
            "size": size_weight,
        },
        "utilities_train_roc_auc": utilities,
        "redundancy_train_spearman_clipped": redundancy,
        "linear_coefficients": linear,
        "quadratic_coefficients": quadratic,
    }


def objective(
    subset: tuple[str, ...], qubo: dict[str, object]
) -> float:
    utilities = qubo["utilities_train_roc_auc"]
    redundancy = qubo["redundancy_train_spearman_clipped"]
    weights = qubo["weights"]
    target_size = qubo["target_size"]
    value = -sum(utilities[receptor_id] for receptor_id in subset)
    value += weights["count"] * len(subset)
    value += weights["size"] * (len(subset) - target_size) ** 2
    for first, second in itertools.combinations(subset, 2):
        value += weights["redundancy"] * redundancy[f"{first}__{second}"]
    return float(value)
