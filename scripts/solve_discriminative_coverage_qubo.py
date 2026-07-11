"""Build and exactly solve a fixed-weight discriminative coverage QUBO."""

from __future__ import annotations

import itertools
from typing import Iterable

try:
    from .analyze_active_coverage import top_active_ids
    from .select_receptor_baselines import read_csv
    from .solve_qubo_receptor_subset import build_qubo, objective
except ImportError:
    from analyze_active_coverage import top_active_ids
    from select_receptor_baselines import read_csv
    from solve_qubo_receptor_subset import build_qubo, objective


def top_label_ids(
    rows: list[dict[str, str]], receptor_id: str, fraction: float, label: str
) -> set[str]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ordered = sorted(rows, key=lambda row: (float(row[receptor_id]), row["ligand_id"]))
    top_n = max(1, int((len(rows) * fraction) + 0.999999))
    return {row["ligand_id"] for row in ordered[:top_n] if row["label"] == label}


def discriminative_terms(
    rows: list[dict[str, str]],
    receptor_ids: list[str],
    fraction: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    active_total = sum(row["label"] == "active" for row in rows)
    decoy_total = sum(row["label"] == "decoy" for row in rows)
    if not active_total or not decoy_total:
        raise ValueError("both active and decoy rows are required")
    active_sets = {
        receptor_id: top_label_ids(rows, receptor_id, fraction, "active")
        for receptor_id in receptor_ids
    }
    decoy_sets = {
        receptor_id: top_label_ids(rows, receptor_id, fraction, "decoy")
        for receptor_id in receptor_ids
    }
    active_rewards = {
        receptor_id: len(active_sets[receptor_id]) / active_total
        for receptor_id in receptor_ids
    }
    decoy_exposure = {
        receptor_id: len(decoy_sets[receptor_id]) / decoy_total
        for receptor_id in receptor_ids
    }
    active_overlaps = {
        f"{first}__{second}": len(active_sets[first] & active_sets[second]) / active_total
        for first, second in itertools.combinations(receptor_ids, 2)
    }
    return active_rewards, decoy_exposure, active_overlaps


def build_discriminative_qubo(
    rows: list[dict[str, str]],
    receptor_ids: list[str],
    target_size: int,
    utility_metric: str = "bedroc",
    utility_normalization: str = "minmax",
    active_weight: float = 0.5,
    decoy_weight: float = 0.5,
    active_overlap_weight: float = 0.5,
    redundancy_weight: float = 0.25,
    count_weight: float = 0.1,
    size_weight: float = 1.0,
    fraction: float = 0.10,
) -> dict[str, object]:
    base = build_qubo(
        rows,
        receptor_ids,
        target_size,
        redundancy_weight,
        count_weight,
        size_weight,
        utility_metric,
        utility_normalization,
    )
    active_rewards, decoy_exposure, active_overlaps = discriminative_terms(
        rows, receptor_ids, fraction
    )
    linear = {
        receptor_id: float(base["linear_coefficients"][receptor_id])
        - active_weight * active_rewards[receptor_id]
        + decoy_weight * decoy_exposure[receptor_id]
        for receptor_id in receptor_ids
    }
    quadratic = {
        key: float(base["quadratic_coefficients"][key])
        + active_overlap_weight * active_overlaps[key]
        for key in base["quadratic_coefficients"]
    }
    return {
        "target_size": target_size,
        "utility_metric": utility_metric,
        "utility_normalization": utility_normalization,
        "weights": {
            "active_coverage": active_weight,
            "decoy_exposure": decoy_weight,
            "active_overlap": active_overlap_weight,
            "redundancy": redundancy_weight,
            "count": count_weight,
            "size": size_weight,
        },
        "active_rewards_train": active_rewards,
        "decoy_exposure_train": decoy_exposure,
        "active_overlap_train": active_overlaps,
        "linear_coefficients": linear,
        "quadratic_coefficients": quadratic,
        "constant": size_weight * target_size**2,
        "convention": "Q(x)=constant+sum_i linear[i]*x_i+sum_i<j quadratic[i__j]*x_i*x_j",
    }


def coefficient_energy(subset: Iterable[str], qubo: dict[str, object]) -> float:
    selected = set(subset)
    value = float(qubo["constant"])
    value += sum(float(qubo["linear_coefficients"][item]) for item in selected)
    value += sum(
        float(coefficient)
        for key, coefficient in qubo["quadratic_coefficients"].items()
        if all(item in selected for item in key.split("__"))
    )
    return value


def select_discriminative_subset(
    rows: list[dict[str, str]], receptor_ids: list[str], target_size: int, **kwargs: float | str
) -> tuple[tuple[str, ...], dict[str, object]]:
    qubo = build_discriminative_qubo(rows, receptor_ids, target_size, **kwargs)
    candidates = [
        (subset, coefficient_energy(subset, qubo))
        for subset in itertools.combinations(receptor_ids, target_size)
    ]
    subset, value = min(candidates, key=lambda item: (item[1], item[0]))
    return subset, {"qubo": qubo, "best_subset": list(subset), "best_energy": value}
