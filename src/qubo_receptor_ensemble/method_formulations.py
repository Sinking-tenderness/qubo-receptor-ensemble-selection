"""Coefficient builders for score-matrix receptor-selection methods."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping

from .screening import ranked_metrics_with_ids


def _score(row: Mapping[str, object], receptor_id: str) -> float:
    return float(row[receptor_id])


def _rank_fractions(
    rows: list[dict[str, object]], receptor_id: str
) -> dict[str, float]:
    ordered = sorted(
        rows,
        key=lambda row: (_score(row, receptor_id), str(row["ligand_id"])),
    )
    denominator = max(1, len(ordered) - 1)
    return {
        str(row["ligand_id"]): index / denominator
        for index, row in enumerate(ordered)
    }


def _rank_weight(rank: float, alpha: float) -> float:
    return math.exp(-alpha * rank)


def _class_contrast(
    rows: list[dict[str, object]],
    weights: Mapping[str, float],
    decoy_penalty_lambda: float,
) -> float:
    active = [weights[str(row["ligand_id"])] for row in rows if row["label"] == "active"]
    decoys = [weights[str(row["ligand_id"])] for row in rows if row["label"] == "decoy"]
    if not active or not decoys:
        raise ValueError("QUBO utility requires both active and decoy rows")
    return sum(active) / len(active) - decoy_penalty_lambda * sum(decoys) / len(decoys)


def _coefficient_qubo(
    *,
    method_id: str,
    target_size: int,
    linear_utility: Mapping[str, float],
    pair_utility: Mapping[str, float],
    config: Mapping[str, object],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    size_penalty = float(config.get("size_penalty", 0.0))
    if size_penalty < 0:
        raise ValueError("size_penalty must be non-negative")
    linear = {
        receptor_id: -float(value) + size_penalty * (1 - 2 * target_size)
        for receptor_id, value in linear_utility.items()
    }
    quadratic = {
        key: -float(value) + 2.0 * size_penalty
        for key, value in pair_utility.items()
    }
    return {
        "method_id": method_id,
        "target_size": target_size,
        "objective_mode": "coefficient",
        "fixed_cardinality": True,
        "constant": size_penalty * target_size**2,
        "linear_coefficients": linear,
        "quadratic_coefficients": quadratic,
        "convention": (
            "Q(x)=constant+sum_i linear[i]*x_i+"
            "sum_i<j quadratic[i__j]*x_i*x_j"
        ),
        **dict(metadata),
    }


def _raw_pair_bedroc_utility(
    rows: list[dict[str, object]],
    first: str,
    second: str,
    alpha: float,
) -> float:
    data = {
        str(row["ligand_id"]): {
            "label": row["label"],
            "score": (_score(row, first) + _score(row, second)) / 2.0,
        }
        for row in rows
    }
    return float(
        ranked_metrics_with_ids(data, bedroc_alpha=alpha)[f"bedroc_alpha_{alpha:g}"]
    )


def build_rank_pair_qubo(
    rows: list[dict[str, object]],
    receptor_ids: list[str],
    target_size: int,
    config: Mapping[str, object],
    method_id: str,
) -> dict[str, object]:
    """Build pair utility or rank-synergy coefficient QUBOs."""
    alpha = float(config.get("bedroc_alpha", 20.0))
    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("bedroc_alpha must be a positive finite number")
    lambda_decoy = float(
        config.get(
            "decoy_penalty_lambda",
            1.5 if method_id == "bedroc20_pair_synergy" else 1.0,
        )
    )
    if lambda_decoy < 0:
        raise ValueError("decoy_penalty_lambda must be non-negative")

    ranks = {
        receptor_id: _rank_fractions(rows, receptor_id)
        for receptor_id in receptor_ids
    }
    singleton = {
        receptor_id: _class_contrast(
            rows,
            {
                str(row["ligand_id"]): _rank_weight(
                    ranks[receptor_id][str(row["ligand_id"])], alpha
                )
                for row in rows
            },
            lambda_decoy,
        )
        for receptor_id in receptor_ids
    }
    pairs: dict[str, float] = {}
    pair_raw: dict[str, float] = {}
    for first, second in itertools.combinations(receptor_ids, 2):
        key = f"{first}__{second}"
        pair_weights = {
            str(row["ligand_id"]): _rank_weight(
                (ranks[first][str(row["ligand_id"])] + ranks[second][str(row["ligand_id"])])
                / 2.0,
                alpha,
            )
            for row in rows
        }
        pair_value = _class_contrast(rows, pair_weights, lambda_decoy)
        pair_raw[key] = pair_value
        pairs[key] = (
            pair_value - (singleton[first] + singleton[second]) / 2.0
            if method_id in {"pair_synergy", "bedroc20_pair_synergy", "rank_sensitive_pair"}
            else pair_value
        )

    if method_id == "pair_utility":
        raw_singleton = {}
        for receptor_id in receptor_ids:
            data = {
                str(row["ligand_id"]): {
                    "label": row["label"],
                    "score": _score(row, receptor_id),
                }
                for row in rows
            }
            raw_singleton[receptor_id] = float(
                ranked_metrics_with_ids(data, bedroc_alpha=alpha)[
                    f"bedroc_alpha_{alpha:g}"
                ]
            )
        singleton = raw_singleton
        pair_raw = {
            key: _raw_pair_bedroc_utility(rows, first, second, alpha)
            for first, second in itertools.combinations(receptor_ids, 2)
            for key in [f"{first}__{second}"]
        }
        pairs = pair_raw

    return _coefficient_qubo(
        method_id=method_id,
        target_size=target_size,
        linear_utility=singleton,
        pair_utility=pairs,
        config=config,
        metadata={
            "utility_metric": "bedroc",
            "bedroc_alpha": alpha,
            "decoy_penalty_lambda": lambda_decoy,
            "rank_normalization": "empirical percentile, best rank is zero",
            "singleton_utility": singleton,
            "pair_utility": pair_raw,
            "pair_coefficient": (
                "pair utility"
                if method_id == "pair_utility"
                else "pair utility minus mean singleton utility"
            ),
        },
    )


def build_auxiliary_coverage_qubo(
    rows: list[dict[str, object]],
    receptor_ids: list[str],
    target_size: int,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Build the score-only active coverage approximation used by early coverage stages."""
    fraction = float(config.get("coverage_fraction", 0.10))
    if not 0 < fraction <= 1:
        raise ValueError("coverage_fraction must be in (0, 1]")
    active_ids = {str(row["ligand_id"]) for row in rows if row["label"] == "active"}
    if not active_ids:
        raise ValueError("coverage QUBO requires active rows")
    top_count = max(1, math.ceil(len(rows) * fraction))
    covered: dict[str, set[str]] = {}
    for receptor_id in receptor_ids:
        ordered = sorted(
            rows,
            key=lambda row: (_score(row, receptor_id), str(row["ligand_id"])),
        )
        covered[receptor_id] = {
            str(row["ligand_id"])
            for row in ordered[:top_count]
            if row["label"] == "active"
        }
    coverage = {
        receptor_id: len(covered[receptor_id]) / len(active_ids)
        for receptor_id in receptor_ids
    }
    overlap = {
        f"{first}__{second}": len(covered[first] & covered[second]) / len(active_ids)
        for first, second in itertools.combinations(receptor_ids, 2)
    }
    base = build_rank_pair_qubo(
        rows,
        receptor_ids,
        target_size,
        config,
        "rank_sensitive_pair",
    )
    linear = {
        receptor_id: float(base["linear_coefficients"][receptor_id])
        - float(config.get("coverage_weight", 0.5)) * coverage[receptor_id]
        for receptor_id in receptor_ids
    }
    quadratic = {
        key: float(base["quadratic_coefficients"][key])
        + float(config.get("overlap_weight", 0.5)) * overlap[key]
        for key in overlap
    }
    return {
        **base,
        "method_id": "auxiliary_coverage",
        "linear_coefficients": linear,
        "quadratic_coefficients": quadratic,
        "coverage_fraction": fraction,
        "coverage_rewards": coverage,
        "coverage_overlap": overlap,
        "coverage_encoding": "pairwise active-top-set coverage approximation",
    }


def build_rankbin_qubo(
    rows: list[dict[str, object]],
    receptor_ids: list[str],
    target_size: int,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Build the compact receptor-level approximation to the rank-bin utility."""
    bin_count = int(config.get("bin_count", 32))
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    alpha = float(config.get("bedroc_alpha", 20.0))
    ranks = {
        receptor_id: _rank_fractions(rows, receptor_id)
        for receptor_id in receptor_ids
    }
    singleton = {
        receptor_id: _class_contrast(
            rows,
            {
                str(row["ligand_id"]): math.floor(
                    bin_count
                    * _rank_weight(
                        ranks[receptor_id][str(row["ligand_id"])], alpha
                    )
                )
                / bin_count
                for row in rows
            },
            float(config.get("decoy_penalty_lambda", 1.0)),
        )
        for receptor_id in receptor_ids
    }
    pair = {
        f"{first}__{second}": 0.0
        for first, second in itertools.combinations(receptor_ids, 2)
    }
    return _coefficient_qubo(
        method_id="rankbin_bedroc20",
        target_size=target_size,
        linear_utility=singleton,
        pair_utility=pair,
        config=config,
        metadata={
            "utility_metric": "bedroc_rankbin",
            "bedroc_alpha": alpha,
            "bin_count": bin_count,
            "encoding": "receptor-level rank-bin approximation",
        },
    )
