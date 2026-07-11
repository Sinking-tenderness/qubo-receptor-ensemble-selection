"""Build and exhaustively solve a coverage-aware receptor QUBO prototype."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

try:
    from .analyze_active_coverage import top_active_ids
    from .select_receptor_baselines import metrics_for_subset, read_csv
    from .solve_qubo_receptor_subset import build_qubo, objective
except ImportError:
    from analyze_active_coverage import top_active_ids
    from select_receptor_baselines import metrics_for_subset, read_csv
    from solve_qubo_receptor_subset import build_qubo, objective


def coverage_terms(rows: list[dict[str, str]], receptor_ids: list[str], fraction: float):
    active_total = {row["ligand_id"] for row in rows if row["label"] == "active"}
    if not active_total:
        raise ValueError("coverage requires at least one active ligand")
    sets = {
        receptor_id: top_active_ids(rows, receptor_id, fraction)
        for receptor_id in receptor_ids
    }
    rewards = {
        receptor_id: len(active_ids) / len(active_total)
        for receptor_id, active_ids in sets.items()
    }
    overlaps = {
        f"{first}__{second}": len(sets[first] & sets[second]) / len(active_total)
        for first, second in itertools.combinations(receptor_ids, 2)
    }
    return sets, rewards, overlaps


def coverage_objective(
    subset: tuple[str, ...],
    base_qubo: dict[str, object],
    coverage_rewards: dict[str, float],
    overlaps: dict[str, float],
    coverage_weight: float,
    overlap_weight: float,
) -> float:
    value = objective(subset, base_qubo)
    value -= coverage_weight * sum(coverage_rewards[receptor_id] for receptor_id in subset)
    for first, second in itertools.combinations(subset, 2):
        value += overlap_weight * overlaps[f"{first}__{second}"]
    return float(value)


def combined_coefficients(
    base_qubo: dict[str, object],
    coverage_rewards: dict[str, float],
    overlaps: dict[str, float],
    coverage_weight: float,
    overlap_weight: float,
) -> dict[str, object]:
    """Return QUBO coefficients under Q(x)=c+sum(h_i x_i)+sum(J_ij x_i x_j)."""
    linear = {
        receptor_id: float(value) - coverage_weight * coverage_rewards[receptor_id]
        for receptor_id, value in base_qubo["linear_coefficients"].items()
    }
    quadratic = {
        key: float(value) + overlap_weight * overlaps[key]
        for key, value in base_qubo["quadratic_coefficients"].items()
    }
    constant = float(
        base_qubo["weights"]["size"] * base_qubo["target_size"] ** 2
    )
    return {
        "constant": constant,
        "linear": linear,
        "quadratic": quadratic,
        "convention": "Q(x)=constant+sum_i linear[i]*x_i+sum_i<j quadratic[i__j]*x_i*x_j",
    }


def coefficient_energy(subset: tuple[str, ...], coefficients: dict[str, object]) -> float:
    selected = set(subset)
    value = float(coefficients["constant"])
    value += sum(
        float(coefficients["linear"][receptor_id])
        for receptor_id in selected
    )
    value += sum(
        float(value)
        for key, value in coefficients["quadratic"].items()
        if all(receptor_id in selected for receptor_id in key.split("__"))
    )
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--receptor", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=2)
    parser.add_argument("--coverage-fraction", type=float, default=0.10)
    parser.add_argument("--coverage-weight", type=float, default=0.50)
    parser.add_argument("--overlap-weight", type=float, default=0.50)
    parser.add_argument("--redundancy-weight", type=float, default=0.25)
    parser.add_argument("--count-weight", type=float, default=0.10)
    parser.add_argument("--size-weight", type=float, default=1.0)
    parser.add_argument("--utility-metric", choices=["roc_auc", "bedroc", "ef5"], default="roc_auc")
    parser.add_argument("--utility-normalization", choices=["none", "minmax"], default="minmax")
    args = parser.parse_args()

    matrix_rows = read_csv(args.matrix)
    split_rows = read_csv(args.split_manifest)
    split_by_ligand = {row["ligand_id"]: row["split"] for row in split_rows}
    train_rows = [row for row in matrix_rows if split_by_ligand[row["ligand_id"]] == "train"]
    rows_by_split = {
        split: [row for row in matrix_rows if split_by_ligand[row["ligand_id"]] == split]
        for split in ("train", "validation", "test")
    }
    base_qubo = build_qubo(
        train_rows,
        args.receptor,
        args.target_size,
        args.redundancy_weight,
        args.count_weight,
        args.size_weight,
        args.utility_metric,
        args.utility_normalization,
    )
    coverage_sets, coverage_rewards, overlaps = coverage_terms(
        train_rows, args.receptor, args.coverage_fraction
    )
    coefficients = combined_coefficients(
        base_qubo,
        coverage_rewards,
        overlaps,
        args.coverage_weight,
        args.overlap_weight,
    )
    candidates = []
    for size in range(len(args.receptor) + 1):
        for subset in itertools.combinations(args.receptor, size):
            candidates.append(
                {
                    "subset": list(subset),
                    "size": size,
                    "objective": coverage_objective(
                        subset,
                        base_qubo,
                        coverage_rewards,
                        overlaps,
                        args.coverage_weight,
                        args.overlap_weight,
                    ),
                }
            )
    candidates.sort(key=lambda row: (row["objective"], row["subset"]))
    best = candidates[0]
    subset = tuple(best["subset"])
    result = {
        "selection_split": "train",
        "target_size": args.target_size,
        "coverage_encoding": "pairwise union exact for target_size=2; approximation for larger subsets",
        "weights": {
            "coverage": args.coverage_weight,
            "coverage_overlap": args.overlap_weight,
        },
        "coverage_fraction": args.coverage_fraction,
        "coverage_sets_train": {key: sorted(value) for key, value in coverage_sets.items()},
        "coverage_rewards_train": coverage_rewards,
        "coverage_overlap_train": overlaps,
        "base_qubo": base_qubo,
        "qubo_coefficients": coefficients,
        "best_subset": best,
        "selected_subset_metrics": {
            split: metrics_for_subset(rows, subset, "min_score")
            for split, rows in rows_by_split.items()
        },
        "all_candidates": candidates,
    }
    result["coefficient_energy_check"] = {
        "+".join(candidate["subset"]): coefficient_energy(
            tuple(candidate["subset"]), coefficients
        )
        for candidate in candidates
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(result["best_subset"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
