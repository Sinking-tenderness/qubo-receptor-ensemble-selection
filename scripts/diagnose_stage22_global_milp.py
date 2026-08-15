"""Use a classical MILP oracle to diagnose Stage 22 search quality.

This is a post-hoc, structure-only diagnostic.  It solves the frozen reduced
coverage-plus-diversity objective globally (when HiGHS finishes within the
time limit) without changing the Stage 22 Go/No-Go gate.  The purpose is to
separate a difficult objective from an inadequate multistart search heuristic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    descriptor,
    distance_matrix,
    file_sha256,
    load_target,
    read_json,
    rooted,
    write_json,
)
from scripts.run_stage22_structural_state_coverage_qubo import (
    CANDIDATE_METHOD,
    build_coverage_terms,
    direct_greedy,
    objective_components,
)


def build_milp(
    ids: list[str],
    matrix: np.ndarray,
    terms: dict[str, Any],
    target_k: int,
    diversity_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """Build a binary MILP exactly equivalent to the reduced objective."""
    candidate_count = len(ids)
    state_ids = list(terms["state_ids"])
    pairs = [
        (ids[first], ids[second])
        for first in range(candidate_count)
        for second in range(first + 1, candidate_count)
    ]
    x_offset = 0
    y_offset = candidate_count
    pair_offset = y_offset + len(state_ids)
    variable_count = pair_offset + len(pairs)
    c = np.zeros(variable_count, dtype=float)
    c[y_offset : y_offset + len(state_ids)] = -float(terms["state_weight"])
    denominator = max(1, target_k * (target_k - 1) // 2)
    for pair_index, (first, second) in enumerate(pairs):
        first_index = ids.index(first)
        second_index = ids.index(second)
        c[pair_offset + pair_index] = -float(diversity_weight) * float(
            matrix[first_index, second_index]
        ) / denominator

    rows: list[tuple[dict[int, float], float, float]] = []
    rows.append(
        (
            {x_offset + index: 1.0 for index in range(candidate_count)},
            float(target_k),
            float(target_k),
        )
    )
    id_to_index = {value: index for index, value in enumerate(ids)}
    for state_index, state_id in enumerate(state_ids):
        coefficients = {y_offset + state_index: 1.0}
        for conformer_id in terms["incidence"][state_id]:
            coefficients[x_offset + id_to_index[conformer_id]] = -1.0
        rows.append((coefficients, -np.inf, 0.0))
    for pair_index, (first, second) in enumerate(pairs):
        first_index = x_offset + id_to_index[first]
        second_index = x_offset + id_to_index[second]
        w_index = pair_offset + pair_index
        rows.extend(
            [
                ({w_index: 1.0, first_index: -1.0}, -np.inf, 0.0),
                ({w_index: 1.0, second_index: -1.0}, -np.inf, 0.0),
                ({w_index: 1.0, first_index: -1.0, second_index: -1.0}, -1.0, np.inf),
            ]
        )
    constraint_matrix = lil_matrix((len(rows), variable_count), dtype=float)
    lower = np.empty(len(rows), dtype=float)
    upper = np.empty(len(rows), dtype=float)
    for row_index, (coefficients, row_lower, row_upper) in enumerate(rows):
        for column, value in coefficients.items():
            constraint_matrix[row_index, column] = value
        lower[row_index] = row_lower
        upper[row_index] = row_upper
    return c, constraint_matrix.tocsr(), lower, upper, pairs


def solve_one(
    target_id: str,
    fraction: float,
    target: dict[str, Any],
    target_spec: dict[str, Any],
    target_k: int,
    diversity_weight: float,
    time_limit: float,
    mip_rel_gap: float,
    node_limit: int | None,
) -> dict[str, Any]:
    ids = target["ids"]
    matrix = distance_matrix(ids, target["distances"])
    terms = build_coverage_terms(ids, matrix, fraction)
    c, constraint_matrix, lower, upper, pairs = build_milp(
        ids, matrix, terms, target_k, diversity_weight
    )
    started = time.perf_counter()
    options: dict[str, Any] = {
        "time_limit": float(time_limit),
        "mip_rel_gap": float(mip_rel_gap),
    }
    if node_limit is not None:
        options["node_limit"] = int(node_limit)
    result = milp(
        c=c,
        integrality=np.ones(len(c), dtype=float),
        bounds=Bounds(np.zeros(len(c)), np.ones(len(c))),
        constraints=LinearConstraint(constraint_matrix, lower, upper),
        options=options,
    )
    elapsed = time.perf_counter() - started
    if result.x is None:
        raise RuntimeError(
            f"{target_id}/{fraction}/{target_k}: MILP returned no incumbent: {result.message}"
        )
    x = np.rint(result.x[: len(ids)]).astype(int)
    selected = tuple(sorted(ids[index] for index, value in enumerate(x) if value))
    if len(selected) != target_k:
        raise RuntimeError(
            f"{target_id}/{fraction}/{target_k}: MILP incumbent has wrong cardinality"
        )
    metrics = objective_components(
        selected, ids, matrix, terms, diversity_weight
    )
    greedy = direct_greedy(ids, matrix, terms, target_k, diversity_weight)
    greedy_metrics = objective_components(
        greedy, ids, matrix, terms, diversity_weight
    )
    return {
        "target_id": target_id,
        "neighborhood_fraction": fraction,
        "k": target_k,
        "selected_subset": list(selected),
        "coverage_fraction": metrics["coverage_fraction"],
        "mean_pair_distance_normalized": metrics["mean_pair_distance_normalized"],
        "composite_objective": metrics["composite_objective"],
        "direct_greedy_subset": list(greedy),
        "direct_greedy_objective": greedy_metrics["composite_objective"],
        "delta_vs_direct_greedy": metrics["composite_objective"]
        - greedy_metrics["composite_objective"],
        "solver": "scipy.optimize.milp-highs",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "mip_gap": None
        if getattr(result, "mip_gap", None) is None
        else float(result.mip_gap),
        "mip_node_count": None
        if getattr(result, "mip_node_count", None) is None
        else int(result.mip_node_count),
        "variable_count": len(c),
        "pair_variable_count": len(pairs),
        "constraint_count": constraint_matrix.shape[0],
        "elapsed_seconds": elapsed,
        "time_limit_seconds": time_limit,
    }


def run(
    config_path: Path,
    root: Path,
    output_path: Path,
    time_limit: float,
    mip_rel_gap: float,
    node_limit: int | None,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    diagnostic = config["diagnostic"]
    target_k = int(config["go_no_go"]["required_common_k"])
    diversity_weight = float(diagnostic["diversity_weight"])
    rows: list[dict[str, Any]] = []
    for target_id, target_spec in config["targets"].items():
        target = load_target(root, target_id, target_spec)
        for fraction in diagnostic["neighborhood_fractions"]:
            row = solve_one(
                target_id,
                float(fraction),
                target,
                target_spec,
                target_k,
                    diversity_weight,
                    time_limit,
                    mip_rel_gap,
                    node_limit,
                )
            rows.append(row)
            print(
                json.dumps(
                    {
                        "target_id": target_id,
                        "neighborhood_fraction": float(fraction),
                        "status": row["solver_status"],
                        "objective": row["composite_objective"],
                        "delta_vs_direct_greedy": row["delta_vs_direct_greedy"],
                        "mip_gap": row["mip_gap"],
                        "elapsed_seconds": row["elapsed_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    result = {
        "schema_version": "1.0",
        "status": "stage22_global_milp_diagnostic_complete",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "purpose": "Post-hoc, structure-only diagnosis of whether multistart search misses a better global solution; this does not revise the frozen Stage22 gate.",
        "solver_limits": {
            "time_limit_seconds": time_limit,
            "mip_rel_gap": mip_rel_gap,
            "node_limit": node_limit,
        },
        "evidence_boundary": {
            "docking_scores_read": 0,
            "ligand_labels_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "rows": rows,
    }
    output = rooted(root, output_path.as_posix())
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage22_structural_state_coverage_qubo.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage22_global_milp_diagnostic.json"),
    )
    parser.add_argument("--time-limit", type=float, default=180.0)
    parser.add_argument("--mip-rel-gap", type=float, default=0.05)
    parser.add_argument("--node-limit", type=int, default=None)
    args = parser.parse_args()
    run(
        args.config,
        args.root,
        args.output,
        args.time_limit,
        args.mip_rel_gap,
        args.node_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
