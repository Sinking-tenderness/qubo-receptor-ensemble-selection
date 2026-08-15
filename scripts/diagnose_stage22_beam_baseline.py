"""Compare Stage 22 coverage-QUBO candidates with deterministic beam search.

This is a post-hoc, structure-only diagnostic.  Direct greedy is loaded as a
separate baseline; beam widths retain multiple feasible partial subsets and
form increasingly strong classical forward-search baselines without reading
docking outcomes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    distance_matrix,
    file_sha256,
    load_target,
    read_csv,
    read_json,
    rooted,
    write_json,
)
from scripts.run_stage22_structural_state_coverage_qubo import (
    CANDIDATE_METHOD,
    build_coverage_terms,
    improve_by_swaps,
    objective_components,
    objective_key,
)


def beam_search(
    ids: list[str],
    matrix: Any,
    terms: dict[str, Any],
    target_k: int,
    diversity_weight: float,
    beam_width: int,
) -> tuple[tuple[str, ...], dict[str, float]]:
    if beam_width < 1:
        raise ValueError("beam width must be positive")
    states: list[tuple[str, ...]] = [()]
    for depth in range(target_k):
        expanded: set[tuple[str, ...]] = set()
        new_depth = depth + 1
        last_feasible_index = len(ids) - (target_k - new_depth) - 1
        for state in states:
            last_index = -1 if not state else ids.index(state[-1])
            for candidate_index in range(
                last_index + 1, last_feasible_index + 1
            ):
                expanded.add(tuple(sorted((*state, ids[candidate_index]))))
        states = sorted(
            expanded,
            key=lambda state: objective_key(
                state, ids, matrix, terms, diversity_weight
            ),
        )[:beam_width]
    selected = states[0]
    return selected, objective_components(
        selected, ids, matrix, terms, diversity_weight
    )


def run(
    config_path: Path,
    root: Path,
    output_path: Path,
    beam_width_override: list[int] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    result_path = rooted(root, config["outputs"]["result_json"])
    stage22_result = read_json(result_path)
    selection_path = rooted(root, stage22_result["outputs"]["selection_csv"]["path"])
    selection_rows = read_csv(selection_path)
    diagnostic = config["diagnostic"]
    target_k = int(config["go_no_go"]["required_common_k"])
    diversity_weight = float(diagnostic["diversity_weight"])
    beam_widths = [
        int(value)
        for value in (
            beam_width_override
            if beam_width_override is not None
            else diagnostic.get("beam_widths", [32, 128, 512, 2048])
        )
    ]
    rows: list[dict[str, Any]] = []
    for target_id, target_spec in config["targets"].items():
        target = load_target(root, target_id, target_spec)
        ids = target["ids"]
        matrix = distance_matrix(ids, target["distances"])
        for fraction_value in diagnostic["neighborhood_fractions"]:
            fraction = float(fraction_value)
            terms = build_coverage_terms(ids, matrix, fraction)
            qrow = next(
                row
                for row in selection_rows
                if row["target_id"] == target_id
                and row["method"] == CANDIDATE_METHOD
                and int(row["k"]) == target_k
                and abs(float(row["neighborhood_fraction"]) - fraction) < 1e-12
            )
            qobjective = float(qrow["composite_objective"])
            for beam_width in beam_widths:
                started = time.perf_counter()
                selected, metrics = beam_search(
                    ids,
                    matrix,
                    terms,
                    target_k,
                    diversity_weight,
                    beam_width,
                )
                refined, refined_metrics, refinement_iterations = improve_by_swaps(
                    selected,
                    ids,
                    matrix,
                    terms,
                    target_k,
                    diversity_weight,
                )
                elapsed = time.perf_counter() - started
                rows.append(
                    {
                        "target_id": target_id,
                        "neighborhood_fraction": fraction,
                        "k": target_k,
                        "beam_width": beam_width,
                        "selected_subset": list(selected),
                        "coverage_fraction": metrics["coverage_fraction"],
                        "mean_pair_distance_normalized": metrics[
                            "mean_pair_distance_normalized"
                        ],
                        "composite_objective": metrics["composite_objective"],
                        "delta_vs_direct_greedy": metrics["composite_objective"]
                        - float(
                            next(
                                row["composite_objective"]
                                for row in selection_rows
                                if row["target_id"] == target_id
                                and row["method"] == "direct_greedy"
                                and int(row["k"]) == target_k
                                and abs(float(row["neighborhood_fraction"]) - fraction)
                                < 1e-12
                            )
                        ),
                        "delta_vs_stage22_candidate": metrics["composite_objective"]
                        - qobjective,
                        "refined_subset": list(refined),
                        "refined_coverage_fraction": refined_metrics[
                            "coverage_fraction"
                        ],
                        "refined_mean_pair_distance_normalized": refined_metrics[
                            "mean_pair_distance_normalized"
                        ],
                        "refined_composite_objective": refined_metrics[
                            "composite_objective"
                        ],
                        "refined_delta_vs_stage22_candidate": refined_metrics[
                            "composite_objective"
                        ]
                        - qobjective,
                        "refinement_iterations": refinement_iterations,
                        "elapsed_seconds": elapsed,
                    }
                )
                print(
                    json.dumps(
                        {
                            "target_id": target_id,
                            "neighborhood_fraction": fraction,
                            "beam_width": beam_width,
                            "objective": metrics["composite_objective"],
                            "delta_vs_stage22_candidate": metrics["composite_objective"]
                            - qobjective,
                            "refined_objective": refined_metrics[
                                "composite_objective"
                            ],
                            "refined_delta_vs_stage22_candidate": refined_metrics[
                                "composite_objective"
                            ]
                            - qobjective,
                            "elapsed_seconds": elapsed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    result = {
        "schema_version": "1.0",
        "status": "stage22_beam_baseline_diagnostic_complete",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "purpose": "Post-hoc structure-only comparison of the Stage 22 candidate with increasingly strong deterministic classical beam search.",
        "evidence_boundary": {
            "docking_scores_read": 0,
            "ligand_labels_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "beam_widths": beam_widths,
        "rows": rows,
        "interpretation_boundary": "A beam result equal to or better than the Stage 22 candidate weakens any claim of QUBO-specific classical superiority. A candidate that remains better than the tested beam widths is only a search signal, not quantum advantage or prospective docking benefit.",
    }
    output = rooted(root, output_path.as_posix())
    write_json(output, result)
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
        default=Path("data/stage22_beam_baseline_diagnostic.json"),
    )
    parser.add_argument("--beam-widths", nargs="+", type=int, default=None)
    args = parser.parse_args()
    run(args.config, args.root, args.output, args.beam_widths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
