"""Run the frozen Stage23 single-scale QUBO on the independent BACE1 pool."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_stage22_beam_baseline import beam_search
from scripts.run_stage21_structure_aware_qubo import (
    descriptor,
    distance_matrix,
    file_sha256,
    load_target,
    read_json,
    rooted,
    write_csv,
    write_json,
)
from scripts.run_stage22_structural_state_coverage_qubo import (
    assignment_for_subset,
    build_auxiliary_qubo,
    build_coverage_terms,
    coefficient_stats,
    direct_greedy,
    improve_by_swaps,
    objective_components,
    qubo_energy,
    qubo_hash,
)
from scripts.run_stage23_qubo_sampler_stability import anneal_read


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if config["evidence_timing"]["new_docking_jobs"]:
        raise ValueError("Stage25 cannot launch docking")
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage25 outputs exist; pass --overwrite")
    target_spec = {
        "reference_id": config["target"]["reference_id"],
        "inputs": config["target"]["inputs"],
    }
    target_id = str(config["target"]["target_id"])
    target = load_target(root, target_id, target_spec)
    ids = target["ids"]
    matrix = distance_matrix(ids, target["distances"])
    objective = config["objective"]
    sampler = config["sampler"]
    gate = config["gate"]
    target_k = int(objective["k"])
    fraction = float(objective["neighborhood_fraction"])
    diversity_weight = float(objective["diversity_weight"])
    terms = build_coverage_terms(ids, matrix, fraction)
    masks = [int(terms["coverage_masks"][value]) for value in ids]

    baseline_rows: list[dict[str, Any]] = []
    greedy = direct_greedy(ids, matrix, terms, target_k, diversity_weight)
    greedy_subset, greedy_metrics, greedy_iterations = improve_by_swaps(
        greedy, ids, matrix, terms, target_k, diversity_weight
    )
    baseline_rows.append(
        {
            "method": "direct_greedy_plus_swap",
            "beam_width": 0,
            "selected_subset": "+".join(greedy_subset),
            **greedy_metrics,
            "refinement_iterations": greedy_iterations,
        }
    )
    for width_value in config["classical_baselines"]["beam_widths"]:
        width = int(width_value)
        started = time.perf_counter()
        beam, _ = beam_search(
            ids, matrix, terms, target_k, diversity_weight, width
        )
        subset, metrics, iterations = improve_by_swaps(
            beam, ids, matrix, terms, target_k, diversity_weight
        )
        baseline_rows.append(
            {
                "method": "beam_plus_swap",
                "beam_width": width,
                "selected_subset": "+".join(subset),
                **metrics,
                "refinement_iterations": iterations,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    strong = max(
        baseline_rows,
        key=lambda row: (float(row["composite_objective"]), row["selected_subset"]),
    )

    read_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for batch in range(int(sampler["batch_count"])):
        seed = int(sampler["base_seed"]) + sum(ord(value) for value in target_id) + batch * 1000003
        rng = random.Random(seed)
        local_reads: list[dict[str, Any]] = []
        for read_index in range(int(sampler["reads_per_batch"])):
            sample = anneal_read(
                len(ids),
                target_k,
                masks,
                matrix,
                diversity_weight,
                int(sampler["sweeps_per_read"]),
                float(sampler["temperature_start"]),
                float(sampler["temperature_end"]),
                rng,
            )
            subset = tuple(ids[index] for index in sample["best_indices"])
            metrics = objective_components(
                subset, ids, matrix, terms, diversity_weight
            )
            row = {
                "target_id": target_id,
                "batch": batch,
                "read": read_index,
                "seed": seed,
                "selected_subset": "+".join(subset),
                **metrics,
                "delta_vs_strong_classical": metrics["composite_objective"]
                - float(strong["composite_objective"]),
                "accepted_moves": sample["accepted_moves"],
                "acceptance_fraction": sample["acceptance_fraction"],
            }
            read_rows.append(row)
            local_reads.append(row)
        best = max(
            local_reads,
            key=lambda row: (float(row["composite_objective"]), row["selected_subset"]),
        )
        batch_record = {
            "target_id": target_id,
            "batch": batch,
            "seed": seed,
            "best_subset": best["selected_subset"],
            "best_objective": float(best["composite_objective"]),
            "delta_vs_strong_classical": float(best["delta_vs_strong_classical"]),
            "within_batch_best_read_count": sum(
                math.isclose(
                    float(row["composite_objective"]),
                    float(best["composite_objective"]),
                    abs_tol=1e-12,
                )
                for row in local_reads
            ),
        }
        batch_rows.append(batch_record)
        print(json.dumps(batch_record, sort_keys=True), flush=True)

    best_objective = max(float(row["best_objective"]) for row in batch_rows)
    tolerance = float(gate["objective_tolerance"])
    within = sum(
        float(row["best_objective"]) >= best_objective - tolerance
        for row in batch_rows
    )
    above = sum(
        float(row["delta_vs_strong_classical"])
        > float(gate["minimum_gain_vs_strong_classical"])
        for row in batch_rows
    )
    best_batch = max(
        batch_rows,
        key=lambda row: (float(row["best_objective"]), row["best_subset"]),
    )
    selected = tuple(best_batch["best_subset"].split("+"))
    selected_metrics = objective_components(
        selected, ids, matrix, terms, diversity_weight
    )
    qubo = build_auxiliary_qubo(
        ids,
        matrix,
        terms,
        target_k,
        diversity_weight,
        float(objective["cardinality_penalty"]),
        float(objective["coverage_constraint_penalty"]),
    )
    energy = qubo_energy(qubo, assignment_for_subset(selected, terms, qubo))
    residual = abs(energy + float(selected_metrics["composite_objective"]))
    if residual > 1e-7:
        raise ValueError("BACE1 QUBO does not match frozen reduced objective")
    within_fraction = within / len(batch_rows)
    above_fraction = above / len(batch_rows)
    gate_passed = (
        within_fraction
        >= float(gate["minimum_batch_fraction_within_tolerance"])
        and above_fraction
        >= float(gate["minimum_batch_fraction_above_strong_classical"])
    )
    write_csv(outputs["read_csv"], read_rows)
    write_csv(outputs["batch_csv"], batch_rows)
    write_csv(outputs["baseline_csv"], baseline_rows)
    boundary = {
        "docking_scores_read": 0,
        "ligand_labels_read": 0,
        "fresh_validation_rows_read": 0,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
    model_record = {
        "schema_version": "1.0",
        "status": "stage25_bace1_prospective_structure_replication_model_record",
        "qubo": {
            "sha256": qubo_hash(qubo),
            "variable_count": len(qubo["variables"]),
            "x_count": len(qubo["variable_groups"]["x"]),
            "state_y_count": len(qubo["variable_groups"]["state_y"]),
            "state_slack_count": len(qubo["variable_groups"]["state_slack"]),
            "selected_energy": energy,
            "equivalence_residual": residual,
            **coefficient_stats(qubo),
        },
        "data_boundary": boundary,
    }
    write_json(outputs["model_record_json"], model_record)
    report = [
        "# Stage 25: BACE1 prospective structural replication",
        "",
        "The Stage23 single-scale objective and sampler were frozen before reading any Stage25 objective outcome.",
        "",
        "| Candidate pool | Sampler | Strong classical | Delta | Stable batch fraction | Winning batch fraction |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {len(ids)} | {best_objective:.6f} | {float(strong['composite_objective']):.6f} | {best_objective - float(strong['composite_objective']):.6f} | {within_fraction:.2f} | {above_fraction:.2f} |",
        "",
        f"Prospective structural replication gate passed: `{str(gate_passed).lower()}`.",
        "Passing authorizes only preparation of a separate small matched Uni-Dock preregistration.",
        "",
        config["interpretation_boundary"],
    ]
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report) + "\n", encoding="ascii")
    input_records = {
        key: descriptor(root, path) for key, path in target["input_paths"].items()
    }
    result = {
        "schema_version": "1.0",
        "status": "stage25_bace1_prospective_structure_replication_complete",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "implementation": {
            "path": Path(__file__).resolve().relative_to(root).as_posix(),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "inputs": input_records,
        "target_record": {
            "target_id": target_id,
            "candidate_count": len(ids),
            "hard_gate_excluded_count": len(target["excluded_hard_gate"]),
            "strong_classical_method": strong["method"],
            "strong_classical_beam_width": int(strong["beam_width"]),
            "strong_classical_objective": float(strong["composite_objective"]),
            "best_sampler_objective": best_objective,
            "best_sampler_subset": list(selected),
            "delta_vs_strong_classical": best_objective
            - float(strong["composite_objective"]),
            "within_tolerance_batch_fraction": within_fraction,
            "above_strong_classical_batch_fraction": above_fraction,
        },
        "decision": {
            "prospective_structure_replication_gate_passed": gate_passed,
            "matched_small_docking_preregistration_authorized": gate_passed,
            "new_docking_jobs_authorized_by_this_stage": False,
            "quantum_hardware_authorized": False,
        },
        "data_boundary": boundary,
        "outputs": {
            key: descriptor(root, path)
            for key, path in outputs.items()
            if key != "result_json"
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "target_record": result["target_record"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage25_bace1_prospective_structure_replication.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
