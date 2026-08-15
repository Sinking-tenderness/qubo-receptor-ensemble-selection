"""Run independent fixed-cardinality simulated-annealing QUBO reads."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    descriptor,
    distance_matrix,
    file_sha256,
    load_target,
    read_csv,
    read_json,
    rooted,
    write_csv,
    write_json,
)
from scripts.run_stage22_structural_state_coverage_qubo import (
    build_coverage_terms,
    direct_greedy,
    objective_components,
)


def fast_value(
    selected: list[int],
    masks: list[int],
    matrix: np.ndarray,
    diversity_weight: float,
) -> float:
    covered = 0
    pair_sum = 0.0
    for position, index in enumerate(selected):
        covered |= masks[index]
        for other in selected[:position]:
            pair_sum += float(matrix[index, other])
    pair_count = max(1, len(selected) * (len(selected) - 1) // 2)
    return covered.bit_count() / len(masks) + diversity_weight * pair_sum / pair_count


def anneal_read(
    candidate_count: int,
    target_k: int,
    masks: list[int],
    matrix: np.ndarray,
    diversity_weight: float,
    sweeps: int,
    temperature_start: float,
    temperature_end: float,
    rng: random.Random,
) -> dict[str, Any]:
    selected = sorted(rng.sample(range(candidate_count), target_k))
    selected_set = set(selected)
    current = fast_value(selected, masks, matrix, diversity_weight)
    best = current
    best_subset = tuple(selected)
    accepted = 0
    for step in range(sweeps):
        outgoing_position = rng.randrange(target_k)
        outgoing = selected[outgoing_position]
        incoming = rng.randrange(candidate_count)
        while incoming in selected_set:
            incoming = rng.randrange(candidate_count)
        proposal = list(selected)
        proposal[outgoing_position] = incoming
        proposal.sort()
        proposed = fast_value(proposal, masks, matrix, diversity_weight)
        progress = step / max(1, sweeps - 1)
        temperature = max(
            temperature_end,
            temperature_start
            * (temperature_end / temperature_start) ** progress,
        )
        delta = proposed - current
        if delta >= 0.0 or rng.random() < math.exp(max(-700.0, delta / temperature)):
            selected = proposal
            selected_set.remove(outgoing)
            selected_set.add(incoming)
            current = proposed
            accepted += 1
            candidate = tuple(selected)
            if current > best + 1e-12 or (
                math.isclose(current, best, abs_tol=1e-12) and candidate < best_subset
            ):
                best = current
                best_subset = candidate
    return {
        "best_indices": list(best_subset),
        "best_objective": best,
        "accepted_moves": accepted,
        "acceptance_fraction": accepted / max(1, sweeps),
    }


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if config["evidence_timing"]["new_docking_jobs"]:
        raise ValueError("Stage23 cannot launch docking")
    outputs = {
        key: rooted(root, value) for key, value in config["outputs"].items()
    }
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage23 outputs exist; pass --overwrite")
    objective_config = config["objective"]
    sampler_config = config["sampler"]
    gate_config = config["gate"]
    baseline = read_json(rooted(root, config["baselines"]["strong_classical_diagnostic"]))
    if baseline.get("status") != "stage22_beam_baseline_diagnostic_complete":
        raise ValueError("unexpected strong-classical baseline status")
    fractions = [float(value) for value in objective_config["neighborhood_fractions"]]
    primary_fraction = float(objective_config["primary_neighborhood_fraction"])
    target_k = int(objective_config["k"])
    diversity_weight = float(objective_config["diversity_weight"])
    batch_count = int(sampler_config["batch_count"])
    reads_per_batch = int(sampler_config["reads_per_batch"])
    reads: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    target_records: dict[str, Any] = {}
    for target_id, target_spec in config["targets"].items():
        target = load_target(root, target_id, target_spec)
        ids = target["ids"]
        matrix = distance_matrix(ids, target["distances"])
        target_records[target_id] = {
            "candidate_count": len(ids),
            "hard_gate_excluded_count": len(target["excluded_hard_gate"]),
            "fractions": {},
        }
        for fraction in fractions:
            terms = build_coverage_terms(ids, matrix, fraction)
            masks = [int(terms["coverage_masks"][value]) for value in ids]
            classical_rows = [
                row
                for row in baseline["rows"]
                if row["target_id"] == target_id
                and math.isclose(
                    float(row["neighborhood_fraction"]), fraction, abs_tol=1e-12
                )
            ]
            strong_value = max(
                float(row["refined_composite_objective"]) for row in classical_rows
            )
            greedy = direct_greedy(ids, matrix, terms, target_k, diversity_weight)
            greedy_metrics = objective_components(
                greedy, ids, matrix, terms, diversity_weight
            )
            fraction_batches: list[dict[str, Any]] = []
            for batch in range(batch_count):
                seed = (
                    int(sampler_config["base_seed"])
                    + sum(ord(value) for value in target_id)
                    + int(round(fraction * 10000))
                    + batch * 1000003
                )
                rng = random.Random(seed)
                batch_rows: list[dict[str, Any]] = []
                for read_index in range(reads_per_batch):
                    read = anneal_read(
                        len(ids),
                        target_k,
                        masks,
                        matrix,
                        diversity_weight,
                        int(sampler_config["sweeps_per_read"]),
                        float(sampler_config["temperature_start"]),
                        float(sampler_config["temperature_end"]),
                        rng,
                    )
                    subset = tuple(ids[index] for index in read["best_indices"])
                    metrics = objective_components(
                        subset, ids, matrix, terms, diversity_weight
                    )
                    row = {
                        "target_id": target_id,
                        "neighborhood_fraction": fraction,
                        "batch": batch,
                        "read": read_index,
                        "seed": seed,
                        "selected_subset": "+".join(subset),
                        **metrics,
                        "delta_vs_direct_greedy": metrics["composite_objective"]
                        - greedy_metrics["composite_objective"],
                        "delta_vs_strong_classical": metrics["composite_objective"]
                        - strong_value,
                        **{
                            key: value
                            for key, value in read.items()
                            if key != "best_indices"
                        },
                    }
                    reads.append(row)
                    batch_rows.append(row)
                batch_best = max(
                    batch_rows, key=lambda row: (float(row["composite_objective"]), row["selected_subset"])
                )
                batch_record = {
                    "target_id": target_id,
                    "neighborhood_fraction": fraction,
                    "batch": batch,
                    "seed": seed,
                    "best_subset": batch_best["selected_subset"],
                    "best_objective": float(batch_best["composite_objective"]),
                    "delta_vs_direct_greedy": float(
                        batch_best["delta_vs_direct_greedy"]
                    ),
                    "delta_vs_strong_classical": float(
                        batch_best["delta_vs_strong_classical"]
                    ),
                    "within_batch_best_read_count": sum(
                        math.isclose(
                            float(row["composite_objective"]),
                            float(batch_best["composite_objective"]),
                            abs_tol=1e-12,
                        )
                        for row in batch_rows
                    ),
                }
                batches.append(batch_record)
                fraction_batches.append(batch_record)
                print(
                    json.dumps(
                        {
                            "target_id": target_id,
                            "neighborhood_fraction": fraction,
                            "batch": batch,
                            "best_objective": batch_record["best_objective"],
                            "delta_vs_strong_classical": batch_record[
                                "delta_vs_strong_classical"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            best_objective = max(float(row["best_objective"]) for row in fraction_batches)
            tolerance = float(gate_config["objective_tolerance"])
            within_tolerance = sum(
                float(row["best_objective"]) >= best_objective - tolerance
                for row in fraction_batches
            )
            above_strong = sum(
                float(row["delta_vs_strong_classical"])
                > float(gate_config["minimum_gain_vs_strong_classical"])
                for row in fraction_batches
            )
            fraction_record = {
                "strong_classical_objective": strong_value,
                "direct_greedy_objective": greedy_metrics["composite_objective"],
                "batch_count": batch_count,
                "reads_per_batch": reads_per_batch,
                "best_batch_objective": best_objective,
                "within_tolerance_batch_count": within_tolerance,
                "within_tolerance_batch_fraction": within_tolerance / batch_count,
                "above_strong_classical_batch_count": above_strong,
                "above_strong_classical_batch_fraction": above_strong / batch_count,
                "batch_records": fraction_batches,
            }
            target_records[target_id]["fractions"][f"{fraction:.6f}"] = fraction_record
    sensitivity: dict[str, Any] = {}
    for target_id in sorted(config["targets"]):
        target_fractions = target_records[target_id]["fractions"]
        positive = sum(
            target_fractions[f"{fraction:.6f}"]["above_strong_classical_batch_fraction"]
            >= float(gate_config["minimum_batch_fraction_above_strong_classical"])
            for fraction in fractions
        )
        sensitivity[target_id] = {
            "positive_fraction_count": positive,
            "fraction_count": len(fractions),
            "passed": positive
            >= int(gate_config["minimum_positive_sensitivity_fractions"]),
        }
    primary_pass = all(
        target_records[target_id]["fractions"][f"{primary_fraction:.6f}"][
            "within_tolerance_batch_fraction"
        ]
        >= float(gate_config["minimum_batch_fraction_within_tolerance"])
        and target_records[target_id]["fractions"][f"{primary_fraction:.6f}"][
            "above_strong_classical_batch_fraction"
        ]
        >= float(gate_config["minimum_batch_fraction_above_strong_classical"])
        for target_id in config["targets"]
    )
    gate_passed = primary_pass and all(value["passed"] for value in sensitivity.values())
    boundary = {
        "docking_scores_read": 0,
        "ligand_labels_read": 0,
        "fresh_validation_rows_read": 0,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
    write_csv(outputs["read_csv"], reads)
    write_csv(outputs["batch_csv"], batches)
    report_lines = [
        "# Stage 23: QUBO sampler stability",
        "",
        "Post-hoc structure-only diagnostic using independent fixed-cardinality simulated-annealing reads.",
        "",
        f"Frozen objective: `coverage + {diversity_weight:.2f} * mean structural diversity`; `k={target_k}`; primary neighborhood fraction `{primary_fraction:.2f}`.",
        "",
        "| Target | Fraction | Best batch objective | Strong classical | Within tolerance | Above strong classical |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target_id in sorted(target_records):
        for fraction in fractions:
            record = target_records[target_id]["fractions"][f"{fraction:.6f}"]
            report_lines.append(
                f"| {target_id} | {fraction:.2f} | {record['best_batch_objective']:.6f} | {record['strong_classical_objective']:.6f} | {record['within_tolerance_batch_fraction']:.2f} | {record['above_strong_classical_batch_fraction']:.2f} |"
            )
    report_lines.extend(
        [
            "",
            f"Sampler stability gate passed: `{str(gate_passed).lower()}`.",
            "A pass authorizes only a separate small matched Uni-Dock preregistration; it does not establish quantum advantage or authorize hardware.",
            "",
            config["interpretation_boundary"],
        ]
    )
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report_lines) + "\n", encoding="ascii")
    result = {
        "schema_version": "1.0",
        "status": "stage23_qubo_sampler_stability_complete",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "implementation": {
            "path": Path(__file__).resolve().relative_to(root).as_posix(),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "strong_classical_baseline": descriptor(
            root,
            rooted(root, config["baselines"]["strong_classical_diagnostic"]),
        ),
        "inputs": {
            target_id: {
                key: descriptor(root, path)
                for key, path in load_target(root, target_id, spec)["input_paths"].items()
            }
            for target_id, spec in config["targets"].items()
        },
        "target_records": target_records,
        "decision": {
            "primary_pass": primary_pass,
            "sensitivity": sensitivity,
            "sampler_stability_gate_passed": gate_passed,
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
    print(json.dumps({"status": result["status"], "decision": result["decision"]}, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage23_qubo_sampler_stability.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
