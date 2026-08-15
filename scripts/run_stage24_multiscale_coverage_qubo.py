"""Run the Stage24 equal-weight multiscale structural coverage QUBO."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    descriptor,
    distance_matrix,
    file_sha256,
    load_target,
    read_json,
    rooted,
    subset_metrics,
    write_csv,
    write_json,
)
from scripts.run_stage22_structural_state_coverage_qubo import (
    add_linear,
    add_quadratic,
    add_square,
    binary_assignment,
    build_coverage_terms,
    coefficient_stats,
    qubo_energy,
    qubo_hash,
    slack_weights,
)


def multiscale_components(
    subset: tuple[str, ...],
    ids: list[str],
    matrix: np.ndarray,
    terms: list[dict[str, Any]],
    weights: list[float],
    diversity_weight: float,
) -> dict[str, Any]:
    coverages: list[float] = []
    for scale_terms in terms:
        mask = 0
        for conformer_id in subset:
            mask |= int(scale_terms["coverage_masks"][conformer_id])
        coverages.append(mask.bit_count() / len(scale_terms["state_ids"]))
    structural = subset_metrics(subset, ids, matrix, ids[0])
    diversity = float(structural["mean_pair_distance_normalized"])
    weighted_coverage = sum(weight * value for weight, value in zip(weights, coverages))
    return {
        "scale_coverages": coverages,
        "weighted_coverage": weighted_coverage,
        "mean_pair_distance_normalized": diversity,
        "minimum_pair_distance_normalized": float(structural["minimum_pair_distance_normalized"]),
        "composite_objective": weighted_coverage + diversity_weight * diversity,
    }


def fast_multiscale_value(
    selected: list[int],
    masks_by_scale: list[list[int]],
    weights: list[float],
    matrix: np.ndarray,
    diversity_weight: float,
) -> float:
    coverage = 0.0
    for weight, masks in zip(weights, masks_by_scale):
        union = 0
        for index in selected:
            union |= masks[index]
        coverage += weight * union.bit_count() / len(masks)
    pair_sum = sum(
        float(matrix[first, second])
        for position, first in enumerate(selected)
        for second in selected[:position]
    )
    pair_count = max(1, len(selected) * (len(selected) - 1) // 2)
    return coverage + diversity_weight * pair_sum / pair_count


def objective_key(
    subset: tuple[str, ...],
    ids: list[str],
    matrix: np.ndarray,
    terms: list[dict[str, Any]],
    weights: list[float],
    diversity_weight: float,
) -> tuple[float, tuple[str, ...]]:
    value = multiscale_components(subset, ids, matrix, terms, weights, diversity_weight)
    return (-float(value["composite_objective"]), subset)


def direct_greedy(
    ids: list[str],
    matrix: np.ndarray,
    terms: list[dict[str, Any]],
    weights: list[float],
    target_k: int,
    diversity_weight: float,
) -> tuple[str, ...]:
    selected: tuple[str, ...] = ()
    for _ in range(target_k):
        selected = min(
            (
                tuple(sorted((*selected, conformer_id)))
                for conformer_id in ids
                if conformer_id not in selected
            ),
            key=lambda subset: objective_key(
                subset, ids, matrix, terms, weights, diversity_weight
            ),
        )
    return selected


def improve_by_swaps(
    start: tuple[str, ...],
    ids: list[str],
    matrix: np.ndarray,
    terms: list[dict[str, Any]],
    weights: list[float],
    diversity_weight: float,
) -> tuple[tuple[str, ...], dict[str, Any], int]:
    current = tuple(sorted(start))
    metrics = multiscale_components(current, ids, matrix, terms, weights, diversity_weight)
    iterations = 0
    while True:
        best, best_metrics = current, metrics
        selected = set(current)
        for outgoing in current:
            for incoming in ids:
                if incoming in selected:
                    continue
                candidate = tuple(sorted((selected - {outgoing}) | {incoming}))
                candidate_metrics = multiscale_components(
                    candidate, ids, matrix, terms, weights, diversity_weight
                )
                value = float(candidate_metrics["composite_objective"])
                best_value = float(best_metrics["composite_objective"])
                if value > best_value + 1e-12 or (
                    math.isclose(value, best_value, abs_tol=1e-12) and candidate < best
                ):
                    best, best_metrics = candidate, candidate_metrics
        if float(best_metrics["composite_objective"]) > float(metrics["composite_objective"]) + 1e-12:
            current, metrics = best, best_metrics
            iterations += 1
            continue
        return current, metrics, iterations


def beam_search(
    ids: list[str],
    matrix: np.ndarray,
    terms: list[dict[str, Any]],
    weights: list[float],
    target_k: int,
    diversity_weight: float,
    beam_width: int,
) -> tuple[str, ...]:
    states: list[tuple[str, ...]] = [()]
    for depth in range(target_k):
        expanded: set[tuple[str, ...]] = set()
        new_depth = depth + 1
        last_feasible = len(ids) - (target_k - new_depth) - 1
        for state in states:
            last_index = -1 if not state else ids.index(state[-1])
            for candidate_index in range(last_index + 1, last_feasible + 1):
                expanded.add(tuple((*state, ids[candidate_index])))
        states = sorted(
            expanded,
            key=lambda subset: objective_key(
                subset, ids, matrix, terms, weights, diversity_weight
            ),
        )[:beam_width]
    return states[0]


def anneal_read(
    candidate_count: int,
    target_k: int,
    masks_by_scale: list[list[int]],
    weights: list[float],
    matrix: np.ndarray,
    diversity_weight: float,
    sweeps: int,
    temperature_start: float,
    temperature_end: float,
    rng: random.Random,
) -> dict[str, Any]:
    selected = sorted(rng.sample(range(candidate_count), target_k))
    selected_set = set(selected)
    current = fast_multiscale_value(selected, masks_by_scale, weights, matrix, diversity_weight)
    best, best_subset, accepted = current, tuple(selected), 0
    for step in range(sweeps):
        outgoing_position = rng.randrange(target_k)
        outgoing = selected[outgoing_position]
        incoming = rng.randrange(candidate_count)
        while incoming in selected_set:
            incoming = rng.randrange(candidate_count)
        proposal = list(selected)
        proposal[outgoing_position] = incoming
        proposal.sort()
        proposed = fast_multiscale_value(proposal, masks_by_scale, weights, matrix, diversity_weight)
        progress = step / max(1, sweeps - 1)
        temperature = max(
            temperature_end,
            temperature_start * (temperature_end / temperature_start) ** progress,
        )
        delta = proposed - current
        if delta >= 0.0 or rng.random() < math.exp(max(-700.0, delta / temperature)):
            selected, current = proposal, proposed
            selected_set.remove(outgoing)
            selected_set.add(incoming)
            accepted += 1
            candidate = tuple(selected)
            if current > best + 1e-12 or (
                math.isclose(current, best, abs_tol=1e-12) and candidate < best_subset
            ):
                best, best_subset = current, candidate
    return {
        "best_indices": list(best_subset),
        "best_objective": best,
        "accepted_moves": accepted,
        "acceptance_fraction": accepted / max(1, sweeps),
    }


def build_multiscale_qubo(
    ids: list[str],
    matrix: np.ndarray,
    terms: list[dict[str, Any]],
    weights: list[float],
    target_k: int,
    diversity_weight: float,
    cardinality_penalty: float,
    constraint_penalty: float,
) -> dict[str, Any]:
    coefficients: dict[str, Any] = {"constant": 0.0, "linear": {}, "quadratic": {}}
    x_names = {conformer_id: f"x__{conformer_id}" for conformer_id in ids}
    add_square(coefficients, -target_k, {name: 1.0 for name in x_names.values()}, cardinality_penalty)
    groups: dict[str, list[str]] = {"x": sorted(x_names.values()), "coverage_y": [], "coverage_slack": []}
    for scale_index, (scale_terms, scale_weight) in enumerate(zip(terms, weights)):
        slack = slack_weights(int(scale_terms["neighbor_count"]) - 1)
        for state_id in scale_terms["state_ids"]:
            y_name = f"y__f{scale_index}__{state_id}"
            groups["coverage_y"].append(y_name)
            expression = {y_name: 1.0}
            for bit_index, bit_weight in enumerate(slack):
                name = f"s__f{scale_index}__{state_id}__{bit_index}"
                groups["coverage_slack"].append(name)
                expression[name] = float(bit_weight)
            for conformer_id in scale_terms["incidence"][state_id]:
                expression[x_names[conformer_id]] = -1.0
            add_square(coefficients, 0.0, expression, constraint_penalty)
            add_linear(
                coefficients,
                y_name,
                -scale_weight * float(scale_terms["state_weight"]),
            )
    denominator = max(1, target_k * (target_k - 1) // 2)
    for first in range(len(ids)):
        for second in range(first + 1, len(ids)):
            add_quadratic(
                coefficients,
                x_names[ids[first]],
                x_names[ids[second]],
                -diversity_weight * float(matrix[first, second]) / denominator,
            )
    variables = sorted(
        set(coefficients["linear"])
        | {
            variable
            for key in coefficients["quadratic"]
            for variable in key.split("::", 1)
        }
    )
    return {
        "constant": float(coefficients["constant"]),
        "linear": {key: float(value) for key, value in coefficients["linear"].items()},
        "quadratic": {key: float(value) for key, value in coefficients["quadratic"].items()},
        "variables": variables,
        "variable_groups": {key: sorted(set(value)) for key, value in groups.items()},
    }


def assignment_for_subset(
    subset: tuple[str, ...],
    terms: list[dict[str, Any]],
    qubo: dict[str, Any],
) -> dict[str, int]:
    selected = set(subset)
    assignment = {variable: 0 for variable in qubo["variables"]}
    for conformer_id in selected:
        assignment[f"x__{conformer_id}"] = 1
    for scale_index, scale_terms in enumerate(terms):
        weights = slack_weights(int(scale_terms["neighbor_count"]) - 1)
        for state_id in scale_terms["state_ids"]:
            count = sum(
                conformer_id in selected
                for conformer_id in scale_terms["incidence"][state_id]
            )
            covered = int(count > 0)
            assignment[f"y__f{scale_index}__{state_id}"] = covered
            for bit_index, bit in binary_assignment(weights, count - covered).items():
                assignment[f"s__f{scale_index}__{state_id}__{bit_index}"] = bit
    return assignment


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if config["evidence_timing"]["new_docking_jobs"]:
        raise ValueError("Stage24 cannot launch docking")
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage24 outputs exist; pass --overwrite")
    objective = config["objective"]
    sampler = config["sampler"]
    gate = config["gate"]
    fractions = [float(value) for value in objective["neighborhood_fractions"]]
    weights = [float(value) for value in objective["scale_weights"]]
    if len(fractions) != len(weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-12):
        raise ValueError("invalid multiscale weights")
    target_k = int(objective["k"])
    diversity_weight = float(objective["diversity_weight"])
    read_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    target_records: dict[str, Any] = {}
    model_records: dict[str, Any] = {}
    input_records: dict[str, Any] = {}
    for target_id, spec in config["targets"].items():
        target = load_target(root, target_id, spec)
        ids = target["ids"]
        matrix = distance_matrix(ids, target["distances"])
        terms = [build_coverage_terms(ids, matrix, fraction) for fraction in fractions]
        masks_by_scale = [
            [int(scale_terms["coverage_masks"][value]) for value in ids]
            for scale_terms in terms
        ]
        input_records[target_id] = {
            key: descriptor(root, path) for key, path in target["input_paths"].items()
        }
        greedy = direct_greedy(ids, matrix, terms, weights, target_k, diversity_weight)
        greedy_refined, greedy_metrics, greedy_iterations = improve_by_swaps(
            greedy, ids, matrix, terms, weights, diversity_weight
        )
        baseline_rows.append(
            {
                "target_id": target_id,
                "method": "direct_greedy_plus_swap",
                "beam_width": 0,
                "selected_subset": "+".join(greedy_refined),
                "composite_objective": greedy_metrics["composite_objective"],
                "weighted_coverage": greedy_metrics["weighted_coverage"],
                "mean_pair_distance_normalized": greedy_metrics["mean_pair_distance_normalized"],
                "refinement_iterations": greedy_iterations,
            }
        )
        for width in config["classical_baselines"]["beam_widths"]:
            started = time.perf_counter()
            beam = beam_search(ids, matrix, terms, weights, target_k, diversity_weight, int(width))
            refined, metrics, iterations = improve_by_swaps(
                beam, ids, matrix, terms, weights, diversity_weight
            )
            baseline_rows.append(
                {
                    "target_id": target_id,
                    "method": "beam_plus_swap",
                    "beam_width": int(width),
                    "selected_subset": "+".join(refined),
                    "composite_objective": metrics["composite_objective"],
                    "weighted_coverage": metrics["weighted_coverage"],
                    "mean_pair_distance_normalized": metrics["mean_pair_distance_normalized"],
                    "refinement_iterations": iterations,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        target_baselines = [row for row in baseline_rows if row["target_id"] == target_id]
        strong = max(target_baselines, key=lambda row: float(row["composite_objective"]))
        target_batches: list[dict[str, Any]] = []
        for batch in range(int(sampler["batch_count"])):
            seed = int(sampler["base_seed"]) + sum(ord(value) for value in target_id) + batch * 1000003
            rng = random.Random(seed)
            local_reads: list[dict[str, Any]] = []
            for read_index in range(int(sampler["reads_per_batch"])):
                sample = anneal_read(
                    len(ids), target_k, masks_by_scale, weights, matrix, diversity_weight,
                    int(sampler["sweeps_per_read"]), float(sampler["temperature_start"]),
                    float(sampler["temperature_end"]), rng,
                )
                subset = tuple(ids[index] for index in sample["best_indices"])
                metrics = multiscale_components(subset, ids, matrix, terms, weights, diversity_weight)
                row = {
                    "target_id": target_id,
                    "batch": batch,
                    "read": read_index,
                    "seed": seed,
                    "selected_subset": "+".join(subset),
                    "scale_coverages": json.dumps(metrics["scale_coverages"], separators=(",", ":")),
                    "weighted_coverage": metrics["weighted_coverage"],
                    "mean_pair_distance_normalized": metrics["mean_pair_distance_normalized"],
                    "minimum_pair_distance_normalized": metrics["minimum_pair_distance_normalized"],
                    "composite_objective": metrics["composite_objective"],
                    "delta_vs_strong_classical": metrics["composite_objective"] - float(strong["composite_objective"]),
                    "accepted_moves": sample["accepted_moves"],
                    "acceptance_fraction": sample["acceptance_fraction"],
                }
                read_rows.append(row)
                local_reads.append(row)
            best = max(local_reads, key=lambda row: (float(row["composite_objective"]), row["selected_subset"]))
            batch_record = {
                "target_id": target_id,
                "batch": batch,
                "seed": seed,
                "best_subset": best["selected_subset"],
                "best_objective": float(best["composite_objective"]),
                "delta_vs_strong_classical": float(best["delta_vs_strong_classical"]),
                "within_batch_best_read_count": sum(
                    math.isclose(float(row["composite_objective"]), float(best["composite_objective"]), abs_tol=1e-12)
                    for row in local_reads
                ),
            }
            batch_rows.append(batch_record)
            target_batches.append(batch_record)
            print(json.dumps(batch_record, sort_keys=True), flush=True)
        best_objective = max(float(row["best_objective"]) for row in target_batches)
        within = sum(
            float(row["best_objective"]) >= best_objective - float(gate["objective_tolerance"])
            for row in target_batches
        )
        above = sum(
            float(row["delta_vs_strong_classical"]) > float(gate["minimum_gain_vs_strong_classical"])
            for row in target_batches
        )
        best_batch = max(target_batches, key=lambda row: float(row["best_objective"]))
        selected = tuple(best_batch["best_subset"].split("+"))
        selected_metrics = multiscale_components(selected, ids, matrix, terms, weights, diversity_weight)
        qubo = build_multiscale_qubo(
            ids, matrix, terms, weights, target_k, diversity_weight,
            float(objective["cardinality_penalty"]), float(objective["coverage_constraint_penalty"]),
        )
        energy = qubo_energy(qubo, assignment_for_subset(selected, terms, qubo))
        residual = abs(energy + float(selected_metrics["composite_objective"]))
        if residual > 1e-7:
            raise ValueError("multiscale QUBO does not match reduced objective")
        target_records[target_id] = {
            "candidate_count": len(ids),
            "hard_gate_excluded_count": len(target["excluded_hard_gate"]),
            "strong_classical_method": strong["method"],
            "strong_classical_beam_width": int(strong["beam_width"]),
            "strong_classical_objective": float(strong["composite_objective"]),
            "best_sampler_objective": best_objective,
            "best_sampler_subset": list(selected),
            "best_sampler_scale_coverages": selected_metrics["scale_coverages"],
            "within_tolerance_batch_fraction": within / len(target_batches),
            "above_strong_classical_batch_fraction": above / len(target_batches),
            "delta_vs_strong_classical": best_objective - float(strong["composite_objective"]),
        }
        model_records[target_id] = {
            "qubo_sha256": qubo_hash(qubo),
            "variable_count": len(qubo["variables"]),
            "x_count": len(qubo["variable_groups"]["x"]),
            "coverage_y_count": len(qubo["variable_groups"]["coverage_y"]),
            "coverage_slack_count": len(qubo["variable_groups"]["coverage_slack"]),
            "selected_energy": energy,
            "equivalence_residual": residual,
            **coefficient_stats(qubo),
        }
    gate_passed = all(
        record["within_tolerance_batch_fraction"] >= float(gate["minimum_batch_fraction_within_tolerance"])
        and record["above_strong_classical_batch_fraction"] >= float(gate["minimum_batch_fraction_above_strong_classical"])
        for record in target_records.values()
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
        "status": "stage24_multiscale_coverage_qubo_model_record",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "target_models": model_records,
        "data_boundary": boundary,
    }
    write_json(outputs["model_record_json"], model_record)
    report = [
        "# Stage 24: multiscale structural coverage QUBO",
        "",
        "Frozen objective: equal-weight coverage at 5%, 10%, and 20% structural neighborhoods plus 0.15 mean pairwise diversity.",
        "",
        "| Target | Sampler | Strong classical | Delta | Stable batch fraction | Winning batch fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target_id, record in target_records.items():
        report.append(
            f"| {target_id} | {record['best_sampler_objective']:.6f} | {record['strong_classical_objective']:.6f} | {record['delta_vs_strong_classical']:.6f} | {record['within_tolerance_batch_fraction']:.2f} | {record['above_strong_classical_batch_fraction']:.2f} |"
        )
    report.extend(
        [
            "",
            f"Multiscale QUBO gate passed: `{str(gate_passed).lower()}`.",
            "Passing authorizes only preparation of a separate small matched Uni-Dock preregistration.",
            "",
            config["interpretation_boundary"],
        ]
    )
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report) + "\n", encoding="ascii")
    result = {
        "schema_version": "1.0",
        "status": "stage24_multiscale_coverage_qubo_complete",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "implementation": {"path": Path(__file__).resolve().relative_to(root).as_posix(), "sha256": file_sha256(Path(__file__).resolve())},
        "inputs": input_records,
        "target_records": target_records,
        "decision": {
            "multiscale_qubo_gate_passed": gate_passed,
            "matched_small_docking_preregistration_authorized": gate_passed,
            "new_docking_jobs_authorized_by_this_stage": False,
            "quantum_hardware_authorized": False,
        },
        "data_boundary": boundary,
        "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key != "result_json"},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(json.dumps({"status": result["status"], "decision": result["decision"], "target_records": target_records}, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage24_multiscale_coverage_qubo.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
