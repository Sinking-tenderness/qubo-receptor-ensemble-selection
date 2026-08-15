"""Independently audit the Stage24 multiscale coverage QUBO."""

from __future__ import annotations

import argparse
import json
import math
import sys
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
)
from scripts.run_stage22_structural_state_coverage_qubo import (
    build_coverage_terms,
    qubo_energy,
    qubo_hash,
)
from scripts.run_stage24_multiscale_coverage_qubo import (
    assignment_for_subset,
    beam_search,
    build_multiscale_qubo,
    direct_greedy,
    improve_by_swaps,
    multiscale_components,
)


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} differs: {actual} != {expected}")


def parse_subset(value: str) -> tuple[str, ...]:
    subset = tuple(sorted(part for part in value.split("+") if part))
    if len(subset) != len(set(subset)):
        raise ValueError("duplicate conformer in subset")
    return subset


def audit(config_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    result_path = rooted(root, config["outputs"]["result_json"])
    result = read_json(result_path)
    if result.get("status") != "stage24_multiscale_coverage_qubo_complete":
        raise ValueError("unexpected Stage24 status")
    if result["config"]["sha256"] != file_sha256(config_path):
        raise ValueError("config hash differs")
    implementation = rooted(root, result["implementation"]["path"])
    if result["implementation"]["sha256"] != file_sha256(implementation):
        raise ValueError("implementation hash differs")
    paths: dict[str, Path] = {}
    for key, descriptor in result["outputs"].items():
        path = rooted(root, descriptor["path"])
        if descriptor["sha256"] != file_sha256(path):
            raise ValueError(f"output hash differs: {key}")
        paths[key] = path
    reads = read_csv(paths["read_csv"])
    batches = read_csv(paths["batch_csv"])
    baselines = read_csv(paths["baseline_csv"])
    model = read_json(paths["model_record_json"])
    objective = config["objective"]
    sampler = config["sampler"]
    gate = config["gate"]
    fractions = [float(value) for value in objective["neighborhood_fractions"]]
    weights = [float(value) for value in objective["scale_weights"]]
    target_k = int(objective["k"])
    expected_reads = len(config["targets"]) * int(sampler["batch_count"]) * int(sampler["reads_per_batch"])
    expected_batches = len(config["targets"]) * int(sampler["batch_count"])
    expected_baselines = len(config["targets"]) * (1 + len(config["classical_baselines"]["beam_widths"]))
    if (len(reads), len(batches), len(baselines)) != (expected_reads, expected_batches, expected_baselines):
        raise ValueError("Stage24 row counts differ")
    read_groups: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in reads:
        read_groups.setdefault((row["target_id"], int(row["batch"])), []).append(row)
    batch_index = {(row["target_id"], int(row["batch"])): row for row in batches}
    baseline_index = {
        (row["target_id"], row["method"], int(row["beam_width"])): row
        for row in baselines
    }
    checked_reads = 0
    checked_baselines = 0
    target_checks: dict[str, Any] = {}
    gate_values: list[bool] = []
    for target_id, spec in config["targets"].items():
        target = load_target(root, target_id, spec)
        ids = target["ids"]
        matrix = distance_matrix(ids, target["distances"])
        terms = [build_coverage_terms(ids, matrix, fraction) for fraction in fractions]
        diversity_weight = float(objective["diversity_weight"])
        greedy = direct_greedy(ids, matrix, terms, weights, target_k, diversity_weight)
        greedy_subset, greedy_metrics, _ = improve_by_swaps(
            greedy, ids, matrix, terms, weights, diversity_weight
        )
        expected = baseline_index[(target_id, "direct_greedy_plus_swap", 0)]
        if parse_subset(expected["selected_subset"]) != greedy_subset:
            raise ValueError(f"{target_id}: direct baseline subset differs")
        close(float(expected["composite_objective"]), float(greedy_metrics["composite_objective"]), f"{target_id}/direct")
        checked_baselines += 1
        baseline_values = [float(greedy_metrics["composite_objective"])]
        for width_value in config["classical_baselines"]["beam_widths"]:
            width = int(width_value)
            beam = beam_search(ids, matrix, terms, weights, target_k, diversity_weight, width)
            subset, metrics, _ = improve_by_swaps(
                beam, ids, matrix, terms, weights, diversity_weight
            )
            expected = baseline_index[(target_id, "beam_plus_swap", width)]
            if parse_subset(expected["selected_subset"]) != subset:
                raise ValueError(f"{target_id}/{width}: beam baseline subset differs")
            close(float(expected["composite_objective"]), float(metrics["composite_objective"]), f"{target_id}/{width}/beam")
            baseline_values.append(float(metrics["composite_objective"]))
            checked_baselines += 1
        strong = max(baseline_values)
        batch_values: list[float] = []
        batch_deltas: list[float] = []
        for batch in range(int(sampler["batch_count"])):
            group = read_groups[(target_id, batch)]
            if len(group) != int(sampler["reads_per_batch"]):
                raise ValueError(f"{target_id}/{batch}: read count differs")
            values: list[float] = []
            for row in group:
                subset = parse_subset(row["selected_subset"])
                if len(subset) != target_k or not set(subset).issubset(ids):
                    raise ValueError(f"{target_id}/{batch}: invalid subset")
                metrics = multiscale_components(
                    subset, ids, matrix, terms, weights, diversity_weight
                )
                close(float(row["composite_objective"]), float(metrics["composite_objective"]), f"{target_id}/{batch}/read")
                close(float(row["delta_vs_strong_classical"]), float(metrics["composite_objective"]) - strong, f"{target_id}/{batch}/delta")
                values.append(float(metrics["composite_objective"]))
                checked_reads += 1
            best = max(values)
            recorded = batch_index[(target_id, batch)]
            close(float(recorded["best_objective"]), best, f"{target_id}/{batch}/best")
            close(float(recorded["delta_vs_strong_classical"]), best - strong, f"{target_id}/{batch}/best-delta")
            batch_values.append(best)
            batch_deltas.append(best - strong)
        best = max(batch_values)
        within = sum(
            value >= best - float(gate["objective_tolerance"])
            for value in batch_values
        ) / len(batch_values)
        above = sum(
            delta > float(gate["minimum_gain_vs_strong_classical"])
            for delta in batch_deltas
        ) / len(batch_deltas)
        record = result["target_records"][target_id]
        close(float(record["strong_classical_objective"]), strong, f"{target_id}/strong")
        close(float(record["best_sampler_objective"]), best, f"{target_id}/sampler")
        close(float(record["within_tolerance_batch_fraction"]), within, f"{target_id}/stable")
        close(float(record["above_strong_classical_batch_fraction"]), above, f"{target_id}/winning")
        selected = tuple(record["best_sampler_subset"])
        qubo = build_multiscale_qubo(
            ids,
            matrix,
            terms,
            weights,
            target_k,
            diversity_weight,
            float(objective["cardinality_penalty"]),
            float(objective["coverage_constraint_penalty"]),
        )
        assignment = assignment_for_subset(selected, terms, qubo)
        metrics = multiscale_components(selected, ids, matrix, terms, weights, diversity_weight)
        close(qubo_energy(qubo, assignment), -float(metrics["composite_objective"]), f"{target_id}/qubo")
        if model["target_models"][target_id]["qubo_sha256"] != qubo_hash(qubo):
            raise ValueError(f"{target_id}: QUBO hash differs")
        target_pass = (
            within >= float(gate["minimum_batch_fraction_within_tolerance"])
            and above >= float(gate["minimum_batch_fraction_above_strong_classical"])
        )
        gate_values.append(target_pass)
        target_checks[target_id] = {
            "candidate_count": len(ids),
            "strong_classical_objective": strong,
            "best_sampler_objective": best,
            "within_tolerance_batch_fraction": within,
            "above_strong_classical_batch_fraction": above,
            "target_gate_passed": target_pass,
        }
    gate_passed = all(gate_values)
    if result["decision"]["multiscale_qubo_gate_passed"] != gate_passed:
        raise ValueError("gate decision differs")
    if any(int(value) != 0 for value in result["data_boundary"].values()):
        raise ValueError("nonzero data boundary")
    audit_result = {
        "schema_version": "1.0",
        "status": "stage24_multiscale_coverage_qubo_audit_ok",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "result": {"path": result_path.relative_to(root).as_posix(), "sha256": file_sha256(result_path)},
        "coverage": {
            "read_rows_recomputed": checked_reads,
            "batch_rows_recomputed": len(batches),
            "classical_baselines_recomputed": checked_baselines,
            "target_count": len(config["targets"]),
        },
        "target_checks": target_checks,
        "decision_recomputed": {"multiscale_qubo_gate_passed": gate_passed},
        "checks": {
            "all_output_hashes_verified": True,
            "all_read_objectives_recomputed": True,
            "all_classical_baselines_recomputed": True,
            "multiscale_qubo_hash_and_energy_recomputed": True,
            "data_boundary_zero": True,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
    }
    output = rooted(root, output_path.as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit_result, indent=2, sort_keys=True))
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage24_multiscale_coverage_qubo.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/stage24_multiscale_coverage_qubo_audit.json"))
    args = parser.parse_args()
    audit(args.config, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
