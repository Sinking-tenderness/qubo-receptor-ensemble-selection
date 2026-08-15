"""Diagnose why the frozen BACE1 coverage QUBO is misaligned with BEDROC20."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from scripts.run_stage42d_bace1_large_pool_qubo_screen import (
    BitsetObjective,
    bedroc_metrics,
    build_score_cube,
    descriptor,
    rank_cube,
    read_csv,
    read_json,
    sha256,
    verified,
    write_csv,
    write_json,
)


def finite_spearman(first: list[float], second: list[float]) -> float:
    if np.ptp(first) == 0.0 or np.ptp(second) == 0.0:
        return 0.0
    value = float(spearmanr(first, second).statistic)
    return value if math.isfinite(value) else 0.0


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 42e implementation identity differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    stage42d = read_json(inputs["stage42d_result"])
    if stage42d.get("status") != "stage42d_bace1_large_pool_qubo_screen_complete":
        raise ValueError("Stage 42d source screen is incomplete")
    if stage42d["decision"]["frozen_objective_supported_on_bace1"]:
        raise ValueError("Stage 42e diagnosis is only authorized after the frozen objective fails")
    outputs = {key: root / value for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage 42e outputs exist; pass --overwrite")

    ligands = read_csv(inputs["ligand_manifest"])
    receptors = read_csv(inputs["receptor_manifest"])
    score_rows = read_csv(inputs["scores"])
    ligand_ids = [row["ligand_id"] for row in ligands]
    receptor_ids = [row["conformer_id"] for row in receptors]
    labels = np.asarray([int(row["label"] == "active") for row in ligands], dtype=int)
    scores = build_score_cube(score_rows, ligand_ids, receptor_ids)
    ranks = rank_cube(scores, np.ones(len(ligands), dtype=bool))
    objective = stage42d["objective"]
    scorer = BitsetObjective(
        ranks <= float(objective["favorable_rank_fraction"]), labels, objective
    )

    generator = random.Random(int(config["sampling"]["seed"]))
    samples_per_size = int(config["sampling"]["subsets_per_size"])
    sampled: list[dict[str, Any]] = []
    for size in range(1, 7):
        target = min(samples_per_size, math.comb(len(receptor_ids), size))
        subsets: set[tuple[int, ...]] = set()
        while len(subsets) < target:
            subsets.add(tuple(sorted(generator.sample(range(len(receptor_ids)), size))))
        for subset in sorted(subsets):
            value, components = scorer.score(subset)
            metrics = bedroc_metrics(ranks, labels, subset, float(config["bedroc_alpha"]))
            sampled.append({
                "subset_size": size,
                "subset": "+".join(receptor_ids[index] for index in subset),
                "objective": value,
                **components,
                **metrics,
            })
        print(json.dumps({"subset_size": size, "sample_count": target}), flush=True)

    component_keys = [
        "objective",
        "active_majority_seed_coverage",
        "active_all_seed_coverage",
        "active_double_receptor_majority_seed_support",
        "decoy_any_seed_exposure",
    ]
    correlation_rows: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {"pooled": sampled}
    for size in range(1, 7):
        groups[f"k{size}"] = [row for row in sampled if row["subset_size"] == size]
    for group_id, rows in groups.items():
        bedroc = [float(row["robust_bedroc_composite"]) for row in rows]
        for component in component_keys:
            correlation_rows.append({
                "group_id": group_id,
                "component": component,
                "sample_count": len(rows),
                "spearman_with_robust_bedroc20": finite_spearman(
                    [float(row[component]) for row in rows], bedroc
                ),
            })

    stage42d_rows = read_csv(inputs["stage42d_selection_metrics"])
    exact_by_size = sorted(
        [row for row in stage42d_rows if row["method"].startswith("exact_qubo_objective_k")],
        key=lambda row: int(row["subset_size"]),
    )
    objectives = [float(row["objective"]) for row in exact_by_size]
    bedroc_values = [float(row["robust_bedroc_composite"]) for row in exact_by_size]
    objective_monotone = all(right > left for left, right in zip(objectives, objectives[1:]))
    bedroc_peak_index = int(np.argmax(bedroc_values))
    pooled_objective_correlation = next(
        float(row["spearman_with_robust_bedroc20"])
        for row in correlation_rows
        if row["group_id"] == "pooled" and row["component"] == "objective"
    )
    fixed_k_correlations = [
        float(row["spearman_with_robust_bedroc20"])
        for row in correlation_rows
        if row["group_id"] != "pooled" and row["component"] == "objective"
    ]
    cardinality_pressure = objective_monotone and bedroc_peak_index < len(exact_by_size) - 1
    rank_misaligned = (
        pooled_objective_correlation
        < float(config["diagnostic_gate"]["maximum_supported_pooled_spearman"])
        or int(exact_by_size[bedroc_peak_index]["subset_size"])
        != int(exact_by_size[int(np.argmax(objectives))]["subset_size"])
    )
    diagnosis = {
        "objective_monotonically_increases_k1_to_k6": objective_monotone,
        "bedroc_optimal_k": int(exact_by_size[bedroc_peak_index]["subset_size"]),
        "objective_optimal_k": int(exact_by_size[int(np.argmax(objectives))]["subset_size"]),
        "pooled_objective_bedroc_spearman": pooled_objective_correlation,
        "median_fixed_k_objective_bedroc_spearman": float(np.median(fixed_k_correlations)),
        "binary_top10_coverage_is_rank_misaligned": rank_misaligned,
        "cardinality_pressure_detected": cardinality_pressure,
        "old_objective_retuning_authorized": False,
        "analytically_weighted_rank_sensitive_redesign_authorized": True,
        "fresh_validation_authorized": False,
        "quantum_hardware_authorized": False,
    }

    write_csv(outputs["sample_metrics_csv"], sampled)
    write_csv(outputs["correlations_csv"], correlation_rows)
    report = [
        "# Stage42e BACE1 QUBO rank-alignment diagnosis",
        "",
        f"- Pooled objective/BEDROC20 Spearman: `{pooled_objective_correlation:.4f}`.",
        f"- Median fixed-k Spearman: `{np.median(fixed_k_correlations):.4f}`.",
        f"- Objective-optimal k: `{diagnosis['objective_optimal_k']}`; BEDROC-optimal k: `{diagnosis['bedroc_optimal_k']}`.",
        f"- Rank-sensitive redesign authorized: `{diagnosis['analytically_weighted_rank_sensitive_redesign_authorized']}`.",
        "",
        config["interpretation_boundary"],
        "",
    ]
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report), encoding="ascii")
    result = {
        "schema_version": "1.0",
        "status": "stage42e_bace1_qubo_rank_alignment_diagnosis_complete",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, Path(__file__).resolve()),
        "sample_count": len(sampled),
        "diagnosis": diagnosis,
        "exact_k_ladder": [
            {
                "subset_size": int(row["subset_size"]),
                "objective": float(row["objective"]),
                "robust_bedroc_composite": float(row["robust_bedroc_composite"]),
            }
            for row in exact_by_size
        ],
        "data_boundary": {
            "train_rows_read": 266,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {
            key: descriptor(root, path)
            for key, path in outputs.items()
            if key != "result_json"
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(json.dumps(diagnosis, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage42e_bace1_qubo_rank_alignment_diagnosis.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
