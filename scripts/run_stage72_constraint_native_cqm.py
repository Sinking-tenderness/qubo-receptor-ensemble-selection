"""Build and stress-test a constraint-native CQM for the frozen Stage70 objective."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import dimod
import numpy as np


TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verified(root: Path, value: dict[str, Any], label: str) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage72 frozen {label} identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage72 frozen {label} size differs: {path}")
    return path


def runtime_versions() -> dict[str, str]:
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": np.__version__,
        "dimod": importlib.metadata.version("dimod"),
    }


def verify_runtime(config: dict[str, Any]) -> dict[str, str]:
    observed = runtime_versions()
    for key, expected in config["constraint_native_model"][
        "runtime_versions"
    ].items():
        if observed.get(key) != str(expected):
            raise ValueError(
                f"Stage72 runtime version differs for {key}: "
                f"{observed.get(key)} != {expected}"
            )
    return observed


def redundancy_matrix(record: dict[str, Any]) -> np.ndarray:
    count = len(record["receptor_ids"])
    matrix = np.zeros((count, count), dtype=float)
    values = iter(float(value) for value in record["stable_redundancy_upper_triangle"])
    for left, right in itertools.combinations(range(count), 2):
        value = next(values)
        matrix[left, right] = value
        matrix[right, left] = value
    try:
        next(values)
    except StopIteration:
        return matrix
    raise ValueError("Stage72 redundancy upper triangle has excess values")


def build_native_model(record: dict[str, Any]) -> dict[str, Any]:
    receptor_ids = [str(value) for value in record["receptor_ids"]]
    receptor_count = len(receptor_ids)
    deficits = np.asarray(record["integer_deficits"], dtype=int)
    if len(deficits) != receptor_count:
        raise ValueError("Stage72 receptor and deficit counts differ")
    subset_size = int(record["reference_k"])
    maximum_deficit = int(record["maximum_integer_deficit"])
    redundancy = redundancy_matrix(record)
    pair_indices = list(itertools.combinations(range(receptor_count), 2))
    raw_coefficients = np.asarray(
        [redundancy[left, right] for left, right in pair_indices], dtype=float
    )
    coefficient_minimum = float(np.min(raw_coefficients))
    coefficient_maximum = float(np.max(raw_coefficients))
    pair_center = (coefficient_minimum + coefficient_maximum) / 2.0
    centered_coefficients = raw_coefficients - pair_center
    scale = float(np.max(np.abs(centered_coefficients)))
    if scale <= TOLERANCE:
        raise ValueError("Stage72 centered objective has no programmable coefficient")
    objective_offset = pair_center * math.comb(subset_size, 2)
    variables = [f"x{index:03d}" for index in range(receptor_count)]
    binaries = [dimod.Binary(variable) for variable in variables]
    objective = float(objective_offset)
    for (left, right), coefficient in zip(pair_indices, centered_coefficients):
        objective += float(coefficient) * binaries[left] * binaries[right]
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(objective)
    cqm.add_constraint(
        sum(binaries) == subset_size,
        label="cardinality_exact",
    )
    cqm.add_constraint(
        sum(
            int(deficits[index]) * binaries[index]
            for index in range(receptor_count)
        )
        <= maximum_deficit,
        label="quality_floor",
    )
    selected_lookup = {value: index for index, value in enumerate(receptor_ids)}
    selected_indices = tuple(
        sorted(
            selected_lookup[value]
            for value in str(record["selected_subset"]).split("+")
        )
    )
    canonical_record = {
        "variable_order": variables,
        "objective": {
            "pair_center": pair_center,
            "quadratic_pair_order": [list(pair) for pair in pair_indices],
            "quadratic_coefficients": [
                float(value) for value in centered_coefficients
            ],
            "offset": objective_offset,
        },
        "constraints": {
            "cardinality_exact": {
                "sense": "==",
                "linear_coefficients": [1] * receptor_count,
                "rhs": subset_size,
            },
            "quality_floor": {
                "sense": "<=",
                "linear_coefficients": [int(value) for value in deficits],
                "rhs": maximum_deficit,
            },
        },
    }
    return {
        "record": record,
        "receptor_ids": receptor_ids,
        "receptor_count": receptor_count,
        "deficits": deficits,
        "subset_size": subset_size,
        "maximum_deficit": maximum_deficit,
        "redundancy": redundancy,
        "pair_indices": pair_indices,
        "raw_coefficients": raw_coefficients,
        "pair_center": pair_center,
        "centered_coefficients": centered_coefficients,
        "objective_offset": objective_offset,
        "normalization_scale": scale,
        "normalized_coefficients": centered_coefficients / scale,
        "normalized_offset": objective_offset / scale,
        "variables": variables,
        "selected_indices": selected_indices,
        "cqm": cqm,
        "canonical_record": canonical_record,
        "cqm_sha256": canonical_sha256(canonical_record),
    }


def subset_name(model: dict[str, Any], subset: tuple[int, ...]) -> str:
    return "+".join(model["receptor_ids"][index] for index in subset)


def build_feasible_states(model: dict[str, Any]) -> dict[str, Any]:
    subsets: list[tuple[int, ...]] = []
    raw_objectives: list[float] = []
    receptor_count = int(model["receptor_count"])
    for subset in itertools.combinations(
        range(receptor_count), int(model["subset_size"])
    ):
        if int(np.sum(model["deficits"][list(subset)])) > int(
            model["maximum_deficit"]
        ):
            continue
        raw_objective = float(
            sum(
                model["redundancy"][left, right]
                for left, right in itertools.combinations(subset, 2)
            )
        )
        subsets.append(subset)
        raw_objectives.append(raw_objective)
        sample = {
            variable: int(index in subset)
            for index, variable in enumerate(model["variables"])
        }
        if not model["cqm"].check_feasible(sample, rtol=0.0, atol=0.0):
            raise ValueError("Stage72 CQM rejects a mathematically feasible state")
        observed = float(model["cqm"].objective.energy(sample))
        if not math.isclose(
            observed, raw_objective, rel_tol=0.0, abs_tol=1e-10
        ):
            raise ValueError("Stage72 centered CQM objective changes feasible energy")
    if not subsets:
        raise ValueError("Stage72 model has no feasible state")
    feature_matrix = np.asarray(
        [
            [int(left in subset and right in subset) for left, right in model["pair_indices"]]
            for subset in subsets
        ],
        dtype=np.uint8,
    )
    subset_lookup = {subset: index for index, subset in enumerate(subsets)}
    selected_index = subset_lookup.get(model["selected_indices"])
    if selected_index is None:
        raise ValueError("Stage72 source selection is infeasible")
    return {
        "subsets": subsets,
        "raw_objectives": np.asarray(raw_objectives, dtype=float),
        "features": feature_matrix,
        "selected_index": selected_index,
    }


def exact_optimum(
    model: dict[str, Any],
    states: dict[str, Any],
    coefficients: np.ndarray,
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    energies = float(model["normalized_offset"]) + states["features"] @ coefficients
    order = sorted(
        range(len(states["subsets"])),
        key=lambda index: (float(energies[index]), states["subsets"][index]),
    )
    best_energy = float(energies[order[0]])
    optimal = [index for index in order if float(energies[index]) <= best_energy + tolerance]
    chosen = min(optimal, key=lambda index: states["subsets"][index])
    selected_index = int(states["selected_index"])
    selected_set = set(model["selected_indices"])
    chosen_set = set(states["subsets"][chosen])
    raw_best = min(float(states["raw_objectives"][index]) for index in optimal)
    return {
        "best_energy": best_energy,
        "best_index": chosen,
        "best_subset": states["subsets"][chosen],
        "optimal_subset_count": len(optimal),
        "selected_remains_optimal": selected_index in optimal,
        "selected_remains_unique": len(optimal) == 1 and optimal[0] == selected_index,
        "selected_subset_jaccard": len(selected_set & chosen_set)
        / len(selected_set | chosen_set),
        "base_objective_regret": raw_best
        - float(states["raw_objectives"][selected_index]),
        "energies": energies,
        "order": order,
    }


def graph_diagnostics(
    model: dict[str, Any], states: dict[str, Any]
) -> dict[str, Any]:
    subsets = states["subsets"]
    objectives = states["raw_objectives"]
    adjacency = {
        index: [
            other
            for other in range(len(subsets))
            if other != index
            and len(set(subsets[index]) ^ set(subsets[other])) == 2
        ]
        for index in range(len(subsets))
    }
    seen: set[int] = set()
    component_count = 0
    for index in range(len(subsets)):
        if index in seen:
            continue
        component_count += 1
        stack = [index]
        seen.add(index)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    optimum = min(
        range(len(subsets)), key=lambda index: (objectives[index], subsets[index])
    )
    final_counts: dict[int, int] = {}
    recovery_count = 0
    for start in range(len(subsets)):
        current = start
        while True:
            improving = [
                neighbor
                for neighbor in adjacency[current]
                if objectives[neighbor] < objectives[current] - TOLERANCE
            ]
            if not improving:
                break
            current = min(
                improving, key=lambda index: (objectives[index], subsets[index])
            )
        final_counts[current] = final_counts.get(current, 0) + 1
        recovery_count += int(current == optimum)
    return {
        "feasible_swap_graph_component_count": component_count,
        "feasible_swap_graph_connected": component_count == 1,
        "best_improvement_local_minimum_count": len(final_counts),
        "best_improvement_global_recovery_count": recovery_count,
        "best_improvement_global_recovery_rate": recovery_count / len(subsets),
    }


def perturb_coefficients(
    model: dict[str, Any], noise_model: str, level: float, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(model["normalized_coefficients"], dtype=float)
    if noise_model == "none":
        changed = source.copy()
    elif noise_model == "round_to_nearest_full_scale":
        if level <= 0:
            raise ValueError("Stage72 quantization step must be positive")
        changed = np.rint(source / level) * level
    elif noise_model == "iid_gaussian_full_scale":
        if level <= 0:
            raise ValueError("Stage72 Gaussian sigma must be positive")
        changed = source + np.random.default_rng(seed).normal(
            0.0, level, size=len(source)
        )
    else:
        raise ValueError(f"unknown Stage72 noise model: {noise_model}")
    delta = changed - source
    return changed, {
        "perturbed_coefficients_sha256": canonical_sha256(
            [float(value) for value in changed]
        ),
        "maximum_absolute_coefficient_delta": float(np.max(np.abs(delta))),
        "rms_coefficient_delta": float(np.sqrt(np.mean(delta * delta))),
        "zeroed_nonzero_coefficient_count": int(
            np.sum((np.abs(source) > TOLERANCE) & (changed == 0.0))
        ),
    }


def coefficient_seed(
    base_seed: int, model_index: int, level_index: int, repeat: int
) -> int:
    return int(base_seed + model_index * 100_000 + level_index * 1_000 + repeat)


def model_metric_row(
    model: dict[str, Any],
    states: dict[str, Any],
    stage71_landscape: dict[tuple[str, int], dict[str, str]],
) -> dict[str, Any]:
    exact = exact_optimum(model, states, model["normalized_coefficients"])
    distinct = [
        index
        for index in exact["order"]
        if exact["energies"][index] > exact["best_energy"] + TOLERANCE
    ]
    normalized_gap = (
        float(exact["energies"][distinct[0]] - exact["best_energy"])
        if distinct
        else math.nan
    )
    raw_gap = normalized_gap * float(model["normalization_scale"])
    target_id = str(model["record"]["target_id"])
    outer_fold = int(model["record"]["outer_fold"])
    source = stage71_landscape[(target_id, outer_fold)]
    stage71_gap = float(source["normalized_feasible_energy_gap"])
    graph = graph_diagnostics(model, states)
    nonzero = np.abs(model["centered_coefficients"])
    nonzero = nonzero[nonzero > TOLERANCE]
    raw_nonzero = np.abs(model["raw_coefficients"])
    raw_nonzero = raw_nonzero[raw_nonzero > TOLERANCE]
    row = {
        "target_id": target_id,
        "outer_fold": outer_fold,
        "receptor_count": int(model["receptor_count"]),
        "source_stage70_logical_variable_count": int(
            model["record"]["qubo_summary"]["logical_variable_count"]
        ),
        "native_logical_variable_count": int(model["receptor_count"]),
        "logical_variable_reduction": int(
            model["record"]["qubo_summary"]["logical_variable_count"]
        )
        - int(model["receptor_count"]),
        "source_stage70_quadratic_coefficient_count": int(
            model["record"]["qubo_summary"]["quadratic_coefficient_count"]
        ),
        "native_objective_quadratic_coefficient_count": len(
            model["pair_indices"]
        ),
        "quadratic_coefficient_reduction": int(
            model["record"]["qubo_summary"]["quadratic_coefficient_count"]
        )
        - len(model["pair_indices"]),
        "explicit_constraint_count": len(model["cqm"].constraints),
        "cqm_num_biases": int(model["cqm"].num_biases()),
        "raw_pair_coefficient_minimum": float(np.min(model["raw_coefficients"])),
        "raw_pair_coefficient_maximum": float(np.max(model["raw_coefficients"])),
        "raw_pair_coefficient_dynamic_range": float(
            np.max(raw_nonzero) / np.min(raw_nonzero)
        ),
        "pair_midpoint_center": float(model["pair_center"]),
        "centered_maximum_absolute_coefficient": float(
            model["normalization_scale"]
        ),
        "centered_minimum_absolute_nonzero_coefficient": float(np.min(nonzero)),
        "centered_coefficient_dynamic_range": float(
            np.max(nonzero) / np.min(nonzero)
        ),
        "midpoint_scale_improvement_factor_vs_raw": float(
            np.max(np.abs(model["raw_coefficients"]))
            / model["normalization_scale"]
        ),
        "feasible_receptor_subset_count": len(states["subsets"]),
        "selected_subset": subset_name(model, model["selected_indices"]),
        "native_exact_best_subset": subset_name(model, exact["best_subset"]),
        "selected_is_native_exact_optimum": bool(
            exact["selected_remains_optimal"]
        ),
        "native_exact_optimum_unique": int(exact["optimal_subset_count"]) == 1,
        "native_normalized_feasible_energy_gap": normalized_gap,
        "native_raw_feasible_energy_gap": raw_gap,
        "stage71_penalty_bqm_normalized_feasible_energy_gap": stage71_gap,
        "normalized_gap_improvement_factor_vs_stage71": normalized_gap
        / stage71_gap
        if math.isfinite(normalized_gap) and math.isfinite(stage71_gap)
        else math.nan,
        "cqm_sha256": model["cqm_sha256"],
        **graph,
    }
    return row


def trial_row(
    model: dict[str, Any],
    states: dict[str, Any],
    noise_model: str,
    level: float,
    repeat: int,
    seed: int,
) -> dict[str, Any]:
    coefficients, perturbation = perturb_coefficients(
        model, noise_model, level, seed
    )
    exact = exact_optimum(model, states, coefficients)
    return {
        "target_id": model["record"]["target_id"],
        "outer_fold": int(model["record"]["outer_fold"]),
        "noise_model": noise_model,
        "noise_level": float(level),
        "repeat": repeat,
        "coefficient_noise_seed": seed,
        **perturbation,
        "exact_feasible_best_subset": subset_name(model, exact["best_subset"]),
        "exact_feasible_best_perturbed_energy": exact["best_energy"],
        "exact_feasible_optimum_degeneracy": exact["optimal_subset_count"],
        "exact_selected_remains_optimal": exact["selected_remains_optimal"],
        "exact_selected_remains_unique": exact["selected_remains_unique"],
        "exact_selected_subset_jaccard": exact["selected_subset_jaccard"],
        "exact_base_objective_regret": exact["base_objective_regret"],
    }


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def summarize_trials(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["noise_model"]), float(row["noise_level"]), "ALL")
        grouped.setdefault(key, []).append(row)
        target_key = (
            str(row["noise_model"]),
            float(row["noise_level"]),
            str(row["target_id"]),
        )
        grouped.setdefault(target_key, []).append(row)
    output: list[dict[str, Any]] = []
    for (noise_model, level, scope), selected in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        output.append(
            {
                "noise_model": noise_model,
                "noise_level": level,
                "scope": scope,
                "trial_count": len(selected),
                "model_count": len(
                    {
                        (str(row["target_id"]), int(row["outer_fold"]))
                        for row in selected
                    }
                ),
                "exact_selected_optimal_rate": statistics.fmean(
                    int(truth(row["exact_selected_remains_optimal"]))
                    for row in selected
                ),
                "exact_selected_unique_rate": statistics.fmean(
                    int(truth(row["exact_selected_remains_unique"]))
                    for row in selected
                ),
                "mean_exact_subset_jaccard": statistics.fmean(
                    float(row["exact_selected_subset_jaccard"])
                    for row in selected
                ),
                "mean_exact_base_objective_regret": statistics.fmean(
                    float(row["exact_base_objective_regret"])
                    for row in selected
                ),
                "maximum_absolute_coefficient_delta": max(
                    float(row["maximum_absolute_coefficient_delta"])
                    for row in selected
                ),
                "mean_rms_coefficient_delta": statistics.fmean(
                    float(row["rms_coefficient_delta"]) for row in selected
                ),
                "mean_zeroed_nonzero_coefficient_count": statistics.fmean(
                    int(row["zeroed_nonzero_coefficient_count"])
                    for row in selected
                ),
            }
        )
    return output


def summary_at(
    summaries: list[dict[str, Any]], noise_model: str, level: float, scope: str
) -> dict[str, Any]:
    matches = [
        row
        for row in summaries
        if row["noise_model"] == noise_model
        and math.isclose(float(row["noise_level"]), level, rel_tol=0.0, abs_tol=1e-15)
        and row["scope"] == scope
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Stage72 summary lookup differs: {noise_model}/{level}/{scope}"
        )
    return matches[0]


def noise_gate(
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
    noise_model: str,
    level: float,
) -> dict[str, Any]:
    overall = summary_at(summaries, noise_model, level, "ALL")
    target_rows = [
        summary_at(summaries, noise_model, level, target)
        for target in config["experiment"]["target_order"]
    ]
    gate = config["formulation_gate"]
    result = {
        "noise_model": noise_model,
        "noise_level": level,
        "overall_exact_unique_rate": float(overall["exact_selected_unique_rate"]),
        "worst_target_exact_unique_rate": min(
            float(row["exact_selected_unique_rate"]) for row in target_rows
        ),
    }
    result["gate_passed"] = bool(
        result["overall_exact_unique_rate"]
        >= float(gate["minimum_overall_exact_unique_rate"])
        and result["worst_target_exact_unique_rate"]
        >= float(gate["minimum_worst_target_exact_unique_rate"])
    )
    return result


def robustness_envelope(
    summaries: list[dict[str, Any]], config: dict[str, Any], noise_model: str
) -> dict[str, Any]:
    levels = sorted(
        {
            float(row["noise_level"])
            for row in summaries
            if row["noise_model"] == noise_model and row["scope"] == "ALL"
        }
    )
    passing = [
        level
        for level in levels
        if noise_gate(summaries, config, noise_model, level)["gate_passed"]
    ]
    return {
        "noise_model": noise_model,
        "largest_tested_level_passing_project_gate": max(passing)
        if passing
        else None,
        "passing_level_count": len(passing),
        "tested_level_count": len(levels),
    }


def report_text(result: dict[str, Any]) -> str:
    formulation = result["formulation_summary"]
    matched = result["matched_stage71_reference_gate"]
    stress = result["stress_reference_gate"]
    return rf"""# Stage72 constraint-native CQM

## Question

Can the frozen Stage70 receptor-portfolio objective retain its exact solution while moving cardinality and quality constraints out of penalty-expanded BQM coefficients?

## Formulation

Stage72 minimizes

$$
\sum_{{i<j}}(R_{{ij}}-c)x_ix_j+c\binom{{k}}{{2}},
$$

subject to

$$
\sum_i x_i=k, \qquad \sum_i d_i x_i\le D.
$$

Here $c=(\min R+\max R)/2$. Because every feasible state has exactly $\binom{{k}}{{2}}$ selected pairs, the midpoint transformation preserves the original objective exactly while reducing programmable full scale.

## Result

- Exact source optima preserved: `{formulation['exact_source_optimum_count']}/{formulation['model_count']}`.
- Connected feasible swap graphs: `{formulation['connected_feasible_swap_graph_count']}/{formulation['model_count']}`.
- Minimum normalized-gap improvement versus Stage71: `{formulation['minimum_normalized_gap_improvement_factor_vs_stage71']:.6g}x`.
- Maximum native logical variables: `{formulation['maximum_native_logical_variable_count']}`.
- Maximum removed slack variables: `{formulation['maximum_logical_variable_reduction']}`.

## Noise gates

- Matched Stage71 reference $10^{{-6}}$: quantization `{matched['quantization']['gate_passed']}`, Gaussian `{matched['gaussian']['gate_passed']}`.
- Stress reference $10^{{-3}}$: quantization `{stress['quantization']['gate_passed']}`, Gaussian `{stress['gaussian']['gate_passed']}`.
- Constraint-native formulation freeze authorized: `{result['decision']['constraint_native_formulation_freeze_authorized']}`.
- Direct QPU execution authorized: `{result['decision']['direct_qpu_execution_authorized']}`.

## Boundary

This establishes a precision-robust logical constrained model, not a hardware implementation. CQM hybrid solvers or constraint-preserving gate/annealing methods still require a separate solver-scaling and embedding study. No hardware speedup or quantum advantage is claimed.
"""


def compute(config: dict[str, Any], root: Path) -> dict[str, Any]:
    implementation_paths = {
        key: verified(root, value, key)
        for key, value in config["implementation"].items()
    }
    input_paths = {
        key: verified(root, value, key) for key, value in config["inputs"].items()
    }
    versions = verify_runtime(config)
    stage71_result = read_json(input_paths["stage71_result"])
    stage71_audit = read_json(input_paths["stage71_audit"])
    if not stage71_result["decision"]["constraint_native_reformulation_authorized"]:
        raise ValueError("Stage72 requires Stage71 constraint-native authorization")
    if stage71_audit.get("status") != (
        "stage71_qubo_coefficient_noise_robustness_independent_audit_ok"
    ):
        raise ValueError("Stage72 requires the Stage71 independent audit")
    model_record = read_json(input_paths["stage70_model_record"])
    if int(model_record["model_count"]) != int(
        config["experiment"]["required_model_count"]
    ):
        raise ValueError("Stage72 frozen model count differs")
    models = [build_native_model(record) for record in model_record["models"]]
    states = [build_feasible_states(model) for model in models]
    stage71_landscape_rows = read_csv(input_paths["stage71_exact_landscape"])
    stage71_landscape = {
        (row["target_id"], int(row["outer_fold"])): row
        for row in stage71_landscape_rows
    }
    metric_rows = [
        model_metric_row(model, current_states, stage71_landscape)
        for model, current_states in zip(models, states)
    ]
    noise = config["noise_screen"]
    base_seed = int(noise["coefficient_noise_seed_base"])
    trial_rows: list[dict[str, Any]] = []
    for model_index, (model, current_states) in enumerate(zip(models, states)):
        trial_rows.append(
            trial_row(
                model,
                current_states,
                "none",
                0.0,
                0,
                coefficient_seed(base_seed, model_index, 0, 0),
            )
        )
        for level_index, value in enumerate(noise["quantization_steps"], start=1):
            trial_rows.append(
                trial_row(
                    model,
                    current_states,
                    "round_to_nearest_full_scale",
                    float(value),
                    0,
                    coefficient_seed(base_seed, model_index, level_index, 0),
                )
            )
        offset = 1 + len(noise["quantization_steps"])
        for level_index, value in enumerate(noise["gaussian_sigmas"]):
            for repeat in range(int(noise["gaussian_repeats_per_level"])):
                trial_rows.append(
                    trial_row(
                        model,
                        current_states,
                        "iid_gaussian_full_scale",
                        float(value),
                        repeat,
                        coefficient_seed(
                            base_seed, model_index, offset + level_index, repeat
                        ),
                    )
                )
        print(
            json.dumps(
                {
                    "target_id": model["record"]["target_id"],
                    "outer_fold": model["record"]["outer_fold"],
                    "trial_rows_completed": len(trial_rows),
                }
            )
        )
    summaries = summarize_trials(trial_rows)
    finite_improvements = [
        float(row["normalized_gap_improvement_factor_vs_stage71"])
        for row in metric_rows
        if math.isfinite(
            float(row["normalized_gap_improvement_factor_vs_stage71"])
        )
    ]
    formulation_summary = {
        "model_count": len(metric_rows),
        "exact_source_optimum_count": sum(
            bool(row["selected_is_native_exact_optimum"])
            and bool(row["native_exact_optimum_unique"])
            for row in metric_rows
        ),
        "connected_feasible_swap_graph_count": sum(
            bool(row["feasible_swap_graph_connected"]) for row in metric_rows
        ),
        "single_local_minimum_model_count": sum(
            int(row["best_improvement_local_minimum_count"]) == 1
            for row in metric_rows
        ),
        "minimum_best_improvement_global_recovery_rate": min(
            float(row["best_improvement_global_recovery_rate"])
            for row in metric_rows
        ),
        "finite_gap_comparison_count": len(finite_improvements),
        "minimum_normalized_gap_improvement_factor_vs_stage71": min(
            finite_improvements
        ),
        "maximum_normalized_gap_improvement_factor_vs_stage71": max(
            finite_improvements
        ),
        "minimum_native_normalized_feasible_energy_gap": min(
            float(row["native_normalized_feasible_energy_gap"])
            for row in metric_rows
            if math.isfinite(float(row["native_normalized_feasible_energy_gap"]))
        ),
        "maximum_native_logical_variable_count": max(
            int(row["native_logical_variable_count"]) for row in metric_rows
        ),
        "maximum_logical_variable_reduction": max(
            int(row["logical_variable_reduction"]) for row in metric_rows
        ),
        "maximum_quadratic_coefficient_reduction": max(
            int(row["quadratic_coefficient_reduction"]) for row in metric_rows
        ),
    }
    matched_level = float(config["formulation_gate"]["matched_stage71_reference"])
    stress_level = float(config["formulation_gate"]["stress_reference"])
    matched_gate = {
        "quantization": noise_gate(
            summaries,
            config,
            "round_to_nearest_full_scale",
            matched_level,
        ),
        "gaussian": noise_gate(
            summaries, config, "iid_gaussian_full_scale", matched_level
        ),
    }
    stress_gate = {
        "quantization": noise_gate(
            summaries,
            config,
            "round_to_nearest_full_scale",
            stress_level,
        ),
        "gaussian": noise_gate(
            summaries, config, "iid_gaussian_full_scale", stress_level
        ),
    }
    envelopes = {
        model: robustness_envelope(summaries, config, model)
        for model in (
            "round_to_nearest_full_scale",
            "iid_gaussian_full_scale",
        )
    }
    gate = config["formulation_gate"]
    formulation_passed = bool(
        formulation_summary["exact_source_optimum_count"]
        == int(gate["required_exact_source_optimum_count"])
        and formulation_summary["connected_feasible_swap_graph_count"]
        == int(gate["required_connected_feasible_swap_graph_count"])
        and formulation_summary["minimum_normalized_gap_improvement_factor_vs_stage71"]
        >= float(gate["minimum_normalized_gap_improvement_factor_vs_stage71"])
        and formulation_summary["maximum_native_logical_variable_count"]
        <= int(gate["maximum_native_logical_variable_count"])
        and matched_gate["quantization"]["gate_passed"]
        and matched_gate["gaussian"]["gate_passed"]
        and stress_gate["quantization"]["gate_passed"]
        and stress_gate["gaussian"]["gate_passed"]
    )
    model_record_output = {
        "schema_version": "1.0",
        "algorithm_id": config["constraint_native_model"]["algorithm_id"],
        "model_count": len(models),
        "models": [
            {
                "target_id": model["record"]["target_id"],
                "outer_fold": int(model["record"]["outer_fold"]),
                "reference_k": int(model["subset_size"]),
                "receptor_ids": model["receptor_ids"],
                "integer_deficits": [int(value) for value in model["deficits"]],
                "maximum_integer_deficit": int(model["maximum_deficit"]),
                "selected_subset": subset_name(model, model["selected_indices"]),
                "pair_midpoint_center": float(model["pair_center"]),
                "objective_offset": float(model["objective_offset"]),
                "normalization_scale": float(model["normalization_scale"]),
                "centered_pair_coefficients": [
                    float(value) for value in model["centered_coefficients"]
                ],
                "cqm_sha256": model["cqm_sha256"],
            }
            for model in models
        ],
    }
    output_paths = {
        key: root / str(value) for key, value in config["outputs"].items()
    }
    write_csv(output_paths["model_metrics_csv"], metric_rows)
    write_csv(output_paths["noise_trials_csv"], trial_rows)
    write_csv(output_paths["noise_summary_csv"], summaries)
    write_json(output_paths["model_record_json"], model_record_output)
    payload = {
        "formulation_summary": formulation_summary,
        "matched_stage71_reference_gate": matched_gate,
        "stress_reference_gate": stress_gate,
        "robustness_envelopes": envelopes,
        "formulation_gate_passed": formulation_passed,
    }
    result = {
        "schema_version": "1.0",
        "status": "stage72_constraint_native_cqm_complete",
        "experiment_class": (
            "post-hoc constraint-native logical-model and coefficient-noise "
            "development on frozen historical objectives"
        ),
        "config": descriptor(root, root / "configs/stage72_constraint_native_cqm.json"),
        "implementation": {
            key: descriptor(root, path) for key, path in implementation_paths.items()
        },
        "inputs": {key: descriptor(root, path) for key, path in input_paths.items()},
        "runtime_versions": versions,
        "formulation_summary": formulation_summary,
        "noise_trial_count": len(trial_rows),
        "noise_summary_count": len(summaries),
        "matched_stage71_reference_gate": matched_gate,
        "stress_reference_gate": stress_gate,
        "robustness_envelopes": envelopes,
        "formulation_gate": {
            "constraint_native_formulation_gate_passed": formulation_passed
        },
        "decision": {
            "constraint_native_formulation_freeze_authorized": formulation_passed,
            "solver_scaling_benchmark_authorized": formulation_passed,
            "direct_qpu_execution_authorized": False,
            "new_target_preregistration_remains_authorized": stage71_result[
                "decision"
            ]["new_target_preregistration_remains_authorized"],
            "quantum_advantage_claim_authorized": False,
            "next_action": (
                "benchmark CQM-hybrid and constraint-preserving solver routes across increasing candidate-pool sizes without hardware execution"
                if formulation_passed
                else "retain Stage71 and redesign the native objective scaling"
            ),
        },
        "data_boundary": {
            "historical_development_targets_read": len(
                config["experiment"]["target_order"]
            ),
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "analysis_payload_sha256": canonical_sha256(payload),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    result["outputs"] = {
        key: descriptor(root, output_paths[key])
        for key in (
            "model_metrics_csv",
            "noise_trials_csv",
            "noise_summary_csv",
            "model_record_json",
        )
    }
    write_json(output_paths["result_json"], result)
    output_paths["report_md"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["report_md"].write_text(
        report_text(result), encoding="utf-8", newline="\n"
    )
    result["outputs"]["report_md"] = descriptor(root, output_paths["report_md"])
    write_json(output_paths["result_json"], result)
    return result


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    expected = root / "configs/stage72_constraint_native_cqm.json"
    if config_path != expected.resolve():
        raise ValueError("Stage72 must run from its frozen repository config")
    config = read_json(config_path)
    result_path = root / str(config["outputs"]["result_json"])
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"Stage72 result exists: {result_path}")
    result = compute(config, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage72_constraint_native_cqm.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
