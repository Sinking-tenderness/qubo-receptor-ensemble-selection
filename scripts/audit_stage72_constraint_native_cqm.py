"""Independently audit the Stage72 constraint-native CQM experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

import dimod
import numpy as np


TOLERANCE = 1e-10


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


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def checked(root: Path, value: dict[str, Any], label: str) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage72 {label} identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage72 {label} size differs: {path}")
    return path


def close(observed: Any, expected: Any, label: str, tolerance: float = TOLERANCE) -> None:
    first = float(observed)
    second = float(expected)
    if math.isnan(first) and math.isnan(second):
        return
    if not math.isclose(first, second, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"Stage72 numeric value differs for {label}: {first} != {second}")


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def independent_model(record: dict[str, Any]) -> dict[str, Any]:
    receptor_ids = [str(value) for value in record["receptor_ids"]]
    count = len(receptor_ids)
    deficits = np.asarray(record["integer_deficits"], dtype=int)
    pairs = list(itertools.combinations(range(count), 2))
    raw = np.asarray(record["stable_redundancy_upper_triangle"], dtype=float)
    if len(raw) != len(pairs):
        raise ValueError("Stage72 independent pair coefficient count differs")
    center = (float(np.min(raw)) + float(np.max(raw))) / 2.0
    centered = raw - center
    scale = float(np.max(np.abs(centered)))
    subset_size = int(record["reference_k"])
    maximum = int(record["maximum_integer_deficit"])
    variables = [f"x{index:03d}" for index in range(count)]
    offset = center * math.comb(subset_size, 2)
    lookup = {value: index for index, value in enumerate(receptor_ids)}
    selected = tuple(
        sorted(lookup[value] for value in str(record["selected_subset"]).split("+"))
    )
    canonical = {
        "variable_order": variables,
        "objective": {
            "pair_center": center,
            "quadratic_pair_order": [list(pair) for pair in pairs],
            "quadratic_coefficients": [float(value) for value in centered],
            "offset": offset,
        },
        "constraints": {
            "cardinality_exact": {
                "sense": "==",
                "linear_coefficients": [1] * count,
                "rhs": subset_size,
            },
            "quality_floor": {
                "sense": "<=",
                "linear_coefficients": [int(value) for value in deficits],
                "rhs": maximum,
            },
        },
    }
    binaries = [dimod.Binary(variable) for variable in variables]
    objective = float(offset)
    for (left, right), coefficient in zip(pairs, centered):
        objective += float(coefficient) * binaries[left] * binaries[right]
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(objective)
    cqm.add_constraint(sum(binaries) == subset_size, label="cardinality_exact")
    cqm.add_constraint(
        sum(int(deficits[index]) * binaries[index] for index in range(count))
        <= maximum,
        label="quality_floor",
    )
    return {
        "record": record,
        "receptor_ids": receptor_ids,
        "count": count,
        "deficits": deficits,
        "pairs": pairs,
        "raw": raw,
        "center": center,
        "centered": centered,
        "scale": scale,
        "normalized": centered / scale,
        "offset": offset,
        "normalized_offset": offset / scale,
        "subset_size": subset_size,
        "maximum": maximum,
        "variables": variables,
        "selected": selected,
        "canonical": canonical,
        "hash": canonical_sha256(canonical),
        "cqm": cqm,
    }


def independent_states(model: dict[str, Any]) -> dict[str, Any]:
    subsets: list[tuple[int, ...]] = []
    raw_objectives: list[float] = []
    for subset in itertools.combinations(range(model["count"]), model["subset_size"]):
        if int(np.sum(model["deficits"][list(subset)])) > model["maximum"]:
            continue
        raw_objective = float(
            sum(
                model["raw"][model["pairs"].index((left, right))]
                for left, right in itertools.combinations(subset, 2)
            )
        )
        sample = {
            variable: int(index in subset)
            for index, variable in enumerate(model["variables"])
        }
        if not model["cqm"].check_feasible(sample, rtol=0.0, atol=0.0):
            raise ValueError("Stage72 independently built CQM rejects a feasible state")
        close(
            model["cqm"].objective.energy(sample),
            raw_objective,
            "independent CQM objective",
        )
        subsets.append(subset)
        raw_objectives.append(raw_objective)
    features = np.asarray(
        [
            [int(left in subset and right in subset) for left, right in model["pairs"]]
            for subset in subsets
        ],
        dtype=np.uint8,
    )
    return {
        "subsets": subsets,
        "raw_objectives": np.asarray(raw_objectives, dtype=float),
        "features": features,
        "selected_index": subsets.index(model["selected"]),
    }


def exact_optimum(
    model: dict[str, Any], states: dict[str, Any], coefficients: np.ndarray
) -> dict[str, Any]:
    energies = model["normalized_offset"] + states["features"] @ coefficients
    order = sorted(
        range(len(states["subsets"])),
        key=lambda index: (float(energies[index]), states["subsets"][index]),
    )
    best = float(energies[order[0]])
    optimal = [index for index in order if float(energies[index]) <= best + 1e-12]
    chosen = min(optimal, key=lambda index: states["subsets"][index])
    selected_index = states["selected_index"]
    selected = set(model["selected"])
    chosen_set = set(states["subsets"][chosen])
    raw_best = min(float(states["raw_objectives"][index]) for index in optimal)
    return {
        "energies": energies,
        "order": order,
        "best": best,
        "best_subset": states["subsets"][chosen],
        "degeneracy": len(optimal),
        "selected_optimal": selected_index in optimal,
        "selected_unique": len(optimal) == 1 and optimal[0] == selected_index,
        "jaccard": len(selected & chosen_set) / len(selected | chosen_set),
        "regret": raw_best - float(states["raw_objectives"][selected_index]),
    }


def name(model: dict[str, Any], subset: tuple[int, ...]) -> str:
    return "+".join(model["receptor_ids"][index] for index in subset)


def graph_diagnostics(model: dict[str, Any], states: dict[str, Any]) -> dict[str, Any]:
    subsets = states["subsets"]
    objectives = states["raw_objectives"]
    adjacency = {
        index: [
            other
            for other in range(len(subsets))
            if other != index and len(set(subsets[index]) ^ set(subsets[other])) == 2
        ]
        for index in range(len(subsets))
    }
    seen: set[int] = set()
    components = 0
    for index in range(len(subsets)):
        if index in seen:
            continue
        components += 1
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
    endpoints: set[int] = set()
    recovered = 0
    for start in range(len(subsets)):
        current = start
        while True:
            improving = [
                neighbor
                for neighbor in adjacency[current]
                if objectives[neighbor] < objectives[current] - 1e-12
            ]
            if not improving:
                break
            current = min(
                improving, key=lambda index: (objectives[index], subsets[index])
            )
        endpoints.add(current)
        recovered += int(current == optimum)
    return {
        "component_count": components,
        "connected": components == 1,
        "local_minimum_count": len(endpoints),
        "recovery_count": recovered,
        "recovery_rate": recovered / len(subsets),
    }


def independent_perturbation(
    model: dict[str, Any], noise_model: str, level: float, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    source = model["normalized"]
    if noise_model == "none":
        changed = source.copy()
    elif noise_model == "round_to_nearest_full_scale":
        changed = np.rint(source / level) * level
    elif noise_model == "iid_gaussian_full_scale":
        changed = source + np.random.default_rng(seed).normal(0.0, level, len(source))
    else:
        raise ValueError(f"Stage72 unknown noise model: {noise_model}")
    delta = changed - source
    return changed, {
        "hash": canonical_sha256([float(value) for value in changed]),
        "maximum_delta": float(np.max(np.abs(delta))),
        "rms_delta": float(np.sqrt(np.mean(delta * delta))),
        "zeroed_count": int(
            np.sum((np.abs(source) > 1e-12) & (changed == 0.0))
        ),
    }


def independent_summaries(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str], list[dict[str, str]]] = {}
    for row in rows:
        model = row["noise_model"]
        level = float(row["noise_level"])
        groups.setdefault((model, level, "ALL"), []).append(row)
        groups.setdefault((model, level, row["target_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for (model, level, scope), selected in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        output.append(
            {
                "noise_model": model,
                "noise_level": level,
                "scope": scope,
                "trial_count": len(selected),
                "model_count": len(
                    {(row["target_id"], int(row["outer_fold"])) for row in selected}
                ),
                "exact_selected_optimal_rate": statistics.fmean(
                    int(truth(row["exact_selected_remains_optimal"])) for row in selected
                ),
                "exact_selected_unique_rate": statistics.fmean(
                    int(truth(row["exact_selected_remains_unique"])) for row in selected
                ),
                "mean_exact_subset_jaccard": statistics.fmean(
                    float(row["exact_selected_subset_jaccard"]) for row in selected
                ),
                "mean_exact_base_objective_regret": statistics.fmean(
                    float(row["exact_base_objective_regret"]) for row in selected
                ),
                "maximum_absolute_coefficient_delta": max(
                    float(row["maximum_absolute_coefficient_delta"]) for row in selected
                ),
                "mean_rms_coefficient_delta": statistics.fmean(
                    float(row["rms_coefficient_delta"]) for row in selected
                ),
                "mean_zeroed_nonzero_coefficient_count": statistics.fmean(
                    int(row["zeroed_nonzero_coefficient_count"]) for row in selected
                ),
            }
        )
    return output


def compare_summaries(observed: list[dict[str, str]], expected: list[dict[str, Any]]) -> None:
    if len(observed) != len(expected):
        raise ValueError("Stage72 noise summary row count differs")
    for first, second in zip(observed, expected):
        for key, value in second.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                close(first[key], value, f"noise summary:{key}")
            elif str(first[key]) != str(value):
                raise ValueError(f"Stage72 noise summary differs: {key}")


def find_summary(
    rows: list[dict[str, Any]], model: str, level: float, scope: str
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["noise_model"] == model
        and math.isclose(float(row["noise_level"]), level, rel_tol=0.0, abs_tol=1e-15)
        and row["scope"] == scope
    ]
    if len(matches) != 1:
        raise ValueError("Stage72 independent summary lookup differs")
    return matches[0]


def independent_noise_gate(
    rows: list[dict[str, Any]], config: dict[str, Any], model: str, level: float
) -> dict[str, Any]:
    overall = find_summary(rows, model, level, "ALL")
    target_rows = [
        find_summary(rows, model, level, target)
        for target in config["experiment"]["target_order"]
    ]
    gate = config["formulation_gate"]
    output = {
        "noise_model": model,
        "noise_level": level,
        "overall_exact_unique_rate": float(overall["exact_selected_unique_rate"]),
        "worst_target_exact_unique_rate": min(
            float(row["exact_selected_unique_rate"]) for row in target_rows
        ),
    }
    output["gate_passed"] = bool(
        output["overall_exact_unique_rate"]
        >= float(gate["minimum_overall_exact_unique_rate"])
        and output["worst_target_exact_unique_rate"]
        >= float(gate["minimum_worst_target_exact_unique_rate"])
    )
    return output


def independent_envelope(
    rows: list[dict[str, Any]], config: dict[str, Any], model: str
) -> dict[str, Any]:
    levels = sorted(
        {
            float(row["noise_level"])
            for row in rows
            if row["noise_model"] == model and row["scope"] == "ALL"
        }
    )
    passing = [
        level
        for level in levels
        if independent_noise_gate(rows, config, model, level)["gate_passed"]
    ]
    return {
        "noise_model": model,
        "largest_tested_level_passing_project_gate": max(passing) if passing else None,
        "passing_level_count": len(passing),
        "tested_level_count": len(levels),
    }


def expected_trial_keys(
    config: dict[str, Any], models: list[dict[str, Any]]
) -> set[tuple[Any, ...]]:
    output: set[tuple[Any, ...]] = set()
    noise = config["noise_screen"]
    base = int(noise["coefficient_noise_seed_base"])
    for model_index, model in enumerate(models):
        target = model["record"]["target_id"]
        fold = int(model["record"]["outer_fold"])
        output.add((target, fold, "none", 0.0, 0, base + model_index * 100_000))
        for index, level in enumerate(noise["quantization_steps"], start=1):
            output.add(
                (
                    target,
                    fold,
                    "round_to_nearest_full_scale",
                    float(level),
                    0,
                    base + model_index * 100_000 + index * 1_000,
                )
            )
        offset = 1 + len(noise["quantization_steps"])
        for index, level in enumerate(noise["gaussian_sigmas"]):
            for repeat in range(int(noise["gaussian_repeats_per_level"])):
                output.add(
                    (
                        target,
                        fold,
                        "iid_gaussian_full_scale",
                        float(level),
                        repeat,
                        base
                        + model_index * 100_000
                        + (offset + index) * 1_000
                        + repeat,
                    )
                )
    return output


def run(config_path: Path, result_path: Path, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path.resolve())
    result = read_json(result_path.resolve())
    if result.get("status") != "stage72_constraint_native_cqm_complete":
        raise ValueError("Stage72 source result did not complete")
    if checked(root, result["config"], "config").resolve() != config_path.resolve():
        raise ValueError("Stage72 result config differs")
    for key, value in config["implementation"].items():
        checked(root, value, key)
    for key, value in config["inputs"].items():
        checked(root, value, key)
    outputs = {key: checked(root, value, key) for key, value in result["outputs"].items()}
    source_models = read_json(root / config["inputs"]["stage70_model_record"]["path"])
    models = [independent_model(record) for record in source_models["models"]]
    states = [independent_states(model) for model in models]
    lookup = {
        (model["record"]["target_id"], int(model["record"]["outer_fold"])): (
            model,
            current_states,
        )
        for model, current_states in zip(models, states)
    }
    output_model_record = read_json(outputs["model_record_json"])
    if int(output_model_record["model_count"]) != len(models):
        raise ValueError("Stage72 output model count differs")
    for record in output_model_record["models"]:
        model, _ = lookup[(record["target_id"], int(record["outer_fold"]))]
        if record["cqm_sha256"] != model["hash"]:
            raise ValueError("Stage72 CQM canonical hash differs")
        close(record["pair_midpoint_center"], model["center"], "model center")
        close(record["objective_offset"], model["offset"], "model offset")
        close(record["normalization_scale"], model["scale"], "model scale")
        if record["centered_pair_coefficients"] != [
            float(value) for value in model["centered"]
        ]:
            raise ValueError("Stage72 centered coefficient record differs")
    stage71_rows = read_csv(root / config["inputs"]["stage71_exact_landscape"]["path"])
    stage71_lookup = {
        (row["target_id"], int(row["outer_fold"])): row for row in stage71_rows
    }
    metric_rows = read_csv(outputs["model_metrics_csv"])
    if len(metric_rows) != len(models):
        raise ValueError("Stage72 metric row count differs")
    finite_improvements: list[float] = []
    for row in metric_rows:
        model, current_states = lookup[(row["target_id"], int(row["outer_fold"]))]
        exact = exact_optimum(model, current_states, model["normalized"])
        distinct = [
            index
            for index in exact["order"]
            if exact["energies"][index] > exact["best"] + 1e-12
        ]
        gap = (
            float(exact["energies"][distinct[0]] - exact["best"])
            if distinct
            else math.nan
        )
        stage71_gap = float(
            stage71_lookup[(row["target_id"], int(row["outer_fold"]))][
                "normalized_feasible_energy_gap"
            ]
        )
        improvement = gap / stage71_gap if math.isfinite(gap) else math.nan
        if math.isfinite(improvement):
            finite_improvements.append(improvement)
        graph = graph_diagnostics(model, current_states)
        if row["cqm_sha256"] != model["hash"]:
            raise ValueError("Stage72 metric CQM hash differs")
        if row["native_exact_best_subset"] != name(model, exact["best_subset"]):
            raise ValueError("Stage72 native exact subset differs")
        if truth(row["selected_is_native_exact_optimum"]) != exact["selected_optimal"]:
            raise ValueError("Stage72 native selected-optimum status differs")
        if truth(row["native_exact_optimum_unique"]) != (exact["degeneracy"] == 1):
            raise ValueError("Stage72 native optimum uniqueness differs")
        close(row["pair_midpoint_center"], model["center"], "metric center")
        close(row["centered_maximum_absolute_coefficient"], model["scale"], "metric scale")
        close(row["native_normalized_feasible_energy_gap"], gap, "native gap")
        close(row["normalized_gap_improvement_factor_vs_stage71"], improvement, "gap improvement")
        if int(row["explicit_constraint_count"]) != 2:
            raise ValueError("Stage72 explicit constraint count differs")
        if int(row["cqm_num_biases"]) != int(model["cqm"].num_biases()):
            raise ValueError("Stage72 CQM bias count differs")
        if truth(row["feasible_swap_graph_connected"]) != graph["connected"]:
            raise ValueError("Stage72 feasible graph connectivity differs")
        if int(row["best_improvement_local_minimum_count"]) != graph["local_minimum_count"]:
            raise ValueError("Stage72 local-minimum count differs")
        close(row["best_improvement_global_recovery_rate"], graph["recovery_rate"], "greedy recovery")
    trial_rows = read_csv(outputs["noise_trials_csv"])
    observed_keys = {
        (
            row["target_id"],
            int(row["outer_fold"]),
            row["noise_model"],
            float(row["noise_level"]),
            int(row["repeat"]),
            int(row["coefficient_noise_seed"]),
        )
        for row in trial_rows
    }
    if observed_keys != expected_trial_keys(config, models):
        raise ValueError("Stage72 trial grid or seed schedule differs")
    for row in trial_rows:
        model, current_states = lookup[(row["target_id"], int(row["outer_fold"]))]
        coefficients, perturbation = independent_perturbation(
            model,
            row["noise_model"],
            float(row["noise_level"]),
            int(row["coefficient_noise_seed"]),
        )
        if row["perturbed_coefficients_sha256"] != perturbation["hash"]:
            raise ValueError("Stage72 perturbation hash differs")
        close(row["maximum_absolute_coefficient_delta"], perturbation["maximum_delta"], "maximum delta")
        close(row["rms_coefficient_delta"], perturbation["rms_delta"], "RMS delta")
        if int(row["zeroed_nonzero_coefficient_count"]) != perturbation["zeroed_count"]:
            raise ValueError("Stage72 zeroed coefficient count differs")
        exact = exact_optimum(model, current_states, coefficients)
        if row["exact_feasible_best_subset"] != name(model, exact["best_subset"]):
            raise ValueError("Stage72 noisy exact subset differs")
        close(row["exact_feasible_best_perturbed_energy"], exact["best"], "noisy exact energy")
        if int(row["exact_feasible_optimum_degeneracy"]) != exact["degeneracy"]:
            raise ValueError("Stage72 noisy degeneracy differs")
        if truth(row["exact_selected_remains_optimal"]) != exact["selected_optimal"]:
            raise ValueError("Stage72 noisy selected-optimal status differs")
        if truth(row["exact_selected_remains_unique"]) != exact["selected_unique"]:
            raise ValueError("Stage72 noisy selected-unique status differs")
        close(row["exact_selected_subset_jaccard"], exact["jaccard"], "noisy jaccard")
        close(row["exact_base_objective_regret"], exact["regret"], "noisy regret")
    summaries = independent_summaries(trial_rows)
    compare_summaries(read_csv(outputs["noise_summary_csv"]), summaries)
    formulation = {
        "model_count": len(metric_rows),
        "exact_source_optimum_count": sum(
            truth(row["selected_is_native_exact_optimum"])
            and truth(row["native_exact_optimum_unique"])
            for row in metric_rows
        ),
        "connected_feasible_swap_graph_count": sum(
            truth(row["feasible_swap_graph_connected"]) for row in metric_rows
        ),
        "single_local_minimum_model_count": sum(
            int(row["best_improvement_local_minimum_count"]) == 1 for row in metric_rows
        ),
        "minimum_best_improvement_global_recovery_rate": min(
            float(row["best_improvement_global_recovery_rate"]) for row in metric_rows
        ),
        "finite_gap_comparison_count": len(finite_improvements),
        "minimum_normalized_gap_improvement_factor_vs_stage71": min(finite_improvements),
        "maximum_normalized_gap_improvement_factor_vs_stage71": max(finite_improvements),
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
    for key, value in formulation.items():
        if isinstance(value, float):
            close(result["formulation_summary"][key], value, f"formulation:{key}")
        elif result["formulation_summary"][key] != value:
            raise ValueError(f"Stage72 formulation summary differs: {key}")
    matched_level = float(config["formulation_gate"]["matched_stage71_reference"])
    stress_level = float(config["formulation_gate"]["stress_reference"])
    matched = {
        "quantization": independent_noise_gate(
            summaries, config, "round_to_nearest_full_scale", matched_level
        ),
        "gaussian": independent_noise_gate(
            summaries, config, "iid_gaussian_full_scale", matched_level
        ),
    }
    stress = {
        "quantization": independent_noise_gate(
            summaries, config, "round_to_nearest_full_scale", stress_level
        ),
        "gaussian": independent_noise_gate(
            summaries, config, "iid_gaussian_full_scale", stress_level
        ),
    }
    if matched != result["matched_stage71_reference_gate"]:
        raise ValueError("Stage72 matched-reference gate differs")
    if stress != result["stress_reference_gate"]:
        raise ValueError("Stage72 stress-reference gate differs")
    envelopes = {
        model: independent_envelope(summaries, config, model)
        for model in ("round_to_nearest_full_scale", "iid_gaussian_full_scale")
    }
    if envelopes != result["robustness_envelopes"]:
        raise ValueError("Stage72 robustness envelope differs")
    gate = config["formulation_gate"]
    passed = bool(
        formulation["exact_source_optimum_count"]
        == int(gate["required_exact_source_optimum_count"])
        and formulation["connected_feasible_swap_graph_count"]
        == int(gate["required_connected_feasible_swap_graph_count"])
        and formulation["minimum_normalized_gap_improvement_factor_vs_stage71"]
        >= float(gate["minimum_normalized_gap_improvement_factor_vs_stage71"])
        and formulation["maximum_native_logical_variable_count"]
        <= int(gate["maximum_native_logical_variable_count"])
        and all(row["gate_passed"] for row in matched.values())
        and all(row["gate_passed"] for row in stress.values())
    )
    if bool(result["formulation_gate"]["constraint_native_formulation_gate_passed"]) != passed:
        raise ValueError("Stage72 formulation decision differs")
    if bool(result["decision"]["constraint_native_formulation_freeze_authorized"]) != passed:
        raise ValueError("Stage72 formulation freeze decision differs")
    if bool(result["decision"]["direct_qpu_execution_authorized"]):
        raise ValueError("Stage72 cannot authorize direct hardware execution")
    payload = {
        "formulation_summary": result["formulation_summary"],
        "matched_stage71_reference_gate": result["matched_stage71_reference_gate"],
        "stress_reference_gate": result["stress_reference_gate"],
        "robustness_envelopes": result["robustness_envelopes"],
        "formulation_gate_passed": passed,
    }
    if canonical_sha256(payload) != result["analysis_payload_sha256"]:
        raise ValueError("Stage72 analysis payload hash differs")
    audit = {
        "schema_version": "1.0",
        "status": "stage72_constraint_native_cqm_independent_audit_ok",
        "source_result": descriptor(root, result_path),
        "constraint_native_models_independently_rebuilt": len(models),
        "model_metrics_independently_recomputed": len(metric_rows),
        "noise_trials_independently_recomputed": len(trial_rows),
        "noise_summaries_independently_recomputed": len(summaries),
        "constraint_native_formulation_gate_passed": passed,
        "constraint_native_formulation_freeze_authorized": passed,
        "solver_scaling_benchmark_authorized": passed,
        "direct_qpu_execution_authorized": False,
        "quantum_advantage_claim_authorized": False,
        "data_boundary": result["data_boundary"],
    }
    write_json(output_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage72_constraint_native_cqm.json"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("data/stage72_constraint_native_cqm_result.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage72_constraint_native_cqm_audit.json"),
    )
    args = parser.parse_args()
    run(args.config, args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
