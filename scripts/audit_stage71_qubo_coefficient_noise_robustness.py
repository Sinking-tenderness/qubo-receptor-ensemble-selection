"""Independently audit the frozen Stage71 coefficient-noise diagnosis."""

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
        raise ValueError(f"Stage71 {label} identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage71 {label} size differs: {path}")
    return path


def close(observed: Any, expected: Any, label: str, tolerance: float = TOLERANCE) -> None:
    first = float(observed)
    second = float(expected)
    if math.isnan(first) and math.isnan(second):
        return
    if not math.isclose(first, second, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"Stage71 numeric value differs for {label}: {first} != {second}")


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def independent_model(record: dict[str, Any]) -> dict[str, Any]:
    receptor_ids = [str(value) for value in record["receptor_ids"]]
    n = len(receptor_ids)
    weights = [int(value) for value in record["slack_weights"]]
    deficits = np.asarray(record["integer_deficits"], dtype=int)
    redundancy = np.zeros((n, n), dtype=float)
    source = iter(float(value) for value in record["stable_redundancy_upper_triangle"])
    for left, right in itertools.combinations(range(n), 2):
        value = next(source)
        redundancy[left, right] = value
        redundancy[right, left] = value
    try:
        next(source)
        raise ValueError("Stage71 source redundancy triangle has excess values")
    except StopIteration:
        pass
    k = int(record["reference_k"])
    maximum = int(record["maximum_integer_deficit"])
    center = int(record["integer_center"])
    rhs = maximum - k * center
    if rhs != int(record["centered_rhs"]):
        raise ValueError("Stage71 source centered right-hand side differs")
    pk = float(record["cardinality_penalty"])
    pq = float(record["quality_penalty"])
    centered = deficits.astype(float) - center
    slack = np.asarray(weights, dtype=float)
    linear = np.concatenate(
        [
            pk * (1 - 2 * k) + pq * (centered * centered - 2 * rhs * centered),
            pq * (slack * slack - 2 * rhs * slack),
        ]
    )
    pairs: list[tuple[int, int]] = []
    quadratic_parts: list[np.ndarray] = []
    for left, right in itertools.combinations(range(n), 2):
        pairs.append((left, right))
    receptor_upper = np.triu_indices(n, 1)
    quadratic_parts.append(
        2 * pk
        + redundancy[receptor_upper]
        + 2 * pq * (centered[:, None] * centered[None, :])[receptor_upper]
    )
    for left in range(n):
        for right in range(len(weights)):
            pairs.append((left, n + right))
    quadratic_parts.append((2 * pq * centered[:, None] * slack[None, :]).ravel())
    for left, right in itertools.combinations(range(len(weights)), 2):
        pairs.append((n + left, n + right))
    slack_upper = np.triu_indices(len(weights), 1)
    quadratic_parts.append(
        (2 * pq * slack[:, None] * slack[None, :])[slack_upper]
    )
    quadratic_array = np.concatenate(quadratic_parts)
    constant = float(pk * k**2 + pq * rhs**2)
    source_hash = canonical_sha256(
        {
            "center": center,
            "centered_rhs": rhs,
            "linear": [float(value) for value in linear],
            "quadratic": [float(value) for value in quadratic_array],
            "constant": constant,
        }
    )
    if source_hash != str(record["qubo_summary"]["qubo_sha256"]).upper():
        raise ValueError("Stage71 independent QUBO hash differs")
    scale = float(np.max(np.abs(np.concatenate([linear, quadratic_array]))))
    lookup = {value: index for index, value in enumerate(receptor_ids)}
    selected = tuple(
        sorted(lookup[value] for value in str(record["selected_subset"]).split("+"))
    )
    return {
        "record": record,
        "receptor_ids": receptor_ids,
        "n": n,
        "weights": weights,
        "deficits": deficits,
        "redundancy": redundancy,
        "k": k,
        "maximum": maximum,
        "pairs": pairs,
        "linear": linear / scale,
        "quadratic": quadratic_array / scale,
        "constant": constant / scale,
        "scale": scale,
        "selected": selected,
        "source_hash": source_hash,
    }


def independent_states(model: dict[str, Any]) -> dict[str, Any]:
    slack_patterns: dict[int, list[tuple[int, ...]]] = {}
    for bits in itertools.product((0, 1), repeat=len(model["weights"])):
        value = sum(weight * bit for weight, bit in zip(model["weights"], bits))
        slack_patterns.setdefault(value, []).append(bits)
    assignments: list[tuple[int, ...]] = []
    groups: dict[tuple[int, ...], list[int]] = {}
    for subset in itertools.combinations(range(model["n"]), model["k"]):
        deficit = int(np.sum(model["deficits"][list(subset)]))
        if deficit > model["maximum"]:
            continue
        slack = model["maximum"] - deficit
        current: list[int] = []
        for slack_bits in slack_patterns.get(slack, []):
            receptor_bits = tuple(int(index in subset) for index in range(model["n"]))
            current.append(len(assignments))
            assignments.append(receptor_bits + slack_bits)
        if not current:
            raise ValueError("Stage71 independent slack representation is incomplete")
        groups[subset] = current
    matrix = np.asarray(assignments, dtype=np.uint8)
    left = np.asarray([pair[0] for pair in model["pairs"]], dtype=int)
    right = np.asarray([pair[1] for pair in model["pairs"]], dtype=int)
    return {
        "assignments": matrix,
        "quadratic_features": matrix[:, left] * matrix[:, right],
        "groups": groups,
    }


def energies(
    model: dict[str, Any], states: dict[str, Any], linear: np.ndarray, quadratic: np.ndarray
) -> np.ndarray:
    return (
        model["constant"]
        + states["assignments"] @ linear
        + states["quadratic_features"] @ quadratic
    )


def exact_optimum(
    model: dict[str, Any], states: dict[str, Any], linear: np.ndarray, quadratic: np.ndarray
) -> dict[str, Any]:
    values = energies(model, states, linear, quadratic)
    rows: list[tuple[float, tuple[int, ...], int]] = []
    for subset, indices in states["groups"].items():
        index = min(indices, key=lambda value: (float(values[value]), value))
        rows.append((float(values[index]), subset, index))
    rows.sort(key=lambda row: (row[0], row[1]))
    best = rows[0][0]
    optimal = [row for row in rows if row[0] <= best + 1e-12]
    chosen = min(optimal, key=lambda row: row[1])
    selected_row = next(row for row in rows if row[1] == model["selected"])
    selected_set = set(model["selected"])
    best_set = set(chosen[1])
    objective = min(
        sum(
            model["redundancy"][left, right]
            for left, right in itertools.combinations(row[1], 2)
        )
        for row in optimal
    )
    return {
        "rows": rows,
        "best_energy": best,
        "best_subset": chosen[1],
        "best_bits": tuple(int(value) for value in states["assignments"][chosen[2]]),
        "degeneracy": len(optimal),
        "selected_optimal": any(row[1] == model["selected"] for row in optimal),
        "selected_unique": len(optimal) == 1 and optimal[0][1] == model["selected"],
        "selected_energy": selected_row[0],
        "jaccard": len(selected_set & best_set) / len(selected_set | best_set),
        "regret": float(objective - model["record"]["selected_redundancy_sum"]),
    }


def name(model: dict[str, Any], subset: tuple[int, ...]) -> str:
    return "+".join(model["receptor_ids"][index] for index in subset)


def independent_perturbation(
    model: dict[str, Any], noise_model: str, level: float, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    base = np.concatenate([model["linear"], model["quadratic"]])
    if noise_model == "none":
        changed = base.copy()
    elif noise_model == "round_to_nearest_full_scale":
        changed = np.rint(base / level) * level
    elif noise_model == "iid_gaussian_full_scale":
        changed = base + np.random.default_rng(seed).normal(0.0, level, len(base))
    else:
        raise ValueError(f"Stage71 unknown noise model: {noise_model}")
    delta = changed - base
    return (
        changed[: len(model["linear"])],
        changed[len(model["linear"]) :],
        {
            "hash": canonical_sha256([float(value) for value in changed]),
            "maximum_delta": float(np.max(np.abs(delta))),
            "rms_delta": float(np.sqrt(np.mean(delta * delta))),
            "zeroed_count": int(
                np.sum((np.abs(base) > 1e-12) & (changed == 0.0))
            ),
        },
    )


def decode(model: dict[str, Any], value: str) -> dict[str, Any]:
    bits = tuple(int(item) for item in value)
    if len(bits) != model["n"] + len(model["weights"]):
        raise ValueError("Stage71 stored bit string length differs")
    selected = tuple(index for index, bit in enumerate(bits[: model["n"]]) if bit)
    card = len(selected) - model["k"]
    quality = (
        sum(int(model["deficits"][index]) for index in selected)
        + sum(
            weight * bit
            for weight, bit in zip(model["weights"], bits[model["n"] :])
        )
        - model["maximum"]
    )
    feasible = card == 0 and quality == 0
    return {
        "bits": bits,
        "selected": selected,
        "name": name(model, selected),
        "card": card,
        "quality": quality,
        "feasible": feasible,
        "exact": feasible and selected == model["selected"],
    }


def bit_energy(
    model: dict[str, Any], bits: tuple[int, ...], linear: np.ndarray, quadratic: np.ndarray
) -> float:
    return float(
        model["constant"]
        + np.dot(np.asarray(bits, dtype=float), linear)
        + sum(
            value * bits[left] * bits[right]
            for (left, right), value in zip(model["pairs"], quadratic)
        )
    )


def independent_summaries(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str], list[dict[str, str]]] = {}
    for row in rows:
        model = str(row["noise_model"])
        level = float(row["noise_level"])
        groups.setdefault((model, level, "ALL"), []).append(row)
        groups.setdefault((model, level, str(row["target_id"])), []).append(row)
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
                "local_feasible_rate": statistics.fmean(
                    int(truth(row["local_best_feasible"])) for row in selected
                ),
                "local_exact_selected_rate": statistics.fmean(
                    int(truth(row["local_best_exact_selected"])) for row in selected
                ),
                "blind_best_feasible_rate": statistics.fmean(
                    int(truth(row["blind_best_feasible"])) for row in selected
                ),
                "blind_best_exact_selected_rate": statistics.fmean(
                    int(truth(row["blind_best_exact_selected"])) for row in selected
                ),
                "mean_blind_feasible_sample_fraction": statistics.fmean(
                    float(row["blind_feasible_sample_fraction"]) for row in selected
                ),
                "mean_blind_exact_sample_fraction": statistics.fmean(
                    float(row["blind_exact_sample_fraction"]) for row in selected
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


def compare_summaries(
    observed: list[dict[str, str]], expected: list[dict[str, Any]]
) -> None:
    if len(observed) != len(expected):
        raise ValueError("Stage71 summary row count differs")
    for first, second in zip(observed, expected):
        for key, value in second.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                close(first[key], value, f"noise summary:{key}")
            elif str(first[key]) != str(value):
                raise ValueError(f"Stage71 summary value differs: {key}")


def expected_trial_keys(config: dict[str, Any], models: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    output: set[tuple[Any, ...]] = set()
    noise = config["noise_screen"]
    base_seed = int(noise["coefficient_noise_seed_base"])
    for model_index, model in enumerate(models):
        target = model["record"]["target_id"]
        fold = int(model["record"]["outer_fold"])
        output.add((target, fold, "none", 0.0, 0, base_seed + model_index * 100_000))
        for index, level in enumerate(noise["quantization_steps"], start=1):
            output.add(
                (
                    target,
                    fold,
                    "round_to_nearest_full_scale",
                    float(level),
                    0,
                    base_seed + model_index * 100_000 + index * 1_000,
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
                        base_seed
                        + model_index * 100_000
                        + (offset + index) * 1_000
                        + repeat,
                    )
                )
    return output


def find_summary(
    summaries: list[dict[str, Any]], model: str, level: float, scope: str
) -> dict[str, Any]:
    values = [
        row
        for row in summaries
        if row["noise_model"] == model
        and math.isclose(float(row["noise_level"]), level, rel_tol=0.0, abs_tol=1e-15)
        and row["scope"] == scope
    ]
    if len(values) != 1:
        raise ValueError("Stage71 independent summary lookup differs")
    return values[0]


def independent_envelope(
    summaries: list[dict[str, Any]], config: dict[str, Any], model: str
) -> dict[str, Any]:
    gate = config["robustness_gate"]
    levels = sorted(
        {
            float(row["noise_level"])
            for row in summaries
            if row["noise_model"] == model and row["scope"] == "ALL"
        }
    )
    passing: list[float] = []
    for level in levels:
        overall = find_summary(summaries, model, level, "ALL")
        target_rows = [
            find_summary(summaries, model, level, target)
            for target in config["experiment"]["target_order"]
        ]
        if (
            float(overall["exact_selected_unique_rate"])
            >= float(gate["minimum_overall_exact_unique_rate"])
            and min(float(row["exact_selected_unique_rate"]) for row in target_rows)
            >= float(gate["minimum_worst_target_exact_unique_rate"])
            and float(overall["local_feasible_rate"])
            >= float(gate["minimum_local_feasible_rate"])
        ):
            passing.append(level)
    return {
        "noise_model": model,
        "largest_tested_level_passing_project_gate": max(passing) if passing else None,
        "passing_level_count": len(passing),
        "tested_level_count": len(levels),
    }


def run(
    config_path: Path, result_path: Path, root: Path, output_path: Path
) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path.resolve())
    result = read_json(result_path.resolve())
    if result.get("status") != "stage71_qubo_coefficient_noise_robustness_complete":
        raise ValueError("Stage71 source result did not complete")
    if checked(root, result["config"], "config").resolve() != config_path.resolve():
        raise ValueError("Stage71 result config differs")
    for key, value in config["implementation"].items():
        checked(root, value, key)
    for key, value in config["inputs"].items():
        checked(root, value, key)
    outputs = {key: checked(root, value, key) for key, value in result["outputs"].items()}
    model_record = read_json(root / config["inputs"]["stage70_model_record"]["path"])
    models = [independent_model(record) for record in model_record["models"]]
    states = [independent_states(model) for model in models]
    lookup = {
        (str(model["record"]["target_id"]), int(model["record"]["outer_fold"])): (
            model,
            current_states,
        )
        for model, current_states in zip(models, states)
    }
    landscape_rows = read_csv(outputs["exact_landscape_csv"])
    if len(landscape_rows) != len(models):
        raise ValueError("Stage71 exact landscape row count differs")
    finite_gaps: list[float] = []
    for row in landscape_rows:
        model, current_states = lookup[(row["target_id"], int(row["outer_fold"]))]
        optimum = exact_optimum(
            model, current_states, model["linear"], model["quadratic"]
        )
        second = next(
            (value[0] for value in optimum["rows"] if value[0] > optimum["rows"][0][0] + 1e-12),
            math.nan,
        )
        gap = second - optimum["rows"][0][0] if math.isfinite(second) else math.nan
        if math.isfinite(gap):
            finite_gaps.append(gap)
        if row["exact_best_subset"] != name(model, optimum["best_subset"]):
            raise ValueError("Stage71 exact landscape subset differs")
        if truth(row["selected_is_exact_optimum"]) != optimum["selected_optimal"]:
            raise ValueError("Stage71 exact source-optimum status differs")
        if truth(row["exact_optimum_unique"]) != (optimum["degeneracy"] == 1):
            raise ValueError("Stage71 exact degeneracy status differs")
        close(row["normalized_feasible_energy_gap"], gap, "landscape gap")
        close(row["raw_feasible_energy_gap"], gap * model["scale"], "raw landscape gap")
        if int(row["feasible_receptor_subset_count"]) != len(current_states["groups"]):
            raise ValueError("Stage71 feasible subset count differs")
        if int(row["feasible_binary_state_count"]) != len(current_states["assignments"]):
            raise ValueError("Stage71 feasible binary-state count differs")
    calibration_rows = read_csv(outputs["sampler_calibration_csv"])
    if len(calibration_rows) != len(models):
        raise ValueError("Stage71 calibration row count differs")
    for row in calibration_rows:
        model, _ = lookup[(row["target_id"], int(row["outer_fold"]))]
        decoded = decode(model, row["best_bits"])
        if row["best_subset"] != decoded["name"]:
            raise ValueError("Stage71 calibration subset differs")
        if truth(row["best_feasible"]) != decoded["feasible"]:
            raise ValueError("Stage71 calibration feasibility differs")
        if truth(row["best_exact_selected"]) != decoded["exact"]:
            raise ValueError("Stage71 calibration exact-selection status differs")
        if int(row["best_cardinality_residual"]) != decoded["card"]:
            raise ValueError("Stage71 calibration cardinality residual differs")
        if int(row["best_quality_residual"]) != decoded["quality"]:
            raise ValueError("Stage71 calibration quality residual differs")
        close(
            row["best_perturbed_energy"],
            bit_energy(model, decoded["bits"], model["linear"], model["quadratic"]),
            "calibration energy",
        )
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
        raise ValueError("Stage71 noise trial grid or seed schedule differs")
    for row in trial_rows:
        model, current_states = lookup[(row["target_id"], int(row["outer_fold"]))]
        linear, quadratic, perturbation = independent_perturbation(
            model,
            row["noise_model"],
            float(row["noise_level"]),
            int(row["coefficient_noise_seed"]),
        )
        if row["perturbed_coefficients_sha256"] != perturbation["hash"]:
            raise ValueError("Stage71 perturbation hash differs")
        close(
            row["maximum_absolute_coefficient_delta"],
            perturbation["maximum_delta"],
            "maximum coefficient delta",
        )
        close(row["rms_coefficient_delta"], perturbation["rms_delta"], "RMS delta")
        if int(row["zeroed_nonzero_coefficient_count"]) != perturbation["zeroed_count"]:
            raise ValueError("Stage71 zeroed coefficient count differs")
        optimum = exact_optimum(model, current_states, linear, quadratic)
        if row["exact_feasible_best_subset"] != name(model, optimum["best_subset"]):
            raise ValueError("Stage71 perturbed exact subset differs")
        if row["exact_feasible_best_bits"] != "".join(map(str, optimum["best_bits"])):
            raise ValueError("Stage71 perturbed exact bits differ")
        close(
            row["exact_feasible_best_perturbed_energy"],
            optimum["best_energy"],
            "perturbed exact energy",
        )
        if int(row["exact_feasible_optimum_degeneracy"]) != optimum["degeneracy"]:
            raise ValueError("Stage71 perturbed degeneracy differs")
        if truth(row["exact_selected_remains_optimal"]) != optimum["selected_optimal"]:
            raise ValueError("Stage71 perturbed selected-optimal status differs")
        if truth(row["exact_selected_remains_unique"]) != optimum["selected_unique"]:
            raise ValueError("Stage71 perturbed selected-unique status differs")
        close(row["exact_selected_subset_jaccard"], optimum["jaccard"], "jaccard")
        close(row["exact_base_objective_regret"], optimum["regret"], "objective regret")
        for prefix in ("local", "blind"):
            decoded = decode(model, row[f"{prefix}_best_bits"])
            if row[f"{prefix}_best_subset"] != decoded["name"]:
                raise ValueError(f"Stage71 {prefix} subset differs")
            if truth(row[f"{prefix}_best_feasible"]) != decoded["feasible"]:
                raise ValueError(f"Stage71 {prefix} feasibility differs")
            if truth(row[f"{prefix}_best_exact_selected"]) != decoded["exact"]:
                raise ValueError(f"Stage71 {prefix} exact-selection status differs")
            close(
                row[f"{prefix}_best_perturbed_energy"],
                bit_energy(model, decoded["bits"], linear, quadratic),
                f"{prefix} perturbed energy",
            )
    expected_summaries = independent_summaries(trial_rows)
    compare_summaries(read_csv(outputs["noise_summary_csv"]), expected_summaries)
    calibration = {
        "model_count": len(calibration_rows),
        "best_feasible_count": sum(truth(row["best_feasible"]) for row in calibration_rows),
        "best_exact_selected_count": sum(
            truth(row["best_exact_selected"]) for row in calibration_rows
        ),
    }
    calibration["best_feasible_rate"] = calibration["best_feasible_count"] / len(calibration_rows)
    calibration["best_exact_selected_rate"] = calibration["best_exact_selected_count"] / len(calibration_rows)
    calibration["calibration_gate_passed"] = bool(
        calibration["best_feasible_rate"]
        >= float(config["sampler_calibration_gate"]["minimum_best_feasible_rate"])
        and calibration["best_exact_selected_rate"]
        >= float(config["sampler_calibration_gate"]["minimum_best_exact_selected_rate"])
    )
    if calibration != result["sampler_calibration"]:
        raise ValueError("Stage71 calibration summary differs")
    landscape = {
        "model_count": len(landscape_rows),
        "unique_source_optimum_count": sum(
            truth(row["selected_is_exact_optimum"]) and truth(row["exact_optimum_unique"])
            for row in landscape_rows
        ),
        "finite_second_energy_count": len(finite_gaps),
        "minimum_normalized_feasible_energy_gap": min(finite_gaps),
        "maximum_normalized_feasible_energy_gap": max(finite_gaps),
        "minimum_raw_feasible_energy_gap": min(
            float(row["raw_feasible_energy_gap"])
            for row in landscape_rows
            if math.isfinite(float(row["raw_feasible_energy_gap"]))
        ),
        "maximum_feasible_receptor_subset_count": max(
            int(row["feasible_receptor_subset_count"]) for row in landscape_rows
        ),
    }
    for key, value in landscape.items():
        if isinstance(value, float):
            close(result["exact_landscape"][key], value, f"landscape summary:{key}")
        elif result["exact_landscape"][key] != value:
            raise ValueError(f"Stage71 landscape summary differs: {key}")
    envelopes = {
        model: independent_envelope(expected_summaries, config, model)
        for model in ("round_to_nearest_full_scale", "iid_gaussian_full_scale")
    }
    if envelopes != result["robustness_envelopes"]:
        raise ValueError("Stage71 robustness envelope differs")
    gate_config = config["robustness_gate"]
    reference_passes: dict[str, bool] = {}
    for key, model, level_key in (
        ("quantization", "round_to_nearest_full_scale", "reference_quantization_step"),
        ("gaussian", "iid_gaussian_full_scale", "reference_gaussian_sigma"),
    ):
        level = float(gate_config[level_key])
        overall = find_summary(expected_summaries, model, level, "ALL")
        targets = [
            find_summary(expected_summaries, model, level, target)
            for target in config["experiment"]["target_order"]
        ]
        reference_passes[key] = bool(
            float(overall["exact_selected_unique_rate"])
            >= float(gate_config["minimum_overall_exact_unique_rate"])
            and min(float(row["exact_selected_unique_rate"]) for row in targets)
            >= float(gate_config["minimum_worst_target_exact_unique_rate"])
            and float(overall["local_feasible_rate"])
            >= float(gate_config["minimum_local_feasible_rate"])
        )
    robust = bool(
        calibration["calibration_gate_passed"]
        and reference_passes["quantization"]
        and reference_passes["gaussian"]
    )
    if bool(result["robustness_gate"]["coefficient_robust_logical_bqm_gate_passed"]) != robust:
        raise ValueError("Stage71 robustness gate differs")
    if bool(result["decision"]["direct_qpu_execution_authorized"]):
        raise ValueError("Stage71 cannot authorize QPU after the failed Stage70 precision gate")
    if bool(result["decision"]["constraint_native_reformulation_authorized"]) != (not robust):
        raise ValueError("Stage71 constraint-native route decision differs")
    payload = {
        "exact_landscape": result["exact_landscape"],
        "sampler_calibration": result["sampler_calibration"],
        "robustness_envelopes": result["robustness_envelopes"],
        "robustness_gate": result["robustness_gate"],
    }
    if canonical_sha256(payload) != result["analysis_payload_sha256"]:
        raise ValueError("Stage71 analysis payload hash differs")
    audit = {
        "schema_version": "1.0",
        "status": "stage71_qubo_coefficient_noise_robustness_independent_audit_ok",
        "source_result": descriptor(root, result_path),
        "logical_models_independently_reconstructed": len(models),
        "exact_landscapes_independently_enumerated": len(landscape_rows),
        "calibration_samples_independently_redecoded": len(calibration_rows),
        "noise_trials_independently_recomputed": len(trial_rows),
        "noise_summaries_independently_recomputed": len(expected_summaries),
        "coefficient_robust_logical_bqm_gate_passed": robust,
        "direct_qpu_execution_authorized": False,
        "constraint_native_reformulation_authorized": not robust,
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
        default=Path("configs/stage71_qubo_coefficient_noise_robustness.json"),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("data/stage71_qubo_coefficient_noise_robustness_result.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage71_qubo_coefficient_noise_robustness_audit.json"),
    )
    args = parser.parse_args()
    run(args.config, args.result, args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
