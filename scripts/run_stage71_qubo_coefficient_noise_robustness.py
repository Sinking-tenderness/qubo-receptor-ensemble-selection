"""Measure coefficient-noise robustness of the frozen Stage70 logical QUBOs."""

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
from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSampler


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
        raise ValueError(f"Stage71 frozen {label} identity differs: {path}")
    if "size_bytes" in value and path.stat().st_size != int(value["size_bytes"]):
        raise ValueError(f"Stage71 frozen {label} size differs: {path}")
    return path


def runtime_versions() -> dict[str, str]:
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": np.__version__,
        "dimod": importlib.metadata.version("dimod"),
        "dwave_samplers": importlib.metadata.version("dwave-samplers"),
    }


def verify_runtime(config: dict[str, Any]) -> dict[str, str]:
    observed = runtime_versions()
    expected = config["software_sampler"]["runtime_versions"]
    for key, value in expected.items():
        if observed.get(key) != str(value):
            raise ValueError(
                f"Stage71 runtime version differs for {key}: "
                f"{observed.get(key)} != {value}"
            )
    return observed


def redundancy_matrix(record: dict[str, Any]) -> np.ndarray:
    receptor_count = len(record["receptor_ids"])
    matrix = np.zeros((receptor_count, receptor_count), dtype=float)
    values = iter(float(value) for value in record["stable_redundancy_upper_triangle"])
    for left, right in itertools.combinations(range(receptor_count), 2):
        value = next(values)
        matrix[left, right] = value
        matrix[right, left] = value
    try:
        next(values)
    except StopIteration:
        return matrix
    raise ValueError("Stage71 redundancy upper triangle has excess values")


def reconstruct_model(record: dict[str, Any]) -> dict[str, Any]:
    receptor_ids = [str(value) for value in record["receptor_ids"]]
    receptor_count = len(receptor_ids)
    slack_weights = [int(value) for value in record["slack_weights"]]
    slack_count = len(slack_weights)
    deficits = np.asarray(record["integer_deficits"], dtype=int)
    if len(deficits) != receptor_count:
        raise ValueError("Stage71 receptor and deficit counts differ")
    redundancy = redundancy_matrix(record)
    subset_size = int(record["reference_k"])
    maximum_deficit = int(record["maximum_integer_deficit"])
    center = int(record["integer_center"])
    centered_rhs = maximum_deficit - subset_size * center
    if centered_rhs != int(record["centered_rhs"]):
        raise ValueError("Stage71 centered right-hand side differs")
    penalty_k = float(record["cardinality_penalty"])
    penalty_q = float(record["quality_penalty"])
    centered = deficits.astype(float) - center
    slack = np.asarray(slack_weights, dtype=float)
    linear_parts = [
        penalty_k * (1 - 2 * subset_size)
        + penalty_q * (centered * centered - 2 * centered_rhs * centered)
    ]
    if slack_count:
        linear_parts.append(
            penalty_q * (slack * slack - 2 * centered_rhs * slack)
        )
    linear_raw = np.concatenate(linear_parts)
    pair_indices: list[tuple[int, int]] = []
    quadratic_parts: list[np.ndarray] = []
    receptor_pairs = list(itertools.combinations(range(receptor_count), 2))
    pair_indices.extend(receptor_pairs)
    receptor_upper = np.triu_indices(receptor_count, 1)
    quadratic_parts.append(
        2 * penalty_k
        + redundancy[receptor_upper]
        + 2
        * penalty_q
        * (centered[:, None] * centered[None, :])[receptor_upper]
    )
    if slack_count:
        receptor_slack_pairs = [
            (left, receptor_count + right)
            for left in range(receptor_count)
            for right in range(slack_count)
        ]
        pair_indices.extend(receptor_slack_pairs)
        quadratic_parts.append(
            (2 * penalty_q * centered[:, None] * slack[None, :]).ravel()
        )
        slack_pairs = list(itertools.combinations(range(slack_count), 2))
        pair_indices.extend(
            (receptor_count + left, receptor_count + right)
            for left, right in slack_pairs
        )
        slack_upper = np.triu_indices(slack_count, 1)
        quadratic_parts.append(
            (2 * penalty_q * slack[:, None] * slack[None, :])[slack_upper]
        )
    quadratic_raw = np.concatenate(quadratic_parts)
    constant_raw = float(
        penalty_k * subset_size**2 + penalty_q * centered_rhs**2
    )
    source_hash = canonical_sha256(
        {
            "center": center,
            "centered_rhs": centered_rhs,
            "linear": [float(value) for value in linear_raw],
            "quadratic": [float(value) for value in quadratic_raw],
            "constant": constant_raw,
        }
    )
    if source_hash != str(record["qubo_summary"]["qubo_sha256"]).upper():
        raise ValueError("Stage71 reconstructed QUBO hash differs from Stage70")
    scale = float(np.max(np.abs(np.concatenate([linear_raw, quadratic_raw]))))
    if not math.isclose(
        scale,
        float(record["qubo_summary"]["maximum_absolute_coefficient"]),
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("Stage71 reconstructed QUBO scale differs from Stage70")
    variables = [f"x{index:03d}" for index in range(receptor_count)] + [
        f"s{index:03d}" for index in range(slack_count)
    ]
    selected_names = str(record["selected_subset"]).split("+")
    lookup = {value: index for index, value in enumerate(receptor_ids)}
    selected_indices = tuple(sorted(lookup[value] for value in selected_names))
    return {
        "record": record,
        "receptor_ids": receptor_ids,
        "receptor_count": receptor_count,
        "slack_weights": slack_weights,
        "slack_count": slack_count,
        "deficits": deficits,
        "redundancy": redundancy,
        "subset_size": subset_size,
        "maximum_deficit": maximum_deficit,
        "variables": variables,
        "pair_indices": pair_indices,
        "linear": linear_raw / scale,
        "quadratic": quadratic_raw / scale,
        "constant": constant_raw / scale,
        "normalization_scale": scale,
        "selected_indices": selected_indices,
        "source_qubo_sha256": source_hash,
    }


def build_feasible_states(model: dict[str, Any]) -> dict[str, Any]:
    weights = model["slack_weights"]
    patterns: dict[int, list[tuple[int, ...]]] = {}
    for bits in itertools.product((0, 1), repeat=len(weights)):
        value = sum(weight * bit for weight, bit in zip(weights, bits))
        patterns.setdefault(value, []).append(bits)
    assignments: list[list[int]] = []
    subsets: list[tuple[int, ...]] = []
    state_groups: dict[tuple[int, ...], list[int]] = {}
    receptor_count = int(model["receptor_count"])
    for subset in itertools.combinations(
        range(receptor_count), int(model["subset_size"])
    ):
        deficit = int(np.sum(model["deficits"][list(subset)]))
        if deficit > int(model["maximum_deficit"]):
            continue
        slack_value = int(model["maximum_deficit"]) - deficit
        if slack_value not in patterns:
            raise ValueError("Stage71 feasible slack value is not representable")
        group: list[int] = []
        for slack_bits in patterns[slack_value]:
            receptor_bits = [int(index in subset) for index in range(receptor_count)]
            group.append(len(assignments))
            assignments.append(receptor_bits + list(slack_bits))
            subsets.append(subset)
        state_groups[subset] = group
    if not assignments:
        raise ValueError("Stage71 model has no feasible fixed-k state")
    assignment_matrix = np.asarray(assignments, dtype=np.uint8)
    left = np.asarray([pair[0] for pair in model["pair_indices"]], dtype=int)
    right = np.asarray([pair[1] for pair in model["pair_indices"]], dtype=int)
    quadratic_features = assignment_matrix[:, left] * assignment_matrix[:, right]
    return {
        "assignments": assignment_matrix,
        "quadratic_features": quadratic_features,
        "subsets": subsets,
        "groups": state_groups,
    }


def state_energies(
    model: dict[str, Any],
    states: dict[str, Any],
    linear: np.ndarray,
    quadratic: np.ndarray,
) -> np.ndarray:
    return (
        float(model["constant"])
        + states["assignments"] @ linear
        + states["quadratic_features"] @ quadratic
    )


def exact_feasible_optimum(
    model: dict[str, Any],
    states: dict[str, Any],
    linear: np.ndarray,
    quadratic: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    energies = state_energies(model, states, linear, quadratic)
    subset_rows: list[tuple[float, tuple[int, ...], int]] = []
    for subset, indices in states["groups"].items():
        local = min(indices, key=lambda index: (float(energies[index]), index))
        subset_rows.append((float(energies[local]), subset, local))
    subset_rows.sort(key=lambda row: (row[0], row[1]))
    best_energy = subset_rows[0][0]
    optimal = [row for row in subset_rows if row[0] <= best_energy + tolerance]
    selected = model["selected_indices"]
    selected_row = next(row for row in subset_rows if row[1] == selected)
    best_row = min(optimal, key=lambda row: row[1])
    objective_best = min(
        float(
            sum(
                model["redundancy"][left, right]
                for left, right in itertools.combinations(row[1], 2)
            )
        )
        for row in optimal
    )
    selected_set = set(selected)
    best_set = set(best_row[1])
    jaccard = len(selected_set & best_set) / len(selected_set | best_set)
    return {
        "best_energy": best_energy,
        "best_subset": best_row[1],
        "best_assignment": tuple(
            int(value) for value in states["assignments"][best_row[2]]
        ),
        "optimal_subset_count": len(optimal),
        "selected_remains_optimal": any(row[1] == selected for row in optimal),
        "selected_remains_unique": len(optimal) == 1 and optimal[0][1] == selected,
        "selected_energy": selected_row[0],
        "selected_assignment": tuple(
            int(value) for value in states["assignments"][selected_row[2]]
        ),
        "selected_subset_jaccard": jaccard,
        "base_objective_regret": objective_best
        - float(model["record"]["selected_redundancy_sum"]),
        "subset_rows": subset_rows,
    }


def subset_name(model: dict[str, Any], subset: tuple[int, ...]) -> str:
    return "+".join(model["receptor_ids"][index] for index in subset)


def perturb_coefficients(
    model: dict[str, Any], noise_model: str, level: float, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    linear = np.asarray(model["linear"], dtype=float).copy()
    quadratic = np.asarray(model["quadratic"], dtype=float).copy()
    source = np.concatenate([linear, quadratic])
    if noise_model == "none":
        changed = source.copy()
    elif noise_model == "round_to_nearest_full_scale":
        if level <= 0:
            raise ValueError("Stage71 quantization step must be positive")
        changed = np.rint(source / level) * level
    elif noise_model == "iid_gaussian_full_scale":
        if level <= 0:
            raise ValueError("Stage71 Gaussian sigma must be positive")
        rng = np.random.default_rng(seed)
        changed = source + rng.normal(0.0, level, size=len(source))
    else:
        raise ValueError(f"unknown Stage71 noise model: {noise_model}")
    delta = changed - source
    linear_changed = changed[: len(linear)]
    quadratic_changed = changed[len(linear) :]
    metadata = {
        "perturbed_coefficients_sha256": canonical_sha256(
            [float(value) for value in changed]
        ),
        "maximum_absolute_coefficient_delta": float(np.max(np.abs(delta))),
        "rms_coefficient_delta": float(np.sqrt(np.mean(delta * delta))),
        "zeroed_nonzero_coefficient_count": int(
            np.sum((np.abs(source) > TOLERANCE) & (changed == 0.0))
        ),
    }
    return linear_changed, quadratic_changed, metadata


def make_bqm(
    model: dict[str, Any], linear: np.ndarray, quadratic: np.ndarray
) -> dimod.BinaryQuadraticModel:
    linear_biases = {
        variable: float(linear[index])
        for index, variable in enumerate(model["variables"])
    }
    quadratic_biases = {
        (model["variables"][left], model["variables"][right]): float(value)
        for (left, right), value in zip(model["pair_indices"], quadratic)
    }
    return dimod.BinaryQuadraticModel(
        linear_biases,
        quadratic_biases,
        float(model["constant"]),
        dimod.BINARY,
    )


def bits_to_string(bits: tuple[int, ...]) -> str:
    return "".join(str(int(value)) for value in bits)


def sample_to_bits(model: dict[str, Any], sample: Any) -> tuple[int, ...]:
    return tuple(int(sample[variable]) for variable in model["variables"])


def decode_bits(model: dict[str, Any], bits: tuple[int, ...]) -> dict[str, Any]:
    receptor_count = int(model["receptor_count"])
    receptor_bits = bits[:receptor_count]
    slack_bits = bits[receptor_count:]
    selected = tuple(index for index, value in enumerate(receptor_bits) if value)
    cardinality_residual = len(selected) - int(model["subset_size"])
    deficit_sum = int(
        sum(int(model["deficits"][index]) for index in selected)
    )
    slack_value = sum(
        weight * bit for weight, bit in zip(model["slack_weights"], slack_bits)
    )
    quality_residual = deficit_sum + slack_value - int(model["maximum_deficit"])
    feasible = cardinality_residual == 0 and quality_residual == 0
    return {
        "subset": selected,
        "subset_name": subset_name(model, selected),
        "cardinality_residual": cardinality_residual,
        "quality_residual": quality_residual,
        "feasible": feasible,
        "exact_selected": feasible and selected == model["selected_indices"],
    }


def sampler_best(
    model: dict[str, Any],
    bqm: dimod.BinaryQuadraticModel,
    sampler: SimulatedAnnealingSampler,
    descent: SteepestDescentSampler,
    num_reads: int,
    num_sweeps: int,
    beta_range: list[float],
    seed: int,
) -> dict[str, Any]:
    sampled = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_range=beta_range,
        beta_schedule_type="geometric",
        seed=seed,
    )
    polished = descent.sample(bqm, initial_states=sampled)
    rows = list(
        polished.data(fields=["sample", "energy", "num_occurrences"], sorted_by="energy")
    )
    best = rows[0]
    best_bits = sample_to_bits(model, best.sample)
    best_decoded = decode_bits(model, best_bits)
    total = sum(int(row.num_occurrences) for row in rows)
    feasible_count = 0
    exact_count = 0
    for row in rows:
        decoded = decode_bits(model, sample_to_bits(model, row.sample))
        feasible_count += int(row.num_occurrences) * int(decoded["feasible"])
        exact_count += int(row.num_occurrences) * int(decoded["exact_selected"])
    return {
        "bits": best_bits,
        "energy": float(best.energy),
        "decoded": best_decoded,
        "feasible_sample_fraction": feasible_count / total,
        "exact_sample_fraction": exact_count / total,
    }


def local_descent(
    model: dict[str, Any],
    bqm: dimod.BinaryQuadraticModel,
    descent: SteepestDescentSampler,
    initial_bits: tuple[int, ...],
) -> dict[str, Any]:
    initial = {
        variable: int(initial_bits[index])
        for index, variable in enumerate(model["variables"])
    }
    samples = descent.sample(bqm, initial_states=[initial])
    row = next(iter(samples.data(fields=["sample", "energy"], sorted_by="energy")))
    bits = sample_to_bits(model, row.sample)
    return {
        "bits": bits,
        "energy": float(row.energy),
        "decoded": decode_bits(model, bits),
    }


def coefficient_seed(
    base_seed: int, model_index: int, level_index: int, repeat: int
) -> int:
    return int(base_seed + model_index * 100_000 + level_index * 1_000 + repeat)


def sampler_seed(coefficient_noise_seed: int, offset: int) -> int:
    return int((coefficient_noise_seed + offset) % (2**31 - 1))


def landscape_row(model: dict[str, Any], states: dict[str, Any]) -> dict[str, Any]:
    optimum = exact_feasible_optimum(
        model,
        states,
        model["linear"],
        model["quadratic"],
        TOLERANCE,
    )
    rows = optimum["subset_rows"]
    second = next(
        (row[0] for row in rows if row[0] > rows[0][0] + TOLERANCE),
        math.nan,
    )
    gap = second - rows[0][0] if math.isfinite(second) else math.nan
    return {
        "target_id": model["record"]["target_id"],
        "outer_fold": int(model["record"]["outer_fold"]),
        "receptor_count": int(model["receptor_count"]),
        "slack_variable_count": int(model["slack_count"]),
        "logical_variable_count": len(model["variables"]),
        "quadratic_coefficient_count": len(model["quadratic"]),
        "normalization_scale": float(model["normalization_scale"]),
        "feasible_receptor_subset_count": len(states["groups"]),
        "feasible_binary_state_count": len(states["assignments"]),
        "selected_subset": subset_name(model, model["selected_indices"]),
        "exact_best_subset": subset_name(model, optimum["best_subset"]),
        "selected_is_exact_optimum": bool(optimum["selected_remains_optimal"]),
        "exact_optimum_unique": int(optimum["optimal_subset_count"]) == 1,
        "exact_optimum_degeneracy": int(optimum["optimal_subset_count"]),
        "normalized_best_energy": float(rows[0][0]),
        "normalized_second_distinct_energy": float(second),
        "normalized_feasible_energy_gap": float(gap),
        "raw_feasible_energy_gap": float(gap * model["normalization_scale"])
        if math.isfinite(gap)
        else math.nan,
        "source_qubo_sha256": model["source_qubo_sha256"],
    }


def calibration_row(
    model: dict[str, Any],
    sampler: SimulatedAnnealingSampler,
    descent: SteepestDescentSampler,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    bqm = make_bqm(model, model["linear"], model["quadratic"])
    settings = config["software_sampler"]["zero_noise_calibration"]
    best = sampler_best(
        model,
        bqm,
        sampler,
        descent,
        int(settings["num_reads"]),
        int(settings["num_sweeps"]),
        [float(value) for value in settings["beta_range"]],
        seed,
    )
    decoded = best["decoded"]
    return {
        "target_id": model["record"]["target_id"],
        "outer_fold": int(model["record"]["outer_fold"]),
        "sampler_seed": seed,
        "num_reads": int(settings["num_reads"]),
        "num_sweeps": int(settings["num_sweeps"]),
        "best_subset": decoded["subset_name"],
        "best_bits": bits_to_string(best["bits"]),
        "best_perturbed_energy": best["energy"],
        "best_feasible": decoded["feasible"],
        "best_exact_selected": decoded["exact_selected"],
        "best_cardinality_residual": decoded["cardinality_residual"],
        "best_quality_residual": decoded["quality_residual"],
        "feasible_sample_fraction": best["feasible_sample_fraction"],
        "exact_sample_fraction": best["exact_sample_fraction"],
    }


def trial_row(
    model: dict[str, Any],
    states: dict[str, Any],
    sampler: SimulatedAnnealingSampler,
    descent: SteepestDescentSampler,
    config: dict[str, Any],
    noise_model: str,
    level: float,
    repeat: int,
    noise_seed: int,
) -> dict[str, Any]:
    linear, quadratic, perturbation = perturb_coefficients(
        model, noise_model, level, noise_seed
    )
    exact = exact_feasible_optimum(
        model, states, linear, quadratic, TOLERANCE
    )
    bqm = make_bqm(model, linear, quadratic)
    local = local_descent(model, bqm, descent, exact["selected_assignment"])
    settings = config["software_sampler"]["noise_trials"]
    blind = sampler_best(
        model,
        bqm,
        sampler,
        descent,
        int(settings["num_reads"]),
        int(settings["num_sweeps"]),
        [float(value) for value in settings["beta_range"]],
        sampler_seed(noise_seed, int(settings["sampler_seed_offset"])),
    )
    exact_decoded = exact["best_subset"]
    local_decoded = local["decoded"]
    blind_decoded = blind["decoded"]
    return {
        "target_id": model["record"]["target_id"],
        "outer_fold": int(model["record"]["outer_fold"]),
        "noise_model": noise_model,
        "noise_level": float(level),
        "repeat": repeat,
        "coefficient_noise_seed": noise_seed,
        "sampler_seed": sampler_seed(
            noise_seed, int(settings["sampler_seed_offset"])
        ),
        **perturbation,
        "exact_feasible_best_subset": subset_name(model, exact_decoded),
        "exact_feasible_best_bits": bits_to_string(exact["best_assignment"]),
        "exact_feasible_best_perturbed_energy": exact["best_energy"],
        "exact_feasible_optimum_degeneracy": exact["optimal_subset_count"],
        "exact_selected_remains_optimal": exact["selected_remains_optimal"],
        "exact_selected_remains_unique": exact["selected_remains_unique"],
        "exact_selected_subset_jaccard": exact["selected_subset_jaccard"],
        "exact_base_objective_regret": exact["base_objective_regret"],
        "local_best_subset": local_decoded["subset_name"],
        "local_best_bits": bits_to_string(local["bits"]),
        "local_best_perturbed_energy": local["energy"],
        "local_best_feasible": local_decoded["feasible"],
        "local_best_exact_selected": local_decoded["exact_selected"],
        "local_cardinality_residual": local_decoded["cardinality_residual"],
        "local_quality_residual": local_decoded["quality_residual"],
        "blind_best_subset": blind_decoded["subset_name"],
        "blind_best_bits": bits_to_string(blind["bits"]),
        "blind_best_perturbed_energy": blind["energy"],
        "blind_best_feasible": blind_decoded["feasible"],
        "blind_best_exact_selected": blind_decoded["exact_selected"],
        "blind_cardinality_residual": blind_decoded["cardinality_residual"],
        "blind_quality_residual": blind_decoded["quality_residual"],
        "blind_feasible_sample_fraction": blind["feasible_sample_fraction"],
        "blind_exact_sample_fraction": blind["exact_sample_fraction"],
    }


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def summarize_trials(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    targets = sorted({str(row["target_id"]) for row in rows})
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
                "local_feasible_rate": statistics.fmean(
                    int(truth(row["local_best_feasible"])) for row in selected
                ),
                "local_exact_selected_rate": statistics.fmean(
                    int(truth(row["local_best_exact_selected"]))
                    for row in selected
                ),
                "blind_best_feasible_rate": statistics.fmean(
                    int(truth(row["blind_best_feasible"])) for row in selected
                ),
                "blind_best_exact_selected_rate": statistics.fmean(
                    int(truth(row["blind_best_exact_selected"]))
                    for row in selected
                ),
                "mean_blind_feasible_sample_fraction": statistics.fmean(
                    float(row["blind_feasible_sample_fraction"])
                    for row in selected
                ),
                "mean_blind_exact_sample_fraction": statistics.fmean(
                    float(row["blind_exact_sample_fraction"])
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
    if not targets:
        raise ValueError("Stage71 summary has no targets")
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
            f"Stage71 summary lookup differs: {noise_model}/{level}/{scope}"
        )
    return matches[0]


def robustness_envelope(
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
    noise_model: str,
) -> dict[str, Any]:
    gate = config["robustness_gate"]
    levels = sorted(
        {
            float(row["noise_level"])
            for row in summaries
            if row["noise_model"] == noise_model and row["scope"] == "ALL"
        }
    )
    passing: list[float] = []
    for level in levels:
        overall = summary_at(summaries, noise_model, level, "ALL")
        targets = [
            row
            for row in summaries
            if row["noise_model"] == noise_model
            and math.isclose(
                float(row["noise_level"]), level, rel_tol=0.0, abs_tol=1e-15
            )
            and row["scope"] != "ALL"
        ]
        if (
            float(overall["exact_selected_unique_rate"])
            >= float(gate["minimum_overall_exact_unique_rate"])
            and min(float(row["exact_selected_unique_rate"]) for row in targets)
            >= float(gate["minimum_worst_target_exact_unique_rate"])
            and float(overall["local_feasible_rate"])
            >= float(gate["minimum_local_feasible_rate"])
        ):
            passing.append(level)
    return {
        "noise_model": noise_model,
        "largest_tested_level_passing_project_gate": max(passing)
        if passing
        else None,
        "passing_level_count": len(passing),
        "tested_level_count": len(levels),
    }


def report_text(result: dict[str, Any]) -> str:
    landscape = result["exact_landscape"]
    calibration = result["sampler_calibration"]
    gate = result["robustness_gate"]
    quant = result["robustness_envelopes"]["round_to_nearest_full_scale"]
    gaussian = result["robustness_envelopes"]["iid_gaussian_full_scale"]
    return rf"""# Stage71 QUBO coefficient-noise robustness

## Question

Does the frozen Stage70 logical BQM preserve its selected receptor subset after full-scale coefficient quantization or independent additive noise, and can a fixed classical annealing budget recover the unperturbed solution?

## Separation of effects

Stage71 reports two distinct quantities. Exact enumeration over every fixed-$k$ quality-feasible receptor subset measures whether perturbed coefficients change the mathematical optimum. Full-BQM simulated annealing plus steepest descent separately measures finite-budget recovery and constraint leakage. The latter is a software diagnostic, not a quantum-hardware benchmark.

## Exact landscape

- Frozen models checked: `{landscape['model_count']}`.
- Unique Stage70 optima recovered exactly: `{landscape['unique_source_optimum_count']}/{landscape['model_count']}`.
- Minimum normalized feasible energy gap: `{landscape['minimum_normalized_feasible_energy_gap']:.6g}`.
- Maximum normalized feasible energy gap: `{landscape['maximum_normalized_feasible_energy_gap']:.6g}`.

## Sampler calibration

- Zero-noise best feasible recovery: `{calibration['best_feasible_count']}/{calibration['model_count']}`.
- Zero-noise exact-subset recovery: `{calibration['best_exact_selected_count']}/{calibration['model_count']}`.
- Calibration gate passed: `{calibration['calibration_gate_passed']}`.

## Noise envelope

- Quantization: largest tested full-scale step passing the project gate = `{quant['largest_tested_level_passing_project_gate']}`.
- Gaussian noise: largest tested full-scale sigma passing the project gate = `{gaussian['largest_tested_level_passing_project_gate']}`.
- Reference quantization gate passed: `{gate['reference_quantization_gate_passed']}`.
- Reference Gaussian gate passed: `{gate['reference_gaussian_gate_passed']}`.
- Coefficient-robust logical BQM gate passed: `{gate['coefficient_robust_logical_bqm_gate_passed']}`.

## Decision boundary

- Direct QPU execution authorized: `{result['decision']['direct_qpu_execution_authorized']}`.
- Constraint-native reformulation authorized: `{result['decision']['constraint_native_reformulation_authorized']}`.
- New-target preregistration remains authorized: `{result['decision']['new_target_preregistration_remains_authorized']}`.
- Quantum-advantage claim authorized: `{result['decision']['quantum_advantage_claim_authorized']}`.

The tested noise levels are project sensitivity probes, not specifications for any physical annealer. This post-hoc analysis uses four consumed development targets and cannot establish independent efficacy, embedding quality, hardware speedup, or quantum advantage.
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
    stage70_result = read_json(input_paths["stage70_result"])
    stage70_audit = read_json(input_paths["stage70_audit"])
    if not stage70_result["decision"]["coefficient_noise_simulation_authorized"]:
        raise ValueError("Stage71 requires Stage70 noise-simulation authorization")
    if stage70_audit.get("status") != (
        "stage70_constraint_aware_qubo_encoding_independent_audit_ok"
    ):
        raise ValueError("Stage71 requires the Stage70 independent audit")
    model_record = read_json(input_paths["stage70_model_record"])
    if int(model_record["model_count"]) != int(
        config["experiment"]["required_model_count"]
    ):
        raise ValueError("Stage71 frozen model count differs")
    models = [reconstruct_model(record) for record in model_record["models"]]
    states = [build_feasible_states(model) for model in models]
    landscape_rows = [
        landscape_row(model, model_states)
        for model, model_states in zip(models, states)
    ]
    if not all(row["selected_is_exact_optimum"] for row in landscape_rows):
        raise ValueError("Stage71 source selection is not an exact feasible optimum")
    sampler = SimulatedAnnealingSampler()
    descent = SteepestDescentSampler()
    calibration_base_seed = int(
        config["software_sampler"]["zero_noise_calibration"]["seed_base"]
    )
    calibration_rows = [
        calibration_row(
            model,
            sampler,
            descent,
            config,
            calibration_base_seed + index,
        )
        for index, model in enumerate(models)
    ]
    trial_rows: list[dict[str, Any]] = []
    noise = config["noise_screen"]
    base_seed = int(noise["coefficient_noise_seed_base"])
    for model_index, (model, model_states) in enumerate(zip(models, states)):
        trial_rows.append(
            trial_row(
                model,
                model_states,
                sampler,
                descent,
                config,
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
                    model_states,
                    sampler,
                    descent,
                    config,
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
                        model_states,
                        sampler,
                        descent,
                        config,
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
    output_paths = {
        key: root / str(value) for key, value in config["outputs"].items()
    }
    write_csv(output_paths["exact_landscape_csv"], landscape_rows)
    write_csv(output_paths["sampler_calibration_csv"], calibration_rows)
    write_csv(output_paths["noise_trials_csv"], trial_rows)
    write_csv(output_paths["noise_summary_csv"], summaries)
    finite_gaps = [
        float(row["normalized_feasible_energy_gap"])
        for row in landscape_rows
        if math.isfinite(float(row["normalized_feasible_energy_gap"]))
    ]
    exact_landscape = {
        "model_count": len(landscape_rows),
        "unique_source_optimum_count": sum(
            bool(row["selected_is_exact_optimum"])
            and bool(row["exact_optimum_unique"])
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
    calibration_gate = config["sampler_calibration_gate"]
    best_feasible_count = sum(truth(row["best_feasible"]) for row in calibration_rows)
    best_exact_count = sum(
        truth(row["best_exact_selected"]) for row in calibration_rows
    )
    calibration = {
        "model_count": len(calibration_rows),
        "best_feasible_count": best_feasible_count,
        "best_exact_selected_count": best_exact_count,
        "best_feasible_rate": best_feasible_count / len(calibration_rows),
        "best_exact_selected_rate": best_exact_count / len(calibration_rows),
    }
    calibration["calibration_gate_passed"] = bool(
        calibration["best_feasible_rate"]
        >= float(calibration_gate["minimum_best_feasible_rate"])
        and calibration["best_exact_selected_rate"]
        >= float(calibration_gate["minimum_best_exact_selected_rate"])
    )
    envelopes = {
        noise_model: robustness_envelope(summaries, config, noise_model)
        for noise_model in (
            "round_to_nearest_full_scale",
            "iid_gaussian_full_scale",
        )
    }
    gate = config["robustness_gate"]
    reference_results: dict[str, dict[str, Any]] = {}
    for key, noise_model, level_key in (
        (
            "quantization",
            "round_to_nearest_full_scale",
            "reference_quantization_step",
        ),
        ("gaussian", "iid_gaussian_full_scale", "reference_gaussian_sigma"),
    ):
        level = float(gate[level_key])
        overall = summary_at(summaries, noise_model, level, "ALL")
        target_rows = [
            summary_at(summaries, noise_model, level, target)
            for target in config["experiment"]["target_order"]
        ]
        reference_results[key] = {
            "noise_model": noise_model,
            "noise_level": level,
            "overall_exact_unique_rate": float(
                overall["exact_selected_unique_rate"]
            ),
            "worst_target_exact_unique_rate": min(
                float(row["exact_selected_unique_rate"]) for row in target_rows
            ),
            "overall_local_feasible_rate": float(overall["local_feasible_rate"]),
        }
        reference_results[key]["gate_passed"] = bool(
            reference_results[key]["overall_exact_unique_rate"]
            >= float(gate["minimum_overall_exact_unique_rate"])
            and reference_results[key]["worst_target_exact_unique_rate"]
            >= float(gate["minimum_worst_target_exact_unique_rate"])
            and reference_results[key]["overall_local_feasible_rate"]
            >= float(gate["minimum_local_feasible_rate"])
        )
    robustness_gate = {
        "reference_quantization": reference_results["quantization"],
        "reference_gaussian": reference_results["gaussian"],
        "reference_quantization_gate_passed": reference_results["quantization"][
            "gate_passed"
        ],
        "reference_gaussian_gate_passed": reference_results["gaussian"][
            "gate_passed"
        ],
    }
    robustness_gate["coefficient_robust_logical_bqm_gate_passed"] = bool(
        calibration["calibration_gate_passed"]
        and robustness_gate["reference_quantization_gate_passed"]
        and robustness_gate["reference_gaussian_gate_passed"]
    )
    payload = {
        "exact_landscape": exact_landscape,
        "sampler_calibration": calibration,
        "robustness_envelopes": envelopes,
        "robustness_gate": robustness_gate,
    }
    result = {
        "schema_version": "1.0",
        "status": "stage71_qubo_coefficient_noise_robustness_complete",
        "experiment_class": (
            "post-hoc coefficient-noise and finite-budget software-sampler "
            "diagnosis on frozen historical logical BQMs"
        ),
        "config": descriptor(
            root, root / "configs/stage71_qubo_coefficient_noise_robustness.json"
        ),
        "implementation": {
            key: descriptor(root, path) for key, path in implementation_paths.items()
        },
        "inputs": {key: descriptor(root, path) for key, path in input_paths.items()},
        "runtime_versions": versions,
        "exact_landscape": exact_landscape,
        "sampler_calibration": calibration,
        "noise_trial_count": len(trial_rows),
        "noise_summary_count": len(summaries),
        "robustness_envelopes": envelopes,
        "robustness_gate": robustness_gate,
        "decision": {
            "direct_qpu_execution_authorized": bool(
                stage70_result["decision"]["direct_qpu_execution_authorized"]
                and robustness_gate["coefficient_robust_logical_bqm_gate_passed"]
            ),
            "constraint_native_reformulation_authorized": not robustness_gate[
                "coefficient_robust_logical_bqm_gate_passed"
            ],
            "new_target_preregistration_remains_authorized": stage70_result[
                "decision"
            ]["new_target_preregistration_remains_authorized"],
            "quantum_advantage_claim_authorized": False,
            "next_action": (
                "evaluate a constraint-native CQM or feasible-subspace formulation before any hardware job"
                if not robustness_gate["coefficient_robust_logical_bqm_gate_passed"]
                else "retain the frozen logical BQM for a separate embedding feasibility study"
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
            "exact_landscape_csv",
            "sampler_calibration_csv",
            "noise_trials_csv",
            "noise_summary_csv",
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
    expected = root / "configs/stage71_qubo_coefficient_noise_robustness.json"
    if config_path != expected.resolve():
        raise ValueError("Stage71 must run from its frozen repository config")
    config = read_json(config_path)
    result_path = root / str(config["outputs"]["result_json"])
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"Stage71 result exists: {result_path}")
    result = compute(config, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage71_qubo_coefficient_noise_robustness.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
