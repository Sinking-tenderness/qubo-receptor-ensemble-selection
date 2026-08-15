"""Run a label-independent structure-aware conformation-pool QUBO screen.

This stage deliberately reads structural eligibility, invariant pocket features,
and pairwise structural distances only.  It does not read ligand labels,
docking scores, validation rows, test rows, or quantum-hardware results.

The optimization model is

    Q(x) = A (sum_i x_i - k)^2
           - lambda_distance * sum_{i<j} d_ij x_i x_j
           + lambda_quality * sum_i q_i x_i,

where d_ij is a normalized structural distance and q_i is an optional,
label-independent preparation-quality penalty.  The default configuration sets
lambda_quality to zero so that the first concept validation isolates the
structural-diversity term.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np


METHODS = ("qubo_swap_local_search", "maxmin_seeded", "maxsum_greedy")
PROTECTED_MARKERS = (
    "fresh_validation",
    "locked_test",
    "test_rows",
    "bace1_docking",
    "quantum_hardware",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def rooted(root: Path, relative: str) -> Path:
    path = (root / relative.replace("\\", "/")).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"input path leaves repository root: {relative}") from error
    return path


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def numeric(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"non-numeric {key} value: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {key} value: {value!r}")
    return parsed


def minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high):
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def load_target(root: Path, target_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    paths: dict[str, Path] = {}
    for key, relative in spec["inputs"].items():
        if not isinstance(relative, str):
            raise ValueError(f"{target_id}/{key} input path is not a string")
        if any(marker in relative.lower() for marker in PROTECTED_MARKERS):
            raise ValueError(f"protected data path is not allowed: {relative}")
        path = rooted(root, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[key] = path

    pool_rows = read_csv(paths["eligible_pool"])
    audit_rows = read_csv(paths["coordinate_audit"])
    feature_rows = read_csv(paths["feature_matrix"])
    distance_rows = read_csv(paths["pairwise_distances"])

    audit_by_id = {row["conformer_id"]: row for row in audit_rows}
    if len(audit_by_id) != len(audit_rows):
        raise ValueError(f"{target_id}: duplicate coordinate-audit ID")

    pool_by_id: dict[str, dict[str, str]] = {}
    excluded: dict[str, str] = {}
    for row in pool_rows:
        conformer_id = row["conformer_id"]
        if conformer_id in pool_by_id:
            raise ValueError(f"{target_id}: duplicate eligible-pool ID {conformer_id}")
        audit = audit_by_id.get(conformer_id)
        if audit is None or audit.get("status") != "coordinate_eligible":
            raise ValueError(f"{target_id}: pool row lacks a passing coordinate audit")
        # A structure with a globally incomplete residue cannot be repaired by
        # a combinatorial optimizer, so it remains a hard feasibility gate.
        incomplete = int(round(numeric(audit, "global_incomplete_standard_amino_acid_residue_count", 0.0)))
        if incomplete > 0:
            excluded[conformer_id] = "global_incomplete_standard_amino_acid_residue"
            continue
        pool_by_id[conformer_id] = row

    if len(pool_by_id) < 2:
        raise ValueError(f"{target_id}: fewer than two preparation-ready candidates")

    feature_by_id = {row["conformer_id"]: row for row in feature_rows}
    if len(feature_by_id) != len(feature_rows):
        raise ValueError(f"{target_id}: duplicate structural-feature ID")
    missing_features = sorted(set(pool_by_id) - set(feature_by_id))
    if missing_features:
        raise ValueError(f"{target_id}: missing structural features: {missing_features[:3]}")
    feature_columns = [key for key in feature_rows[0] if key != "conformer_id"]
    for conformer_id in pool_by_id:
        for key in feature_columns:
            numeric(feature_by_id[conformer_id], key)

    ids = sorted(pool_by_id)
    id_set = set(ids)
    distances: dict[tuple[str, str], float] = {}
    for row in distance_rows:
        first = row["conformer_id_a"]
        second = row["conformer_id_b"]
        if first == second or first not in id_set or second not in id_set:
            continue
        pair = tuple(sorted((first, second)))
        if pair in distances:
            raise ValueError(f"{target_id}: duplicate structural-distance pair {pair}")
        value = numeric(row, "standardized_pocket_distance")
        if value < 0:
            raise ValueError(f"{target_id}: negative structural distance")
        distances[pair] = value
    expected_pairs = len(ids) * (len(ids) - 1) // 2
    if len(distances) != expected_pairs:
        raise ValueError(
            f"{target_id}: incomplete distances {len(distances)} != {expected_pairs}"
        )

    # Quality is recorded for transparency, but the initial concept screen
    # freezes its coefficient to zero to isolate structural diversity.
    quality_raw: dict[str, float] = {}
    for conformer_id in ids:
        audit = audit_by_id[conformer_id]
        completeness = numeric(audit, "pocket_heavy_atom_completeness_fraction", 1.0)
        residue_fraction = numeric(audit, "pocket_residue_fraction", 1.0)
        quality_raw[conformer_id] = 1.0 - 0.5 * (completeness + residue_fraction)
    quality = minmax(quality_raw)

    return {
        "target_id": target_id,
        "ids": ids,
        "pool_rows": pool_by_id,
        "audit_rows": audit_by_id,
        "feature_count": len(feature_columns),
        "distances": distances,
        "quality_raw": quality_raw,
        "quality": quality,
        "excluded_hard_gate": excluded,
        "input_paths": paths,
    }


def distance_matrix(ids: list[str], distances: dict[tuple[str, str], float]) -> np.ndarray:
    matrix = np.zeros((len(ids), len(ids)), dtype=float)
    index = {value: position for position, value in enumerate(ids)}
    maximum = max(distances.values(), default=0.0)
    scale = maximum if maximum > 0 else 1.0
    for (first, second), value in distances.items():
        i, j = index[first], index[second]
        matrix[i, j] = matrix[j, i] = value / scale
    return matrix


def pair_sum(subset: tuple[str, ...], ids: list[str], matrix: np.ndarray) -> float:
    index = {value: position for position, value in enumerate(ids)}
    positions = [index[value] for value in subset]
    return float(sum(matrix[i, j] for i, j in itertools.combinations(positions, 2)))


def minimum_pair_distance(subset: tuple[str, ...], ids: list[str], matrix: np.ndarray) -> float:
    if len(subset) < 2:
        return 0.0
    index = {value: position for position, value in enumerate(ids)}
    positions = [index[value] for value in subset]
    return float(min(matrix[i, j] for i, j in itertools.combinations(positions, 2)))


def q_energy(
    subset: Iterable[str],
    ids: list[str],
    matrix: np.ndarray,
    quality: dict[str, float],
    target_k: int,
    cardinality_penalty: float,
    lambda_distance: float,
    lambda_quality: float,
) -> float:
    selected = tuple(sorted(subset))
    return float(
        cardinality_penalty * (len(selected) - target_k) ** 2
        - lambda_distance * pair_sum(selected, ids, matrix)
        + lambda_quality * sum(quality[value] for value in selected)
    )


def build_qubo(
    ids: list[str],
    matrix: np.ndarray,
    quality: dict[str, float],
    target_k: int,
    cardinality_penalty: float,
    lambda_distance: float,
    lambda_quality: float,
) -> dict[str, Any]:
    linear: dict[str, float] = {}
    quadratic: dict[str, float] = {}
    for i, conformer_id in enumerate(ids):
        linear[conformer_id] = float(
            cardinality_penalty * (1 - 2 * target_k)
            + lambda_quality * quality[conformer_id]
        )
        for j in range(i + 1, len(ids)):
            quadratic[f"{conformer_id}::{ids[j]}"] = float(
                2 * cardinality_penalty - lambda_distance * matrix[i, j]
            )
    return {
        "variables": ids,
        "constant": float(cardinality_penalty * target_k * target_k),
        "linear": linear,
        "quadratic": quadratic,
        "target_size": target_k,
        "cardinality_penalty": cardinality_penalty,
        "lambda_distance": lambda_distance,
        "lambda_quality": lambda_quality,
        "convention": (
            "Q(x)=constant+sum_i linear[i]*x_i+"
            "sum_i<j quadratic[i::j]*x_i*x_j; minimize Q"
        ),
    }


def maxmin_seeded(
    ids: list[str], matrix: np.ndarray, target_k: int, reference_id: str
) -> tuple[str, ...]:
    if target_k < 1 or target_k > len(ids):
        raise ValueError("invalid target_k")
    index = {value: position for position, value in enumerate(ids)}
    seed = reference_id if reference_id in index else ids[0]
    selected = [seed]
    remaining = set(ids) - {seed}
    while len(selected) < target_k:
        candidates: list[tuple[float, str]] = []
        for candidate in remaining:
            cpos = index[candidate]
            minimum = min(matrix[cpos, index[value]] for value in selected)
            candidates.append((float(minimum), candidate))
        best_value = max(value for value, _ in candidates)
        chosen = min(candidate for value, candidate in candidates if math.isclose(value, best_value, abs_tol=1e-15))
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(sorted(selected))


def maxsum_greedy(ids: list[str], matrix: np.ndarray, target_k: int) -> tuple[str, ...]:
    if target_k < 1 or target_k > len(ids):
        raise ValueError("invalid target_k")
    average = {ids[i]: float(matrix[i].sum()) for i in range(len(ids))}
    first = min(ids, key=lambda value: (-average[value], value))
    selected = [first]
    remaining = set(ids) - {first}
    index = {value: position for position, value in enumerate(ids)}
    while len(selected) < target_k:
        candidates = []
        for candidate in remaining:
            score = sum(matrix[index[candidate], index[value]] for value in selected)
            candidates.append((float(score), candidate))
        best_value = max(value for value, _ in candidates)
        chosen = min(candidate for value, candidate in candidates if math.isclose(value, best_value, abs_tol=1e-15))
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(sorted(selected))


def lexicographic_start(ids: list[str], target_k: int) -> tuple[str, ...]:
    return tuple(ids[:target_k])


def improve_by_swaps(
    start: tuple[str, ...],
    ids: list[str],
    matrix: np.ndarray,
    quality: dict[str, float],
    target_k: int,
    cardinality_penalty: float,
    lambda_distance: float,
    lambda_quality: float,
) -> tuple[tuple[str, ...], float, int]:
    current = tuple(sorted(start))
    current_energy = q_energy(
        current, ids, matrix, quality, target_k, cardinality_penalty,
        lambda_distance, lambda_quality
    )
    iterations = 0
    while True:
        selected = set(current)
        best_subset = current
        best_energy = current_energy
        for outgoing in current:
            for incoming in ids:
                if incoming in selected:
                    continue
                candidate = tuple(sorted((selected - {outgoing}) | {incoming}))
                energy = q_energy(
                    candidate, ids, matrix, quality, target_k,
                    cardinality_penalty, lambda_distance, lambda_quality
                )
                if energy < best_energy - 1e-12 or (
                    math.isclose(energy, best_energy, abs_tol=1e-12)
                    and candidate < best_subset
                ):
                    best_subset, best_energy = candidate, energy
        if best_energy < current_energy - 1e-12:
            current, current_energy = best_subset, best_energy
            iterations += 1
            if iterations > len(ids) * target_k:
                raise RuntimeError("swap local search exceeded iteration guard")
            continue
        return current, current_energy, iterations


def qubo_restarts(
    ids: list[str],
    matrix: np.ndarray,
    quality: dict[str, float],
    target_k: int,
    reference_id: str,
    cardinality_penalty: float,
    lambda_distance: float,
    lambda_quality: float,
    seed: int,
    restart_count: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    starts: list[tuple[str, ...]] = [
        maxmin_seeded(ids, matrix, target_k, reference_id),
        maxsum_greedy(ids, matrix, target_k),
        lexicographic_start(ids, target_k),
    ]
    for _ in range(max(0, restart_count - len(starts))):
        starts.append(tuple(sorted(rng.sample(ids, target_k))))
    rows: list[dict[str, Any]] = []
    for restart, start in enumerate(starts):
        selected, energy, iterations = improve_by_swaps(
            start, ids, matrix, quality, target_k, cardinality_penalty,
            lambda_distance, lambda_quality
        )
        rows.append(
            {
                "restart": restart,
                "start_subset": "+".join(start),
                "selected_subset": "+".join(selected),
                "energy": energy,
                "iterations": iterations,
            }
        )
    return rows


def subset_metrics(
    subset: tuple[str, ...], ids: list[str], matrix: np.ndarray, reference_id: str
) -> dict[str, float]:
    index = {value: position for position, value in enumerate(ids)}
    reference_distances = [
        matrix[index[reference_id], index[value]]
        for value in subset
        if value != reference_id and reference_id in index
    ]
    return {
        "selected_count": len(subset),
        "pair_distance_sum_normalized": pair_sum(subset, ids, matrix),
        "mean_pair_distance_normalized": (
            pair_sum(subset, ids, matrix) / math.comb(len(subset), 2)
            if len(subset) >= 2 else 0.0
        ),
        "minimum_pair_distance_normalized": minimum_pair_distance(subset, ids, matrix),
        "mean_distance_to_reference_normalized": (
            statistics.fmean(reference_distances) if reference_distances else 0.0
        ),
    }


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported Stage21 config schema")
    if config.get("evidence_timing", {}).get("new_docking_jobs"):
        raise ValueError("Stage21 must not launch docking")
    targets_spec = config.get("targets")
    if not isinstance(targets_spec, dict) or not targets_spec:
        raise ValueError("Stage21 has no targets")

    output_paths = {
        key: rooted(root, value) for key, value in config["outputs"].items()
    }
    if not overwrite and any(path.exists() for path in output_paths.values()):
        raise FileExistsError("Stage21 outputs exist; pass --overwrite")

    diagnostic = config["diagnostic"]
    k_values = [int(value) for value in diagnostic["k_values"]]
    if k_values != sorted(set(k_values)) or min(k_values) < 2:
        raise ValueError("Stage21 k_values must be unique, sorted, and >= 2")
    lambda_distance = float(diagnostic["lambda_distance"])
    lambda_quality = float(diagnostic["lambda_quality"])
    cardinality_scale = float(diagnostic["cardinality_penalty_scale"])
    restart_count = int(diagnostic["restart_count"])
    base_seed = int(diagnostic["base_seed"])
    if lambda_distance <= 0 or cardinality_scale <= 0 or restart_count < 3:
        raise ValueError("invalid Stage21 diagnostic settings")

    all_selection_rows: list[dict[str, Any]] = []
    all_restart_rows: list[dict[str, Any]] = []
    target_records: dict[str, Any] = {}
    input_records: dict[str, Any] = {}

    for target_id, spec in targets_spec.items():
        target = load_target(root, target_id, spec)
        ids = target["ids"]
        matrix = distance_matrix(ids, target["distances"])
        reference_id = str(spec["reference_id"])
        if reference_id not in ids:
            raise ValueError(f"{target_id}: reference ID is not eligible")
        input_records[target_id] = {
            key: descriptor(root, path) for key, path in target["input_paths"].items()
        }
        target_record: dict[str, Any] = {
            "candidate_count": len(ids),
            "hard_gate_excluded_count": len(target["excluded_hard_gate"]),
            "hard_gate_excluded_ids": sorted(target["excluded_hard_gate"]),
            "feature_count": target["feature_count"],
            "distance_count": len(target["distances"]),
            "reference_id": reference_id,
            "k_models": {},
        }
        for k in k_values:
            if k > len(ids):
                raise ValueError(f"{target_id}: k={k} exceeds candidate count")
            max_pair = max(float(value) for value in matrix.flat)
            penalty = cardinality_scale * max(1.0, max_pair) * max(1, k)
            qubo = build_qubo(
                ids, matrix, target["quality"], k, penalty,
                lambda_distance, lambda_quality
            )
            restarts = qubo_restarts(
                ids, matrix, target["quality"], k, reference_id, penalty,
                lambda_distance, lambda_quality, base_seed + k, restart_count
            )
            all_restart_rows.extend(
                {"target_id": target_id, "k": k, **row} for row in restarts
            )
            energies = [float(row["energy"]) for row in restarts]
            best_energy = min(energies)
            best_subset = min(
                row["selected_subset"]
                for row in restarts
                if math.isclose(float(row["energy"]), best_energy, abs_tol=1e-12)
            )
            best_frequency = sum(
                row["selected_subset"] == best_subset for row in restarts
            )
            distinct_energies = sorted(
                {
                    round(float(row["energy"]), 12)
                    for row in restarts
                    if float(row["energy"]) > best_energy + 1e-12
                }
            )
            second_energy = distinct_energies[0] if distinct_energies else best_energy
            qubo_subset = tuple(best_subset.split("+"))
            maxmin_subset = maxmin_seeded(ids, matrix, k, reference_id)
            maxsum_subset = maxsum_greedy(ids, matrix, k)
            methods = {
                "qubo_swap_local_search": qubo_subset,
                "maxmin_seeded": maxmin_subset,
                "maxsum_greedy": maxsum_subset,
            }
            model_rows: dict[str, Any] = {}
            for method, subset in methods.items():
                metrics = subset_metrics(subset, ids, matrix, reference_id)
                energy = q_energy(
                    subset, ids, matrix, target["quality"], k, penalty,
                    lambda_distance, lambda_quality
                )
                row = {
                    "target_id": target_id,
                    "k": k,
                    "method": method,
                    "selected_subset": "+".join(subset),
                    "qubo_energy": energy,
                    **metrics,
                    "differs_from_maxmin": method == "qubo_swap_local_search" and subset != maxmin_subset,
                    "differs_from_maxsum": method == "qubo_swap_local_search" and subset != maxsum_subset,
                }
                if method == "qubo_swap_local_search":
                    row.update(
                        {
                            "best_restart_frequency": best_frequency,
                            "best_restart_fraction": best_frequency / len(restarts),
                            "restart_energy_gap": second_energy - best_energy,
                        }
                    )
                all_selection_rows.append(row)
                model_rows[method] = row
            unique_solutions = sorted({row["selected_subset"] for row in restarts})
            target_record["k_models"][str(k)] = {
                "qubo": qubo,
                "selected_subset": list(qubo_subset),
                "selected_energy": q_energy(
                    qubo_subset, ids, matrix, target["quality"], k, penalty,
                    lambda_distance, lambda_quality
                ),
                "restart_count": len(restarts),
                "unique_solution_count": len(unique_solutions),
                "restart_energy_min": min(energies),
                "restart_energy_max": max(energies),
                "restart_energy_mean": statistics.fmean(energies),
                "best_restart_frequency": best_frequency,
                "best_restart_fraction": best_frequency / len(restarts),
                "restart_energy_gap": second_energy - best_energy,
                "state_count": math.comb(len(ids), k),
                "enumerated_state_count": 0,
                "solver": "deterministic_exact-cardinality_one-swap_local_search",
                "methods": model_rows,
            }
        target_records[target_id] = target_record

    write_csv(output_paths["selection_csv"], all_selection_rows)
    write_csv(output_paths["restart_csv"], all_restart_rows)
    model_record = {
        "schema_version": "1.0",
        "status": "stage21_structure_aware_qubo_model_record",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "input_records": input_records,
        "target_models": target_records,
        "data_boundary": {
            "ligand_labels_read": 0,
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "quantum_hardware_jobs": 0,
            "new_docking_jobs": 0,
        },
    }
    write_json(output_paths["model_record_json"], model_record)

    summary_rows = []
    for row in all_selection_rows:
        if row["method"] == "qubo_swap_local_search":
            summary_rows.append(row)
    report_lines = [
        "# Stage 21: structure-aware conformation-pool QUBO",
        "",
        "This is a post-hoc, label-independent structural concept validation.",
        "No docking score, ligand label, fresh-validation row, test row, or quantum hardware result was read.",
        "",
        "## Frozen model",
        "",
        "`Q(x) = A*(sum(x)-k)^2 - lambda_distance*sum(d_ij*x_i*x_j) + lambda_quality*sum(q_i*x_i)`",
        "",
        f"The run used lambda_distance={lambda_distance}, lambda_quality={lambda_quality}, and {restart_count} deterministic restarts per target and k.",
        "The quality term was frozen to zero to isolate structural diversity; invalid structures were removed by a hard preparation gate.",
        "",
        "## Selection results",
        "",
        "| Target | k | QUBO subset | Mean pair distance | Minimum pair distance | Different from max-min | Different from max-sum | Best restart fraction | Unique restart solutions |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target_id in sorted(target_records):
        for k in k_values:
            qrow = next(row for row in summary_rows if row["target_id"] == target_id and row["k"] == k)
            record = target_records[target_id]["k_models"][str(k)]
            report_lines.append(
                f"| {target_id} | {k} | `{qrow['selected_subset']}` | {qrow['mean_pair_distance_normalized']:.6f} | {qrow['minimum_pair_distance_normalized']:.6f} | {str(qrow['selected_subset'] != next(row for row in all_selection_rows if row['target_id']==target_id and row['k']==k and row['method']=='maxmin_seeded')['selected_subset']).lower()} | {str(qrow['selected_subset'] != next(row for row in all_selection_rows if row['target_id']==target_id and row['k']==k and row['method']=='maxsum_greedy')['selected_subset']).lower()} | {record['best_restart_fraction']:.3f} | {record['unique_solution_count']} |"
            )
    report_lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "A different or more stable structural subset is evidence that the selection problem can be moved into a combinatorial objective. It is not evidence of better virtual-screening enrichment, quantum advantage, or biological superiority.",
        "The next gate is to redock only a preregistered, small matched subset if and only if the structural QUBO produces a reproducible difference from max-min and max-sum baselines.",
    ])
    output_paths["report_md"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["report_md"].write_text("\n".join(report_lines) + "\n", encoding="ascii")

    result = {
        "schema_version": "1.0",
        "status": "stage21_structure_aware_qubo_train_only_complete",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "implementation": {"path": Path(__file__).resolve().relative_to(root).as_posix(), "sha256": file_sha256(Path(__file__).resolve())},
        "targets": target_records,
        "inputs": input_records,
        "outputs": {
            key: descriptor(root, path)
            for key, path in output_paths.items()
            if key != "result_json"
        },
        "result_json": {"path": output_paths["result_json"].relative_to(root).as_posix()},
        "data_boundary": model_record["data_boundary"],
        "decision": {
            "reproducible_difference_observed": any(
                row["selected_subset"] != next(
                    other["selected_subset"] for other in all_selection_rows
                    if other["target_id"] == row["target_id"] and other["k"] == row["k"] and other["method"] == "maxmin_seeded"
                )
                for row in all_selection_rows if row["method"] == "qubo_swap_local_search"
            ),
            "stable_difference_observed": any(
                bool(row.get("differs_from_maxmin"))
                and float(row.get("best_restart_fraction", 0.0)) >= 0.5
                for row in all_selection_rows
                if row["method"] == "qubo_swap_local_search"
            ),
            "new_docking_authorized": False,
            "quantum_hardware_authorized": False,
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(output_paths["result_json"], result)
    write_json(output_paths["result_json"], result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
