"""Run the Stage30 group-balanced multiscale-state PPARG QUBO benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import (
    descriptor,
    file_sha256,
    read_json,
    rooted,
    write_csv,
    write_json,
)
from scripts.run_stage29_pparg_md_qubo_scaling import temporal_maximin_order


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_labels(raw: np.ndarray) -> np.ndarray:
    members = {int(label): np.flatnonzero(raw == label) for label in np.unique(raw)}
    ordered = sorted(members, key=lambda label: int(members[label][0]))
    mapping = {label: index for index, label in enumerate(ordered)}
    return np.asarray([mapping[int(label)] for label in raw], dtype=int)


def load_inputs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = {key: rooted(root, value) for key, value in config["inputs"].items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    stage28 = read_json(paths["stage28b_audit"])
    stage29 = read_json(paths["stage29_audit"])
    if stage28.get("status") != "stage28_pparg_multistart_md_ensemble_audit_ok":
        raise ValueError("Stage28b audit status differs")
    if stage29.get("status") != "stage29_pparg_md_qubo_solver_scaling_audit_ok":
        raise ValueError("Stage29 audit status differs")
    frames = read_rows(paths["frame_manifest"])
    if len(frames) != 1200:
        raise ValueError("Stage30 requires exactly 1200 frames")
    frame_ids = np.asarray([row["frame_id"] for row in frames])
    features = np.load(paths["feature_archive"])
    distances = np.load(paths["distance_archive"])
    if not np.array_equal(features["frame_ids"], frame_ids) or not np.array_equal(distances["frame_ids"], frame_ids):
        raise ValueError("Stage30 frame IDs differ across inputs")
    standardized = features["standardized_features"].astype(float)
    condensed = distances["condensed_distances"].astype(float)
    if standardized.shape != (1200, 870) or condensed.shape != (719400,):
        raise ValueError("Stage30 input dimensions differ")
    if not np.all(np.isfinite(standardized)) or not np.all(np.isfinite(condensed)):
        raise ValueError("Stage30 input arrays contain non-finite values")
    matrix = squareform(condensed)
    maximum = float(matrix.max())
    if maximum <= 0:
        raise ValueError("Stage30 distance matrix is degenerate")
    normalized = matrix / maximum
    by_start: dict[int, list[int]] = {}
    for index, row in enumerate(frames):
        by_start.setdefault(int(row["start_index"]), []).append(index)
    if sorted(len(values) for values in by_start.values()) != [150] * 8:
        raise ValueError("Stage30 requires eight starts with 150 frames each")
    centrality = np.zeros(len(frames), dtype=float)
    ordered_global: dict[int, list[int]] = {}
    for start, indices in sorted(by_start.items()):
        local_by_index = {int(frames[index]["local_frame_index"]): index for index in indices}
        order = temporal_maximin_order(local_by_index)
        ordered_global[start] = [local_by_index[value] for value in order]
        local = np.asarray(indices, dtype=int)
        centrality[local] = 1.0 - normalized[np.ix_(local, local)].sum(axis=1) / (len(local) - 1)
    hierarchy = linkage(condensed, method="ward", optimal_ordering=False)
    labels: dict[int, np.ndarray] = {}
    for count_value in config["structural_states"]["cluster_counts"]:
        count = int(count_value)
        value = canonical_labels(fcluster(hierarchy, count, criterion="maxclust"))
        if len(np.unique(value)) != count:
            raise ValueError(f"Ward clustering did not yield {count} states")
        labels[count] = value
    state_separation = np.zeros_like(normalized)
    for value in labels.values():
        state_separation += value[:, None] != value[None, :]
    state_separation /= len(labels)
    return {
        "paths": paths,
        "frames": frames,
        "distance": normalized,
        "centrality": centrality,
        "ordered_global": ordered_global,
        "labels": labels,
        "state_separation": state_separation,
        "distance_maximum": maximum,
    }


def build_model(per_start: int, loaded: dict[str, Any], objective: dict[str, Any]) -> dict[str, Any]:
    groups_global = [tuple(loaded["ordered_global"][start][:per_start]) for start in sorted(loaded["ordered_global"])]
    global_indices = np.asarray([value for group in groups_global for value in group], dtype=int)
    groups = [tuple(range(index * per_start, (index + 1) * per_start)) for index in range(len(groups_global))]
    k = int(objective["selected_count"])
    if len(groups) != int(objective["group_count"]) or k != len(groups):
        raise ValueError("Stage30 requires one selected frame per start")
    linear = float(objective["within_start_centrality_weight"]) * loaded["centrality"][global_indices] / k
    denominator = math.comb(k, 2)
    pair = (
        float(objective["cross_start_pair_diversity_weight"]) * loaded["distance"][np.ix_(global_indices, global_indices)]
        + float(objective["multiscale_state_separation_weight"]) * loaded["state_separation"][np.ix_(global_indices, global_indices)]
    ) / denominator
    for group in groups:
        pair[np.ix_(group, group)] = 0.0
    np.fill_diagonal(pair, 0.0)
    return {
        "per_start": per_start,
        "global_indices": global_indices,
        "groups": groups,
        "linear": linear,
        "pair": pair,
        "k": k,
        "frame_ids": tuple(loaded["frames"][index]["frame_id"] for index in global_indices),
        "source_ids": tuple(loaded["frames"][index]["conformer_id"] for index in global_indices),
        "distance": loaded["distance"][np.ix_(global_indices, global_indices)],
        "state_separation": loaded["state_separation"][np.ix_(global_indices, global_indices)],
    }


def validate_selected(selected: Iterable[int], model: dict[str, Any]) -> tuple[int, ...]:
    chosen = tuple(sorted(int(value) for value in selected))
    if len(chosen) != int(model["k"]) or len(chosen) != len(set(chosen)):
        raise ValueError("Stage30 subset has wrong cardinality")
    if any(sum(value in set(chosen) for value in group) != 1 for group in model["groups"]):
        raise ValueError("Stage30 subset violates exactly-one-per-start")
    return chosen


def subset_objective(selected: Iterable[int], model: dict[str, Any]) -> float:
    chosen = np.asarray(validate_selected(selected, model), dtype=int)
    return float(model["linear"][chosen].sum() + np.triu(model["pair"][np.ix_(chosen, chosen)], 1).sum())


def subset_components(selected: Iterable[int], model: dict[str, Any], objective: dict[str, Any]) -> dict[str, Any]:
    chosen = np.asarray(validate_selected(selected, model), dtype=int)
    upper = np.triu_indices(len(chosen), 1)
    centrality = float(model["linear"][chosen].sum())
    diversity = float(objective["cross_start_pair_diversity_weight"]) * float(model["distance"][np.ix_(chosen, chosen)][upper].mean())
    separation = float(objective["multiscale_state_separation_weight"]) * float(model["state_separation"][np.ix_(chosen, chosen)][upper].mean())
    state_counts = {
        str(count): len(set(int(value) for value in labels))
        for count, labels in (
            (count, loaded_labels[chosen]) for count, loaded_labels in model["local_labels"].items()
        )
    }
    return {
        "objective": centrality + diversity + separation,
        "weighted_within_start_centrality": centrality,
        "weighted_cross_start_pair_diversity": diversity,
        "weighted_multiscale_state_separation": separation,
        "represented_start_count": len(set(model["source_ids"][index] for index in chosen)),
        "represented_state_counts": json.dumps(state_counts, sort_keys=True, separators=(",", ":")),
    }


def better(candidate: tuple[int, ...], value: float, current: tuple[int, ...], current_value: float) -> bool:
    return value > current_value + 1e-12 or (math.isclose(value, current_value, abs_tol=1e-12) and candidate < current)


def greedy_for_order(model: dict[str, Any], order: Iterable[int]) -> tuple[int, ...]:
    selected: list[int] = []
    for group_index in order:
        group = model["groups"][group_index]
        scores = model["linear"][list(group)].copy()
        if selected:
            scores += model["pair"][np.ix_(group, selected)].sum(axis=1)
        selected.append(group[int(np.argmax(scores))])
    return tuple(sorted(selected))


def cyclic_orders(group_count: int) -> list[tuple[int, ...]]:
    forward = tuple(range(group_count))
    reverse = tuple(reversed(forward))
    orders = []
    for base in (forward, reverse):
        for offset in range(group_count):
            orders.append(base[offset:] + base[:offset])
    return orders


def coordinate_descent(start: Iterable[int], model: dict[str, Any]) -> tuple[tuple[int, ...], float, int]:
    current = validate_selected(start, model)
    current_value = subset_objective(current, model)
    iterations = 0
    while True:
        selected_set = set(current)
        best, best_value = current, current_value
        for group in model["groups"]:
            outgoing = next(value for value in group if value in selected_set)
            retained = tuple(value for value in current if value != outgoing)
            scores = model["linear"][list(group)] + model["pair"][np.ix_(group, retained)].sum(axis=1)
            incoming = group[int(np.argmax(scores))]
            candidate = tuple(sorted((*retained, incoming)))
            value = float(model["linear"][list(retained)].sum() + np.triu(model["pair"][np.ix_(retained, retained)], 1).sum() + scores[int(np.argmax(scores))])
            if better(candidate, value, best, best_value):
                best, best_value = candidate, value
        if best_value <= current_value + 1e-12:
            return current, current_value, iterations
        current, current_value = best, best_value
        iterations += 1
        if iterations > len(model["linear"]) * len(model["groups"]):
            raise RuntimeError("Stage30 coordinate descent exceeded iteration guard")


def beam_for_order(model: dict[str, Any], order: Iterable[int], width: int) -> tuple[int, ...]:
    layer: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for group_index in order:
        group = model["groups"][group_index]
        proposals = []
        for selected, value in layer:
            increments = model["linear"][list(group)].copy()
            if selected:
                increments += model["pair"][np.ix_(group, selected)].sum(axis=1)
            proposals.extend((tuple(sorted((*selected, incoming))), value + float(increments[position])) for position, incoming in enumerate(group))
        layer = sorted(proposals, key=lambda item: (-item[1], item[0]))[:width]
    return layer[0][0]


def exact_oracle(model: dict[str, Any], maximum_states: int) -> dict[str, Any] | None:
    state_count = math.prod(len(group) for group in model["groups"])
    if state_count > maximum_states:
        return None
    best: tuple[int, ...] | None = None
    best_value = -math.inf
    for selected in itertools.product(*model["groups"]):
        candidate = tuple(sorted(selected))
        value = subset_objective(candidate, model)
        if best is None or better(candidate, value, best, best_value):
            best, best_value = candidate, value
    return {"selected": best, "objective": best_value, "state_count": state_count}


def anneal(model: dict[str, Any], sampler: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    selected = [rng.choice(group) for group in model["groups"]]
    current_value = subset_objective(selected, model)
    best, best_value = tuple(sorted(selected)), current_value
    sweeps = max(int(sampler["minimum_sweeps_per_read"]), int(sampler["sweeps_per_candidate"]) * len(model["linear"]))
    start_temp = float(sampler["temperature_start"])
    end_temp = float(sampler["temperature_end"])
    accepted = 0
    for step in range(sweeps):
        group_index = rng.randrange(len(model["groups"]))
        group = model["groups"][group_index]
        outgoing = selected[group_index]
        incoming = rng.choice(group)
        while incoming == outgoing:
            incoming = rng.choice(group)
        retained = [value for index, value in enumerate(selected) if index != group_index]
        delta = (
            float(model["linear"][incoming] - model["linear"][outgoing])
            + float(model["pair"][incoming, retained].sum())
            - float(model["pair"][outgoing, retained].sum())
        )
        progress = step / max(1, sweeps - 1)
        temperature = max(end_temp, start_temp * (end_temp / start_temp) ** progress)
        if delta >= 0 or rng.random() < math.exp(max(-700.0, delta / temperature)):
            selected[group_index] = incoming
            current_value += delta
            accepted += 1
            candidate = tuple(sorted(selected))
            if better(candidate, current_value, best, best_value):
                best, best_value = candidate, current_value
    return {"selected": best, "objective": best_value, "acceptance_fraction": accepted / sweeps, "sweeps": sweeps}


def qubo_record(model: dict[str, Any], selected: tuple[int, ...], objective: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    n = len(model["linear"])
    penalty = float(objective["exactly_one_group_penalty"])
    q_linear = -model["linear"] - penalty
    q_pair = -model["pair"].copy()
    for group in model["groups"]:
        for first, second in itertools.combinations(group, 2):
            q_pair[first, second] = q_pair[second, first] = 2 * penalty
    upper = q_pair[np.triu_indices(n, 1)]
    nonzero = np.abs(np.concatenate((q_linear, upper)))
    nonzero = nonzero[nonzero > 1e-15]
    dynamic_range = float(nonzero.max() / nonzero.min())
    chosen = np.asarray(selected, dtype=int)
    energy = float(penalty * len(model["groups"]) + q_linear[chosen].sum() + np.triu(q_pair[np.ix_(chosen, chosen)], 1).sum())
    residual = abs(energy + subset_objective(selected, model))
    digest = hashlib.sha256()
    digest.update(np.asarray(model["global_indices"], dtype=np.int32).tobytes())
    digest.update(np.asarray(q_linear, dtype=np.float64).tobytes())
    digest.update(np.asarray(upper, dtype=np.float64).tobytes())
    couplers = int(np.count_nonzero(np.abs(upper) > 1e-15))
    direct_ready = (
        n <= int(gate["direct_qpu_max_logical_variables"])
        and couplers <= int(gate["direct_qpu_max_quadratic_couplers"])
        and dynamic_range <= float(gate["direct_qpu_max_coefficient_dynamic_range"])
    )
    return {
        "logical_variable_count": n,
        "quadratic_coupler_count": couplers,
        "coupler_density": couplers / max(1, n * (n - 1) / 2),
        "coefficient_minimum_absolute_nonzero": float(nonzero.min()),
        "coefficient_maximum_absolute": float(nonzero.max()),
        "coefficient_dynamic_range": dynamic_range,
        "qubo_sha256": digest.hexdigest().upper(),
        "selected_qubo_energy": energy,
        "equivalence_residual": residual,
        "direct_qpu_ready_under_frozen_thresholds": direct_ready,
    }


def solution_row(pool_id: str, method: str, selected: tuple[int, ...], model: dict[str, Any], objective: dict[str, Any], runtime: float, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "pool_id": pool_id,
        "frames_per_start": model["per_start"],
        "candidate_count": len(model["linear"]),
        "method": method,
        "runtime_seconds": runtime,
        "selected_frame_ids": "+".join(model["frame_ids"][index] for index in selected),
        "selected_source_ids": "+".join(model["source_ids"][index] for index in selected),
        **subset_components(selected, model, objective),
        **extra,
    }


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if outputs["result_json"].exists() and not overwrite:
        raise FileExistsError(f"result exists: {outputs['result_json']}; pass --overwrite")
    loaded = load_inputs(root, config)
    objective, sampler, gate = config["objective"], config["annealing_sampler"], config["gate"]
    state_rows = []
    for index, frame in enumerate(loaded["frames"]):
        state_rows.append({
            "frame_id": frame["frame_id"],
            "global_frame_index": index,
            "start_index": frame["start_index"],
            "conformer_id": frame["conformer_id"],
            "local_frame_index": frame["local_frame_index"],
            "time_ps": frame["time_ps"],
            "within_start_centrality": loaded["centrality"][index],
            **{f"ward_state_{count}": int(labels[index]) for count, labels in loaded["labels"].items()},
        })
    candidate_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    read_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    pool_records: dict[str, Any] = {}
    strict_wins = 0
    stable_cells = 0
    exact_total = 0
    exact_passed_count = 0
    for pool_number, per_start_value in enumerate(config["candidate_scaling"]["frames_per_start"]):
        per_start = int(per_start_value)
        pool_id = f"balanced_m{per_start:03d}_n{per_start * 8:04d}"
        model = build_model(per_start, loaded, objective)
        model["local_labels"] = {count: labels[model["global_indices"]] for count, labels in loaded["labels"].items()}
        n = len(model["linear"])
        print(json.dumps({"pool_id": pool_id, "candidate_count": n, "status": "running"}), flush=True)
        for local_index, global_index in enumerate(model["global_indices"]):
            frame = loaded["frames"][int(global_index)]
            candidate_rows.append({
                "pool_id": pool_id,
                "frames_per_start": per_start,
                "candidate_count": n,
                "local_candidate_index": local_index,
                "global_frame_index": int(global_index),
                "frame_id": frame["frame_id"],
                "start_index": frame["start_index"],
                "conformer_id": frame["conformer_id"],
                "local_frame_index": frame["local_frame_index"],
                "time_ps": frame["time_ps"],
            })
        orders = cyclic_orders(len(model["groups"]))
        started = time.perf_counter()
        cyclic_endpoints = []
        total_iterations = 0
        for order in orders:
            selected, value, iterations = coordinate_descent(greedy_for_order(model, order), model)
            cyclic_endpoints.append((selected, value))
            total_iterations += iterations
        cyclic_selected, _ = max(cyclic_endpoints, key=lambda item: (item[1], tuple(-value for value in item[0])))
        cyclic_row = solution_row(pool_id, "cyclic_greedy_plus_coordinate_descent", cyclic_selected, model, objective, time.perf_counter() - started, {"restart_count": len(orders), "beam_width": 0, "local_search_iterations": total_iterations, "unique_local_optima": len({row[0] for row in cyclic_endpoints}), "state_count": ""})
        started = time.perf_counter()
        beam_endpoints = []
        total_iterations = 0
        for order in orders:
            selected, value, iterations = coordinate_descent(beam_for_order(model, order, int(config["classical_baselines"]["beam_width"])), model)
            beam_endpoints.append((selected, value))
            total_iterations += iterations
        beam_selected, _ = max(beam_endpoints, key=lambda item: (item[1], tuple(-value for value in item[0])))
        beam_row = solution_row(pool_id, "beam64_plus_coordinate_descent", beam_selected, model, objective, time.perf_counter() - started, {"restart_count": len(orders), "beam_width": int(config["classical_baselines"]["beam_width"]), "local_search_iterations": total_iterations, "unique_local_optima": len({row[0] for row in beam_endpoints}), "state_count": ""})
        started = time.perf_counter()
        rng = random.Random(int(sampler["base_seed"]) + pool_number * 1000003 + 31)
        random_endpoints = []
        total_iterations = 0
        for _ in range(int(config["classical_baselines"]["random_coordinate_restart_count"])):
            start = tuple(rng.choice(group) for group in model["groups"])
            selected, value, iterations = coordinate_descent(start, model)
            random_endpoints.append((selected, value))
            total_iterations += iterations
        random_selected, _ = max(random_endpoints, key=lambda item: (item[1], tuple(-value for value in item[0])))
        random_row = solution_row(pool_id, "random64_coordinate_descent", random_selected, model, objective, time.perf_counter() - started, {"restart_count": int(config["classical_baselines"]["random_coordinate_restart_count"]), "beam_width": 0, "local_search_iterations": total_iterations, "unique_local_optima": len({row[0] for row in random_endpoints}), "state_count": ""})
        classical = [cyclic_row, beam_row, random_row]
        solver_rows.extend(classical)
        exact_started = time.perf_counter()
        exact = exact_oracle(model, int(config["exact_oracle"]["maximum_states"]))
        exact_runtime = time.perf_counter() - exact_started
        if exact is not None:
            exact_total += 1
            solver_rows.append(solution_row(pool_id, "exact_oracle", exact["selected"], model, objective, exact_runtime, {"restart_count": 1, "beam_width": 0, "local_search_iterations": 0, "unique_local_optima": 1, "state_count": exact["state_count"]}))
        strong = max(classical, key=lambda row: (float(row["objective"]), row["selected_frame_ids"]))
        local_batches = []
        for batch in range(int(sampler["batch_count"])):
            seed = int(sampler["base_seed"]) + pool_number * 1000003 + batch * 10007
            rng = random.Random(seed)
            local_reads = []
            started = time.perf_counter()
            for read_index in range(int(sampler["reads_per_batch"])):
                sampled = anneal(model, sampler, rng)
                components = subset_components(sampled["selected"], model, objective)
                row = {
                    "pool_id": pool_id,
                    "frames_per_start": per_start,
                    "candidate_count": n,
                    "batch": batch,
                    "read": read_index,
                    "seed": seed,
                    "sweeps": sampled["sweeps"],
                    "selected_frame_ids": "+".join(model["frame_ids"][index] for index in sampled["selected"]),
                    "selected_source_ids": "+".join(model["source_ids"][index] for index in sampled["selected"]),
                    **components,
                    "acceptance_fraction": sampled["acceptance_fraction"],
                    "delta_vs_strong_classical": float(components["objective"]) - float(strong["objective"]),
                }
                read_rows.append(row)
                local_reads.append((row, sampled["selected"]))
            runtime = time.perf_counter() - started
            best_row, best_selected = max(local_reads, key=lambda item: (float(item[0]["objective"]), item[0]["selected_frame_ids"]))
            batch_row = {
                "pool_id": pool_id,
                "frames_per_start": per_start,
                "candidate_count": n,
                "batch": batch,
                "seed": seed,
                "runtime_seconds": runtime,
                "best_objective": float(best_row["objective"]),
                "best_frame_ids": best_row["selected_frame_ids"],
                "delta_vs_strong_classical": float(best_row["objective"]) - float(strong["objective"]),
            }
            batch_rows.append(batch_row)
            local_batches.append((batch_row, best_selected))
        best_batch, annealed_selected = max(local_batches, key=lambda item: (float(item[0]["best_objective"]), item[0]["best_frame_ids"]))
        annealed_row = solution_row(pool_id, "group_feasible_qubo_annealing", annealed_selected, model, objective, sum(float(item[0]["runtime_seconds"]) for item in local_batches), {"restart_count": int(sampler["batch_count"]) * int(sampler["reads_per_batch"]), "beam_width": 0, "local_search_iterations": 0, "unique_local_optima": len({item[0]["best_frame_ids"] for item in local_batches}), "state_count": ""})
        solver_rows.append(annealed_row)
        annealed_value, strong_value = float(annealed_row["objective"]), float(strong["objective"])
        within = sum(float(item[0]["best_objective"]) >= annealed_value - float(gate["objective_tolerance"]) for item in local_batches) / len(local_batches)
        stable = within >= float(gate["minimum_batch_fraction_within_tolerance"])
        exact_value = None if exact is None else float(exact["objective"])
        exact_gap = None if exact_value is None else exact_value - annealed_value
        exact_ok = exact_gap is None or exact_gap <= float(gate["maximum_exact_gap"]) + 1e-12
        if exact is not None:
            exact_passed_count += int(exact_ok)
        strict_win = annealed_value - strong_value > float(gate["minimum_strict_gain"])
        strict_wins += int(strict_win)
        stable_cells += int(stable)
        qrecord = qubo_record(model, annealed_selected, objective, gate)
        model_rows.append({"pool_id": pool_id, "frames_per_start": per_start, "candidate_count": n, **qrecord})
        pool_records[pool_id] = {
            "frames_per_start": per_start,
            "candidate_count": n,
            "combinatorial_state_count": str(per_start ** 8),
            "strong_classical_method": strong["method"],
            "strong_classical_objective": strong_value,
            "annealing_objective": annealed_value,
            "delta_vs_strong_classical": annealed_value - strong_value,
            "annealing_strict_win": strict_win,
            "annealing_batch_fraction_within_tolerance": within,
            "annealing_stable": stable,
            "exact_oracle_objective": exact_value,
            "annealing_gap_to_exact": exact_gap,
            "exact_gate_passed": exact_ok,
            "annealing_selected_frame_ids": annealed_row["selected_frame_ids"].split("+"),
            "annealing_selected_source_ids": annealed_row["selected_source_ids"].split("+"),
            "qubo": qrecord,
        }
    write_csv(outputs["state_manifest_csv"], state_rows)
    write_csv(outputs["candidate_manifest_csv"], candidate_rows)
    write_csv(outputs["solver_results_csv"], solver_rows)
    write_csv(outputs["batch_results_csv"], batch_rows)
    write_csv(outputs["read_results_csv"], read_rows)
    write_csv(outputs["model_scaling_csv"], model_rows)
    cell_count = len(pool_records)
    equivalence = all(float(row["equivalence_residual"]) <= float(gate["maximum_qubo_equivalence_residual"]) for row in model_rows)
    decision = {
        "input_gate_passed": True,
        "scaling_complete": cell_count == int(gate["required_scaling_cell_count"]),
        "exactly_one_per_start_verified": all(int(row["represented_start_count"]) == 8 for row in solver_rows),
        "qubo_equivalence_gate_passed": equivalence,
        "exactness_gate_passed": exact_total > 0 and exact_passed_count == exact_total,
        "annealing_stability_gate_passed": stable_cells == cell_count,
        "cells_strictly_above_strong_classical": strict_wins,
        "solver_novelty_gate_passed": strict_wins >= int(gate["minimum_primary_cells_strictly_above_strong_classical"]),
        "direct_qpu_readiness_gate_passed": all(bool(row["direct_qpu_ready_under_frozen_thresholds"]) for row in model_rows),
        "new_docking_jobs_authorized_by_this_stage": False,
        "quantum_hardware_authorized": False,
    }
    report = [
        "# Stage 30: PPARG group-balanced multiscale-state QUBO",
        "",
        "Frozen objective: 0.3 within-start centrality + 0.5 cross-start distance + 0.2 multiscale state separation; exactly one frame per MD start.",
        "",
        "| Pool | n | Search states | Strong classical | Annealing | Delta | Stable | Exact gap | Variables/couplers |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for record in pool_records.values():
        gap = "NA" if record["annealing_gap_to_exact"] is None else f"{record['annealing_gap_to_exact']:.8g}"
        report.append(f"| m={record['frames_per_start']} | {record['candidate_count']} | {record['combinatorial_state_count']} | {record['strong_classical_objective']:.8f} | {record['annealing_objective']:.8f} | {record['delta_vs_strong_classical']:.8g} | {record['annealing_stable']} | {gap} | {record['qubo']['logical_variable_count']}/{record['qubo']['quadratic_coupler_count']} |")
    report += [
        "",
        f"Construction/equivalence/exactness gates: **{'PASS' if decision['scaling_complete'] and equivalence and decision['exactness_gate_passed'] else 'NO-GO'}**.",
        f"Annealing stability gate: **{'PASS' if decision['annealing_stability_gate_passed'] else 'NO-GO'}**.",
        f"Solver novelty gate: **{'PASS' if decision['solver_novelty_gate_passed'] else 'NO-GO'}** ({strict_wins} strict wins).",
        f"Direct-QPU readiness gate: **{'PASS' if decision['direct_qpu_readiness_gate_passed'] else 'NO-GO'}**.",
        "",
        "No docking scores, ligand labels, validation/test rows, new docking jobs, or quantum-hardware outputs were used.",
    ]
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report) + "\n", encoding="ascii")
    result = {
        "schema_version": "1.0",
        "status": "stage30_pparg_group_balanced_state_qubo_complete",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "implementation": descriptor(root, Path(__file__).resolve()),
        "inputs": {key: descriptor(root, path) for key, path in loaded["paths"].items()},
        "input_statistics": {"frame_count": len(loaded["frames"]), "distance_maximum_before_normalization": loaded["distance_maximum"], "ward_cluster_counts": sorted(loaded["labels"])},
        "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key not in {"result_json", "audit_json"}},
        "pool_records": pool_records,
        "decision": decision,
        "data_boundary": {"docking_scores_read": 0, "ligand_labels_read": 0, "fresh_validation_rows_read": 0, "test_rows_read": 0, "new_docking_jobs": 0, "quantum_hardware_jobs": 0},
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage30_pparg_group_balanced_state_qubo.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
