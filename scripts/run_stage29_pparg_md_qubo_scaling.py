"""Run the Stage29 PPARG MD-derived pure-QUBO solver scaling benchmark."""

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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def temporal_maximin_order(local_indices: Iterable[int]) -> list[int]:
    remaining = sorted(set(int(value) for value in local_indices))
    if not remaining:
        return []
    midpoint = (remaining[0] + remaining[-1]) / 2.0
    first = min(remaining, key=lambda value: (abs(value - midpoint), value))
    selected = [first]
    remaining.remove(first)
    while remaining:
        candidate = max(
            remaining,
            key=lambda value: (min(abs(value - chosen) for chosen in selected), -value),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def balanced_temporal_order(rows: list[dict[str, str]]) -> list[int]:
    by_start: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        by_start.setdefault(int(row["start_index"]), []).append(row)
    per_start: dict[int, list[int]] = {}
    for start, local_rows in sorted(by_start.items()):
        by_local = {int(row["local_frame_index"]): int(row["global_frame_index"]) for row in local_rows}
        local_order = temporal_maximin_order(by_local)
        per_start[start] = [by_local[value] for value in local_order]
    lengths = {len(values) for values in per_start.values()}
    if len(lengths) != 1:
        raise ValueError("Stage29 requires equal frame counts across starts")
    output = []
    for rank in range(next(iter(lengths))):
        output.extend(per_start[start][rank] for start in sorted(per_start))
    if len(output) != len(rows) or len(output) != len(set(output)):
        raise ValueError("balanced temporal order is incomplete or duplicated")
    return output


def build_pools(rows: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    order = balanced_temporal_order(rows)
    pools = []
    for size in config["candidate_pools"]["primary_scaling_sizes"]:
        count = int(size)
        pools.append({
            "pool_id": f"primary_n{count:04d}",
            "pool_class": "primary_scaling",
            "posthoc": False,
            "indices": tuple(order[:count]),
            "rule": f"first {count} frames in the frozen balanced temporal order",
        })
    for spec in config["candidate_pools"]["sensitivity_pools"]:
        pool_id = str(spec["pool_id"])
        if pool_id == "uniform_100ps_n240":
            indices = tuple(int(row["global_frame_index"]) for row in rows if (int(row["local_frame_index"]) + 1) % 5 == 0)
        elif pool_id == "uniform_200ps_n120":
            indices = tuple(int(row["global_frame_index"]) for row in rows if (int(row["local_frame_index"]) + 1) % 10 == 0)
        elif pool_id == "exclude_3d6d_n1050":
            indices = tuple(int(row["global_frame_index"]) for row in rows if row["conformer_id"] != "PPARG_3D6D_aligned")
        else:
            raise ValueError(f"unknown Stage29 sensitivity pool: {pool_id}")
        if len(indices) != int(spec["expected_count"]):
            raise ValueError(f"{pool_id}: unexpected candidate count")
        pools.append({
            "pool_id": pool_id,
            "pool_class": "sensitivity",
            "posthoc": bool(spec["posthoc"]),
            "indices": indices,
            "rule": spec["rule"],
        })
    return pools


def load_inputs(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = {key: rooted(root, value) for key, value in config["inputs"].items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = read_json(paths["stage28b_audit"])
    if audit.get("status") != "stage28_pparg_multistart_md_ensemble_audit_ok":
        raise ValueError("Stage28b audit status is not eligible for Stage29")
    if not audit.get("decision", {}).get("stage29_solver_scaling_authorized"):
        raise ValueError("Stage28b did not authorize Stage29 solver scaling")
    summary = read_json(paths["ensemble_summary"])
    if summary.get("status") != "stage28_pparg_multistart_md_ensemble_complete":
        raise ValueError("Stage28b ensemble is incomplete")
    rows = read_rows(paths["frame_manifest"])
    if len(rows) != 1200:
        raise ValueError("Stage29 requires exactly 1200 Stage28b frames")
    feature_archive = np.load(paths["feature_archive"])
    distance_archive = np.load(paths["distance_archive"])
    frame_ids = np.asarray([row["frame_id"] for row in rows])
    if not np.array_equal(feature_archive["frame_ids"], frame_ids):
        raise ValueError("feature archive frame IDs differ from manifest")
    if not np.array_equal(distance_archive["frame_ids"], frame_ids):
        raise ValueError("distance archive frame IDs differ from manifest")
    if str(distance_archive["metric"]) != "rms_standardized_euclidean":
        raise ValueError("unexpected Stage28b distance metric")
    features = feature_archive["standardized_features"]
    condensed = distance_archive["condensed_distances"]
    if features.shape != (1200, 870) or condensed.shape != (719400,):
        raise ValueError("Stage28b feature or distance dimensions differ")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(condensed)):
        raise ValueError("Stage28b inputs contain non-finite values")
    distances = squareform(condensed.astype(float))
    maximum = float(distances.max())
    if maximum <= 0:
        raise ValueError("Stage28b distance matrix is degenerate")
    normalized = distances / maximum
    centrality = 1.0 - normalized.sum(axis=1) / (len(normalized) - 1)
    return {
        "paths": paths,
        "rows": rows,
        "distance": normalized,
        "centrality": centrality,
        "distance_maximum": maximum,
        "raw_feature_count": int(features.shape[1]),
    }


def build_model(
    pool: dict[str, Any], rows: list[dict[str, str]], full_distance: np.ndarray,
    full_centrality: np.ndarray, objective: dict[str, Any]
) -> dict[str, Any]:
    indices = np.asarray(pool["indices"], dtype=int)
    k = int(objective["selected_count"])
    if len(indices) < k:
        raise ValueError("candidate pool is smaller than selected_count")
    distance = full_distance[np.ix_(indices, indices)]
    centrality = full_centrality[indices]
    start = np.asarray([int(rows[index]["start_index"]) for index in indices])
    time_ps = np.asarray([float(rows[index]["time_ps"]) for index in indices])
    same_start = start[:, None] == start[None, :]
    delta_time = np.abs(time_ps[:, None] - time_ps[None, :])
    redundancy = np.where(
        same_start,
        np.exp(-delta_time / float(objective["temporal_redundancy_decay_ps"])),
        0.0,
    )
    np.fill_diagonal(redundancy, 0.0)
    linear = float(objective["centrality_weight"]) * centrality / k
    pair_denominator = max(1, math.comb(k, 2))
    pair = (
        float(objective["pair_diversity_weight"]) * distance
        - float(objective["temporal_redundancy_weight"]) * redundancy
    ) / pair_denominator
    np.fill_diagonal(pair, 0.0)
    return {
        "indices": indices,
        "frame_ids": tuple(rows[index]["frame_id"] for index in indices),
        "conformer_ids": tuple(rows[index]["conformer_id"] for index in indices),
        "linear": linear,
        "pair": pair,
        "distance": distance,
        "redundancy": redundancy,
        "k": k,
    }


def subset_objective(selected: Iterable[int], model: dict[str, Any]) -> float:
    chosen = np.asarray(tuple(sorted(int(value) for value in selected)), dtype=int)
    if len(chosen) != int(model["k"]) or len(chosen) != len(set(chosen.tolist())):
        raise ValueError("selected subset has wrong fixed cardinality")
    pair = model["pair"][np.ix_(chosen, chosen)]
    return float(model["linear"][chosen].sum() + np.triu(pair, 1).sum())


def subset_components(selected: Iterable[int], model: dict[str, Any], objective: dict[str, Any]) -> dict[str, float]:
    chosen = np.asarray(tuple(sorted(int(value) for value in selected)), dtype=int)
    upper = np.triu_indices(len(chosen), 1)
    centrality = float(model["linear"][chosen].sum())
    diversity = float(objective["pair_diversity_weight"]) * float(model["distance"][np.ix_(chosen, chosen)][upper].mean())
    temporal_penalty = float(objective["temporal_redundancy_weight"]) * float(model["redundancy"][np.ix_(chosen, chosen)][upper].mean())
    return {
        "objective": centrality + diversity - temporal_penalty,
        "weighted_centrality": centrality,
        "weighted_pair_diversity": diversity,
        "weighted_temporal_penalty": temporal_penalty,
        "represented_start_count": len({model["conformer_ids"][index] for index in chosen}),
    }


def better(candidate: tuple[int, ...], value: float, current: tuple[int, ...], current_value: float) -> bool:
    return value > current_value + 1e-12 or (math.isclose(value, current_value, abs_tol=1e-12) and candidate < current)


def greedy(model: dict[str, Any]) -> tuple[int, ...]:
    n, k = len(model["linear"]), int(model["k"])
    selected: list[int] = []
    available = np.ones(n, dtype=bool)
    for _ in range(k):
        scores = model["linear"].copy()
        if selected:
            scores += model["pair"][:, selected].sum(axis=1)
        scores[~available] = -np.inf
        incoming = int(np.argmax(scores))
        selected.append(incoming)
        available[incoming] = False
    return tuple(sorted(selected))


def best_one_swap(start: Iterable[int], model: dict[str, Any]) -> tuple[tuple[int, ...], float, int]:
    n = len(model["linear"])
    current = tuple(sorted(int(value) for value in start))
    current_value = subset_objective(current, model)
    iterations = 0
    while True:
        selected_set = set(current)
        best, best_value = current, current_value
        for outgoing in current:
            retained = tuple(value for value in current if value != outgoing)
            scores = (
                current_value
                - model["linear"][outgoing]
                - model["pair"][outgoing, list(retained)].sum()
                + model["linear"]
                + model["pair"][:, list(retained)].sum(axis=1)
            )
            scores[list(selected_set)] = -np.inf
            incoming = int(np.argmax(scores))
            candidate = tuple(sorted((*retained, incoming)))
            value = float(scores[incoming])
            if better(candidate, value, best, best_value):
                best, best_value = candidate, value
        if best_value <= current_value + 1e-12:
            return current, current_value, iterations
        current, current_value = best, best_value
        iterations += 1
        if iterations > n * int(model["k"]):
            raise RuntimeError("Stage29 one-swap search exceeded iteration guard")


def beam_start(model: dict[str, Any], width: int) -> tuple[int, ...]:
    n, k = len(model["linear"]), int(model["k"])
    layer: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for _ in range(k):
        proposals: dict[tuple[int, ...], float] = {}
        for selected, value in layer:
            chosen = set(selected)
            increments = model["linear"].copy()
            if selected:
                increments += model["pair"][:, list(selected)].sum(axis=1)
            for incoming in range(n):
                if incoming in chosen:
                    continue
                candidate = tuple(sorted((*selected, incoming)))
                candidate_value = value + float(increments[incoming])
                old = proposals.get(candidate)
                if old is None or candidate_value > old:
                    proposals[candidate] = candidate_value
        layer = sorted(proposals.items(), key=lambda item: (-item[1], item[0]))[:width]
    return layer[0][0]


def exact_oracle(model: dict[str, Any], maximum_states: int) -> dict[str, Any] | None:
    n, k = len(model["linear"]), int(model["k"])
    state_count = math.comb(n, k)
    if state_count > maximum_states:
        return None
    best: tuple[int, ...] | None = None
    best_value = -math.inf
    for selected in itertools.combinations(range(n), k):
        value = subset_objective(selected, model)
        if best is None or better(selected, value, best, best_value):
            best, best_value = selected, value
    return {"selected": best, "objective": best_value, "state_count": state_count}


def anneal(model: dict[str, Any], sampler: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    n, k = len(model["linear"]), int(model["k"])
    selected = sorted(rng.sample(range(n), k))
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[selected] = True
    current_value = subset_objective(selected, model)
    best, best_value = tuple(selected), current_value
    accepted = 0
    sweeps = int(sampler["sweeps_per_read"])
    start_temp = float(sampler["temperature_start"])
    end_temp = float(sampler["temperature_end"])
    for step in range(sweeps):
        position = rng.randrange(k)
        outgoing = selected[position]
        incoming = rng.randrange(n)
        while selected_mask[incoming]:
            incoming = rng.randrange(n)
        retained = [value for value in selected if value != outgoing]
        delta = (
            float(model["linear"][incoming] - model["linear"][outgoing])
            + float(model["pair"][incoming, retained].sum())
            - float(model["pair"][outgoing, retained].sum())
        )
        progress = step / max(1, sweeps - 1)
        temperature = max(end_temp, start_temp * (end_temp / start_temp) ** progress)
        if delta >= 0 or rng.random() < math.exp(max(-700.0, delta / temperature)):
            selected_mask[outgoing] = False
            selected_mask[incoming] = True
            selected[position] = incoming
            current_value += delta
            accepted += 1
            candidate = tuple(sorted(selected))
            if better(candidate, current_value, best, best_value):
                best, best_value = candidate, current_value
    return {"selected": best, "objective": best_value, "acceptance_fraction": accepted / max(1, sweeps)}


def qubo_record(model: dict[str, Any], selected: tuple[int, ...], objective: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    n, k = len(model["linear"]), int(model["k"])
    penalty = float(objective["cardinality_penalty"])
    qubo_linear = -model["linear"] + penalty * (1 - 2 * k)
    qubo_pair = -model["pair"] + 2 * penalty
    upper = qubo_pair[np.triu_indices(n, 1)]
    absolute = np.concatenate((np.abs(qubo_linear), np.abs(upper)))
    nonzero = absolute[absolute > 1e-15]
    dynamic_range = float(nonzero.max() / nonzero.min())
    constant = penalty * k * k
    chosen = np.asarray(selected, dtype=int)
    energy = float(constant + qubo_linear[chosen].sum() + np.triu(qubo_pair[np.ix_(chosen, chosen)], 1).sum())
    expected = -subset_objective(selected, model)
    residual = abs(energy - expected)
    digest = hashlib.sha256()
    digest.update(np.asarray(model["indices"], dtype=np.int32).tobytes())
    digest.update(np.asarray(qubo_linear, dtype=np.float64).tobytes())
    digest.update(np.asarray(upper, dtype=np.float64).tobytes())
    couplers = n * (n - 1) // 2
    direct_ready = (
        n <= int(gate["direct_qpu_max_logical_variables"])
        and couplers <= int(gate["direct_qpu_max_quadratic_couplers"])
        and dynamic_range <= float(gate["direct_qpu_max_coefficient_dynamic_range"])
    )
    return {
        "logical_variable_count": n,
        "quadratic_coupler_count": couplers,
        "coupler_density": 1.0,
        "coefficient_minimum_absolute_nonzero": float(nonzero.min()),
        "coefficient_maximum_absolute": float(nonzero.max()),
        "coefficient_dynamic_range": dynamic_range,
        "qubo_sha256": digest.hexdigest().upper(),
        "selected_qubo_energy": energy,
        "equivalence_residual": residual,
        "direct_qpu_ready_under_frozen_thresholds": direct_ready,
    }


def selected_row(pool: dict[str, Any], method: str, selected: tuple[int, ...], model: dict[str, Any], objective: dict[str, Any], runtime: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    components = subset_components(selected, model, objective)
    row = {
        "pool_id": pool["pool_id"],
        "pool_class": pool["pool_class"],
        "posthoc": pool["posthoc"],
        "candidate_count": len(model["linear"]),
        "selected_count": len(selected),
        "method": method,
        "runtime_seconds": runtime,
        "selected_frame_ids": "+".join(model["frame_ids"][index] for index in selected),
        "selected_source_ids": "+".join(model["conformer_ids"][index] for index in selected),
        **components,
    }
    if extra:
        row.update(extra)
    return row


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if outputs["result_json"].exists() and not overwrite:
        raise FileExistsError(f"result exists: {outputs['result_json']}; pass --overwrite")
    loaded = load_inputs(root, config)
    pools = build_pools(loaded["rows"], config)
    objective = config["objective"]
    sampler = config["annealing_sampler"]
    gate = config["gate"]
    pool_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    read_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    pool_records: dict[str, Any] = {}
    primary_strict_wins = 0
    primary_stable_cells = 0
    exact_cells_passed = 0
    exact_cells_total = 0
    for pool_number, pool in enumerate(pools):
        model = build_model(pool, loaded["rows"], loaded["distance"], loaded["centrality"], objective)
        n = len(model["linear"])
        print(json.dumps({"pool_id": pool["pool_id"], "candidate_count": n, "status": "running"}), flush=True)
        for local_index, global_index in enumerate(pool["indices"]):
            frame = loaded["rows"][global_index]
            pool_rows.append({
                "pool_id": pool["pool_id"],
                "pool_class": pool["pool_class"],
                "posthoc": pool["posthoc"],
                "candidate_count": n,
                "local_candidate_index": local_index,
                "global_frame_index": global_index,
                "frame_id": frame["frame_id"],
                "conformer_id": frame["conformer_id"],
                "local_frame_index": frame["local_frame_index"],
                "time_ps": frame["time_ps"],
                "selection_rule": pool["rule"],
            })
        classical: list[dict[str, Any]] = []
        started = time.perf_counter()
        selected, _, iterations = best_one_swap(greedy(model), model)
        classical.append(selected_row(pool, "direct_greedy_plus_best_one_swap", selected, model, objective, time.perf_counter() - started, {"local_search_iterations": iterations, "restart_count": 1, "beam_width": 0, "state_count": ""}))
        started = time.perf_counter()
        selected, _, iterations = best_one_swap(beam_start(model, int(config["classical_baselines"]["beam_width"])), model)
        classical.append(selected_row(pool, "beam32_plus_best_one_swap", selected, model, objective, time.perf_counter() - started, {"local_search_iterations": iterations, "restart_count": 1, "beam_width": int(config["classical_baselines"]["beam_width"]), "state_count": ""}))
        started = time.perf_counter()
        rng = random.Random(int(sampler["base_seed"]) + pool_number * 1000003 + 17)
        random_best: tuple[int, ...] | None = None
        random_best_value = -math.inf
        total_iterations = 0
        for _ in range(int(config["classical_baselines"]["random_multistart_count"])):
            start = tuple(sorted(rng.sample(range(n), int(objective["selected_count"]))))
            candidate, value, iterations = best_one_swap(start, model)
            total_iterations += iterations
            if random_best is None or better(candidate, value, random_best, random_best_value):
                random_best, random_best_value = candidate, value
        if random_best is None:
            raise RuntimeError("random multistart produced no solution")
        classical.append(selected_row(pool, "random32_best_one_swap", random_best, model, objective, time.perf_counter() - started, {"local_search_iterations": total_iterations, "restart_count": int(config["classical_baselines"]["random_multistart_count"]), "beam_width": 0, "state_count": ""}))
        exact_started = time.perf_counter()
        exact = exact_oracle(model, int(config["exact_oracle"]["maximum_states"]))
        exact_runtime = time.perf_counter() - exact_started
        if exact is not None:
            exact_cells_total += 1
            exact_row = selected_row(pool, "exact_oracle", exact["selected"], model, objective, exact_runtime, {"local_search_iterations": 0, "restart_count": 1, "beam_width": 0, "state_count": exact["state_count"]})
            solver_rows.append(exact_row)
        solver_rows.extend(classical)
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
                    "pool_id": pool["pool_id"],
                    "pool_class": pool["pool_class"],
                    "posthoc": pool["posthoc"],
                    "candidate_count": n,
                    "batch": batch,
                    "read": read_index,
                    "seed": seed,
                    "selected_frame_ids": "+".join(model["frame_ids"][index] for index in sampled["selected"]),
                    "selected_source_ids": "+".join(model["conformer_ids"][index] for index in sampled["selected"]),
                    **components,
                    "acceptance_fraction": sampled["acceptance_fraction"],
                    "delta_vs_strong_classical": float(components["objective"]) - float(strong["objective"]),
                }
                read_rows.append(row)
                local_reads.append((row, sampled["selected"]))
            runtime = time.perf_counter() - started
            best_row, best_selected = max(local_reads, key=lambda item: (float(item[0]["objective"]), item[0]["selected_frame_ids"]))
            batch_row = {
                "pool_id": pool["pool_id"],
                "pool_class": pool["pool_class"],
                "posthoc": pool["posthoc"],
                "candidate_count": n,
                "batch": batch,
                "seed": seed,
                "runtime_seconds": runtime,
                "best_objective": float(best_row["objective"]),
                "best_frame_ids": best_row["selected_frame_ids"],
                "best_source_ids": best_row["selected_source_ids"],
                "delta_vs_strong_classical": float(best_row["objective"]) - float(strong["objective"]),
            }
            batch_rows.append(batch_row)
            local_batches.append((batch_row, best_selected))
        best_batch, annealed_selected = max(local_batches, key=lambda item: (float(item[0]["best_objective"]), item[0]["best_frame_ids"]))
        annealed = selected_row(pool, "fixed_cardinality_qubo_annealing", annealed_selected, model, objective, sum(float(item[0]["runtime_seconds"]) for item in local_batches), {"local_search_iterations": 0, "restart_count": int(sampler["batch_count"]) * int(sampler["reads_per_batch"]), "beam_width": 0, "state_count": ""})
        solver_rows.append(annealed)
        annealed_value = float(annealed["objective"])
        strong_value = float(strong["objective"])
        within_fraction = sum(float(item[0]["best_objective"]) >= annealed_value - float(gate["objective_tolerance"]) for item in local_batches) / len(local_batches)
        stable = within_fraction >= float(gate["minimum_batch_fraction_within_tolerance"])
        exact_value = None if exact is None else float(exact["objective"])
        exact_gap = None if exact_value is None else exact_value - annealed_value
        exact_passed = exact_gap is None or exact_gap <= float(gate["maximum_exact_gap"]) + 1e-12
        if exact is not None and exact_passed:
            exact_cells_passed += 1
        strict_win = annealed_value - strong_value > float(gate["minimum_strict_gain"])
        if pool["pool_class"] == "primary_scaling":
            primary_strict_wins += int(strict_win)
            primary_stable_cells += int(stable)
        model_record = qubo_record(model, annealed_selected, objective, gate)
        model_rows.append({"pool_id": pool["pool_id"], "pool_class": pool["pool_class"], "posthoc": pool["posthoc"], "candidate_count": n, **model_record})
        pool_records[pool["pool_id"]] = {
            "pool_class": pool["pool_class"],
            "posthoc": pool["posthoc"],
            "candidate_count": n,
            "selected_count": int(objective["selected_count"]),
            "strong_classical_method": strong["method"],
            "strong_classical_objective": strong_value,
            "annealing_objective": annealed_value,
            "delta_vs_strong_classical": annealed_value - strong_value,
            "annealing_strict_win": strict_win,
            "annealing_batch_fraction_within_tolerance": within_fraction,
            "annealing_stable": stable,
            "exact_oracle_objective": exact_value,
            "annealing_gap_to_exact": exact_gap,
            "exact_gate_passed": exact_passed,
            "annealing_selected_frame_ids": annealed["selected_frame_ids"].split("+"),
            "annealing_selected_source_ids": annealed["selected_source_ids"].split("+"),
            "qubo": model_record,
        }
    write_csv(outputs["pool_manifest_csv"], pool_rows)
    write_csv(outputs["solver_results_csv"], solver_rows)
    write_csv(outputs["batch_results_csv"], batch_rows)
    write_csv(outputs["read_results_csv"], read_rows)
    write_csv(outputs["model_scaling_csv"], model_rows)
    primary_count = sum(record["pool_class"] == "primary_scaling" for record in pool_records.values())
    equivalence_passed = all(float(row["equivalence_residual"]) <= float(gate["maximum_qubo_equivalence_residual"]) for row in model_rows)
    scaling_complete = primary_count == int(gate["required_primary_cell_count"])
    stability_passed = primary_stable_cells == primary_count
    exactness_passed = exact_cells_total > 0 and exact_cells_passed == exact_cells_total
    novelty_passed = primary_strict_wins >= int(gate["minimum_primary_cells_strictly_above_strong_classical"])
    direct_qpu_passed = all(bool(row["direct_qpu_ready_under_frozen_thresholds"]) for row in model_rows if row["pool_class"] == "primary_scaling")
    decision = {
        "input_gate_passed": True,
        "solver_scaling_complete": scaling_complete,
        "qubo_equivalence_gate_passed": equivalence_passed,
        "exactness_gate_passed": exactness_passed,
        "annealing_stability_gate_passed": stability_passed,
        "primary_cells_strictly_above_strong_classical": primary_strict_wins,
        "solver_novelty_gate_passed": novelty_passed,
        "direct_qpu_readiness_gate_passed": direct_qpu_passed,
        "new_docking_jobs_authorized_by_this_stage": False,
        "quantum_hardware_authorized": False,
    }
    report = [
        "# Stage 29: PPARG MD pure-QUBO solver scaling",
        "",
        "Frozen structure-only objective: 0.4 mean centrality + 0.6 mean pair distance - 0.1 mean same-trajectory temporal redundancy; k=8.",
        "",
        "| Pool | Class | n | Strong classical | Annealing | Delta | Stable | Exact gap | QUBO variables/couplers |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for pool in pools:
        record = pool_records[pool["pool_id"]]
        gap = "NA" if record["annealing_gap_to_exact"] is None else f"{record['annealing_gap_to_exact']:.8g}"
        report.append(
            f"| {pool['pool_id']} | {record['pool_class']} | {record['candidate_count']} | {record['strong_classical_objective']:.8f} | {record['annealing_objective']:.8f} | {record['delta_vs_strong_classical']:.8g} | {record['annealing_stable']} | {gap} | {record['qubo']['logical_variable_count']}/{record['qubo']['quadratic_coupler_count']} |"
        )
    report += [
        "",
        f"Scaling/equivalence/exactness gates: **{'PASS' if scaling_complete and equivalence_passed and exactness_passed else 'NO-GO'}**.",
        f"Annealing stability gate: **{'PASS' if stability_passed else 'NO-GO'}**.",
        f"Solver novelty gate: **{'PASS' if novelty_passed else 'NO-GO'}** ({primary_strict_wins} primary strict wins).",
        f"Direct-QPU readiness gate: **{'PASS' if direct_qpu_passed else 'NO-GO'}**.",
        "",
        "The 3D6D exclusion is post-hoc sensitivity evidence and does not affect the primary gate. No docking scores, ligand labels, validation/test rows, new docking jobs, or quantum hardware outputs were used.",
    ]
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text("\n".join(report) + "\n", encoding="ascii")
    result = {
        "schema_version": "1.0",
        "status": "stage29_pparg_md_qubo_solver_scaling_complete",
        "config": {"path": config_path.relative_to(root).as_posix(), "sha256": file_sha256(config_path)},
        "implementation": descriptor(root, Path(__file__).resolve()),
        "inputs": {key: descriptor(root, path) for key, path in loaded["paths"].items()},
        "input_statistics": {
            "frame_count": len(loaded["rows"]),
            "raw_feature_count": loaded["raw_feature_count"],
            "distance_maximum_before_normalization": loaded["distance_maximum"],
        },
        "outputs": {key: descriptor(root, path) for key, path in outputs.items() if key not in {"result_json", "audit_json"}},
        "pool_records": pool_records,
        "decision": decision,
        "data_boundary": {
            "docking_scores_read": 0,
            "ligand_labels_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["result_json"], result)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage29_pparg_md_qubo_solver_scaling.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
