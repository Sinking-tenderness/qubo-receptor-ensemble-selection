"""Run the externally authorized Stage78 Advantage2 hardware PoC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import dimod
import networkx as nx
import numpy as np
from dwave.embedding import EmbeddedStructure
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.embedding.zephyr import find_clique_embedding
from dwave.system import DWaveCliqueSampler, DWaveSampler, FixedEmbeddingComposite
from dwave.system.coupling_groups import coupling_groups
from minorminer import find_embedding


TOLERANCE = 1e-10
PAID_ACKNOWLEDGEMENT = "I_ACCEPT_STAGE78_QPU_CHARGES"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty hardware rows: {path}")
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


def verified(root: Path, descriptor: dict[str, Any], label: str) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file() or sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage78 {label} identity differs: {path}")
    if path.stat().st_size != int(descriptor["size_bytes"]):
        raise ValueError(f"Stage78 {label} size differs: {path}")
    return path


def load_instances(root: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for item in result["outputs"]["instance_files"]:
        metadata_path = verified(root, item["metadata"], "instance metadata")
        metadata = read_json(metadata_path)
        bqm_path = verified(root, metadata["bqm"], "instance BQM")
        moves_path = verified(root, metadata["moves"], "instance moves")
        bqm = dimod.BinaryQuadraticModel.from_serializable(read_json(bqm_path))
        with moves_path.open("r", encoding="utf-8", newline="") as handle:
            moves = list(csv.DictReader(handle))
        if len(moves) != bqm.num_variables:
            raise ValueError(f"move/BQM size mismatch for {metadata['instance_id']}")
        instances.append({"metadata": metadata, "bqm": bqm, "moves": moves})
    return instances


def decode(instance: dict[str, Any], sample: dict[str, int]) -> dict[str, Any]:
    variables = list(instance["bqm"].variables)
    selected = [index for index, name in enumerate(variables) if int(sample[name])]
    removed = [instance["moves"][index]["removed_receptor_id"] for index in selected]
    added = [instance["moves"][index]["added_receptor_id"] for index in selected]
    conflict_free = len(removed) == len(set(removed)) and len(added) == len(set(added))
    deficit_delta = sum(int(instance["moves"][index]["deficit_delta"]) for index in selected)
    feasible = conflict_free and deficit_delta <= 0
    energy = float(instance["bqm"].energy(sample))
    warm = float(instance["metadata"]["warm_energy"])
    guarded_energy = min(warm, energy) if feasible else warm
    exact = float(instance["metadata"]["exact_reference"]["objective_energy"])
    return {
        "feasible": feasible,
        "conflict_free": conflict_free,
        "deficit_delta": deficit_delta,
        "energy": energy,
        "guarded_energy": guarded_energy,
        "strict_improvement": feasible and energy < warm - TOLERANCE,
        "exact_optimum": feasible and math.isclose(energy, exact, abs_tol=1e-8),
        "selected_move_count": len(selected),
    }


def local_validate(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    instances = load_instances(root, result)
    rows: list[dict[str, Any]] = []
    for instance in instances:
        metadata = instance["metadata"]
        variables = list(instance["bqm"].variables)
        zero = {name: 0 for name in variables}
        exact_sample = {
            name: int(name in metadata["exact_reference"]["selected_move_variables"])
            for name in variables
        }
        warm_energy = float(instance["bqm"].energy(zero))
        exact_evaluation = decode(instance, exact_sample)
        if not math.isclose(warm_energy, float(metadata["warm_energy"]), abs_tol=1e-8):
            raise ValueError(f"warm-state identity failed for {metadata['instance_id']}")
        if not exact_evaluation["exact_optimum"]:
            raise ValueError(f"exact certificate failed for {metadata['instance_id']}")
        rows.append(
            {
                "instance_id": metadata["instance_id"],
                "logical_variable_count": instance["bqm"].num_variables,
                "interaction_count": instance["bqm"].num_interactions,
                "warm_identity": True,
                "exact_certificate": True,
            }
        )
    return {
        "status": "stage78_local_execution_bundle_valid",
        "instance_count": len(rows),
        "instances": rows,
        "cloud_queries": 0,
        "qpu_jobs": 0,
        "qpu_reads": 0,
    }


def qpu_sampler(protocol: dict[str, Any], solver_name: str | None) -> DWaveSampler:
    selector: Any = solver_name or dict(protocol["solver_selector"])
    return DWaveSampler(solver=selector, failover=False)


def require_solver_features(qpu: DWaveSampler, protocol: dict[str, Any]) -> None:
    topology = qpu.properties.get("topology", {})
    if str(topology.get("type", "")).lower() != "zephyr":
        raise ValueError(f"Stage78 requires Zephyr, received {topology}")
    if int(qpu.properties.get("num_qubits", len(qpu.nodelist))) < int(
        protocol["minimum_solver_qubit_count"]
    ):
        raise ValueError("Stage78 solver does not meet the minimum qubit count")
    missing = [
        name
        for name in protocol["required_qpu_parameters"]
        if name not in qpu.parameters
    ]
    if missing:
        raise ValueError(f"Stage78 QPU lacks required parameters: {missing}")
    missing_properties = [
        name
        for name in protocol["required_solver_properties"]
        if name not in qpu.properties
    ]
    if missing_properties:
        raise ValueError(
            f"Stage78 QPU lacks required properties: {missing_properties}"
        )


def require_preflight_identity(
    qpu: DWaveSampler, preflight_record: dict[str, Any]
) -> None:
    if str(qpu.solver.id) != str(preflight_record["solver_id"]):
        raise ValueError("Stage78 live solver differs from the frozen preflight solver")
    if qpu.properties.get("graph_id") != preflight_record.get("graph_id"):
        raise ValueError("Stage78 live graph_id differs from the frozen preflight graph")
    if graph_sha256(qpu) != str(preflight_record["working_graph_sha256"]):
        raise ValueError("Stage78 live working graph differs from the frozen preflight graph")


def graph_sha256(qpu: DWaveSampler) -> str:
    payload = {
        "nodes": sorted(int(node) for node in qpu.nodelist),
        "edges": sorted(
            [sorted((int(left), int(right))) for left, right in qpu.edgelist]
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest().upper()


def embedding_metrics(embedding: dict[str, Any]) -> dict[str, Any]:
    lengths = [len(chain) for chain in embedding.values()]
    return {
        "logical_variable_count": len(embedding),
        "physical_qubit_count": sum(lengths),
        "minimum_chain_length": min(lengths),
        "maximum_chain_length": max(lengths),
    }


def preflight(
    root: Path,
    output_root: Path,
    result: dict[str, Any],
    solver_name: str | None,
) -> dict[str, Any]:
    protocol = result["hardware_protocol"]
    instances = load_instances(root, result)
    qpu = qpu_sampler(protocol, solver_name)
    try:
        require_solver_features(qpu, protocol)
        target_graph = qpu.to_networkx_graph()
        identity = dict(qpu.solver.identity)
        graph_id = qpu.properties.get("graph_id")
        embeddings: dict[str, Any] = {}
        required_embedding_count = int(protocol["preflight"]["embedding_count"])
        with DWaveCliqueSampler(
            solver={"identity": identity}, failover=False
        ) as clique_sampler:
            if int(clique_sampler.largest_clique_size) < max(
                item["bqm"].num_variables for item in instances
            ):
                raise ValueError("live working graph cannot embed the frozen K40 cap")
            for variable_count in sorted(
                {item["bqm"].num_variables for item in instances}
            ):
                labels = [f"m{index:03d}" for index in range(variable_count)]
                candidates: list[dict[str, list[int]]] = [
                    {
                        str(variable): [int(qubit) for qubit in chain]
                        for variable, chain in clique_sampler.clique(labels).items()
                    }
                ]
                source_edges = list(itertools.combinations(labels, 2))
                for seed in protocol["preflight"]["embedding_random_seeds"]:
                    if len(candidates) >= required_embedding_count:
                        break
                    candidate = find_embedding(
                        source_edges,
                        list(qpu.edgelist),
                        random_seed=int(seed),
                        timeout=int(protocol["preflight"]["embedding_timeout_seconds"]),
                        tries=int(protocol["preflight"]["embedding_tries"]),
                    )
                    if len(candidate) != variable_count:
                        continue
                    normalized = {
                        str(variable): [int(qubit) for qubit in chain]
                        for variable, chain in candidate.items()
                    }
                    signature = json.dumps(normalized, sort_keys=True)
                    if all(
                        json.dumps(existing, sort_keys=True) != signature
                        for existing in candidates
                    ):
                        candidates.append(normalized)
                if len(candidates) < required_embedding_count:
                    residual = target_graph.copy()
                    residual.remove_nodes_from(
                        {
                            qubit
                            for chain in candidates[0].values()
                            for qubit in chain
                        }
                    )
                    candidate = find_clique_embedding(labels, target_graph=residual)
                    if len(candidate) == variable_count:
                        candidates.append(
                            {
                                str(variable): [int(qubit) for qubit in chain]
                                for variable, chain in candidate.items()
                            }
                        )
                if len(candidates) < required_embedding_count:
                    raise ValueError(
                        f"Stage78 found only {len(candidates)} distinct K{variable_count} embeddings"
                    )
                entries: list[dict[str, Any]] = []
                for embedding_index, embedding in enumerate(
                    candidates[:required_embedding_count]
                ):
                    metrics = embedding_metrics(embedding)
                    if metrics["maximum_chain_length"] > int(
                        protocol["actual_embedding_gate"]["maximum_chain_length"]
                    ):
                        raise ValueError(
                            f"actual K{variable_count} embedding {embedding_index} failed chain gate"
                        )
                    if metrics["physical_qubit_count"] > int(
                        protocol["actual_embedding_gate"][
                            "maximum_physical_qubit_count"
                        ]
                    ):
                        raise ValueError(
                            f"actual K{variable_count} embedding {embedding_index} failed size gate"
                        )
                    entries.append(
                        {
                            "embedding_index": embedding_index,
                            "embedding": embedding,
                            "metrics": metrics,
                            "embedding_sha256": hashlib.sha256(
                                json.dumps(
                                    embedding,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("ascii")
                            ).hexdigest().upper(),
                        }
                    )
                embeddings[str(variable_count)] = entries
        snapshot = {
            "status": "stage78_advantage2_external_preflight_ok",
            "solver_id": str(qpu.solver.id),
            "solver_identity": identity,
            "graph_id": graph_id,
            "solver_topology": qpu.properties.get("topology"),
            "solver_num_qubits_property": int(
                qpu.properties.get("num_qubits", len(qpu.nodelist))
            ),
            "working_qubit_count": len(qpu.nodelist),
            "working_coupler_count": len(qpu.edgelist),
            "working_graph_sha256": graph_sha256(qpu),
            "annealing_time_range": qpu.properties.get("annealing_time_range"),
            "max_anneal_schedule_points": qpu.properties.get(
                "max_anneal_schedule_points"
            ),
            "h_range": qpu.properties.get("h_range"),
            "extended_j_range": qpu.properties.get("extended_j_range"),
            "per_group_coupling_range": qpu.properties.get(
                "per_group_coupling_range"
            ),
            "embeddings": embeddings,
            "cloud_queries": 1,
            "qpu_jobs": 0,
            "qpu_reads": 0,
        }
        write_json(output_root / "preflight.json", snapshot)
        return snapshot
    finally:
        qpu.close()


def paid_authorized(flag: bool) -> None:
    if not flag or os.environ.get("STAGE78_QPU_ACK") != PAID_ACKNOWLEDGEMENT:
        raise PermissionError(
            "Paid Stage78 sampling requires --authorize-paid-qpu and "
            f"STAGE78_QPU_ACK={PAID_ACKNOWLEDGEMENT}"
        )


def deterministic_gauges(variable_count: int, count: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    gauges = [np.ones(variable_count, dtype=np.int8)]
    while len(gauges) < count:
        candidate = np.where(rng.random(variable_count) < 0.5, -1, 1).astype(
            np.int8
        )
        if not any(np.array_equal(candidate, existing) for existing in gauges):
            gauges.append(candidate)
    return [[int(value) for value in gauge] for gauge in gauges]


def gauged_spin_bqm(
    spin_bqm: dimod.BinaryQuadraticModel, gauge: list[int]
) -> tuple[dimod.BinaryQuadraticModel, dict[str, int]]:
    if spin_bqm.vartype is not dimod.SPIN:
        raise ValueError("Stage78 physical gauge requires a SPIN BQM")
    transformed = spin_bqm.copy()
    initial_state: dict[str, int] = {}
    for variable, sign in zip(transformed.variables, gauge):
        if sign == -1:
            transformed.flip_variable(variable)
        initial_state[str(variable)] = -int(sign)
    return transformed, initial_state


def unflip_spin_sample(
    variables: list[str], sample: Any, gauge: list[int]
) -> dict[str, int]:
    return {
        variable: int((int(sign) * int(sample[variable]) + 1) // 2)
        for variable, sign in zip(variables, gauge)
    }


def range_scale(value: float, lower: float, upper: float) -> float:
    if value > 0:
        return value / upper
    if value < 0:
        return value / lower
    return 0.0


def physical_scale_factor(
    physical_bqm: dimod.BinaryQuadraticModel,
    qpu: DWaveSampler,
) -> float:
    h_lower, h_upper = (float(value) for value in qpu.properties["h_range"])
    j_lower, j_upper = (
        float(value) for value in qpu.properties["extended_j_range"]
    )
    factors = [1.0]
    factors.extend(
        range_scale(float(value), h_lower, h_upper)
        for value in physical_bqm.linear.values()
    )
    factors.extend(
        range_scale(float(value), j_lower, j_upper)
        for value in physical_bqm.quadratic.values()
    )
    group_lower, group_upper = (
        float(value) for value in qpu.properties["per_group_coupling_range"]
    )
    hardware_graph = qpu.to_networkx_graph()
    for group in coupling_groups(hardware_graph):
        total = 0.0
        for left, right in group:
            if physical_bqm.has_interaction(left, right):
                total += float(physical_bqm.get_quadratic(left, right))
        factors.append(range_scale(total, group_lower, group_upper))
    return max(factors)


def common_gauge_scale(
    spin_bqm: dimod.BinaryQuadraticModel,
    embedding: dict[str, list[int]],
    gauges: list[list[int]],
    chain_prefactor: float,
    qpu: DWaveSampler,
) -> tuple[float, float]:
    unscaled_chain_strength = float(
        uniform_torque_compensation(
            spin_bqm, embedding=embedding, prefactor=chain_prefactor
        )
    )
    structure = EmbeddedStructure(list(qpu.edgelist), embedding)
    factors: list[float] = []
    for gauge in gauges:
        transformed, _ = gauged_spin_bqm(spin_bqm, gauge)
        physical = structure.embed_bqm(
            transformed,
            chain_strength=unscaled_chain_strength,
            smear_vartype=dimod.SPIN,
        )
        factors.append(physical_scale_factor(physical, qpu))
    return max(factors), unscaled_chain_strength


def reverse_schedule(ramp_us: float, pause_us: float, s_minimum: float) -> list[list[float]]:
    if pause_us > 0:
        return [
            [0.0, 1.0],
            [ramp_us, s_minimum],
            [ramp_us + pause_us, s_minimum],
            [2.0 * ramp_us + pause_us, 1.0],
        ]
    return [[0.0, 1.0], [ramp_us, s_minimum], [2.0 * ramp_us, 1.0]]


def embedding_for(
    instance: dict[str, Any],
    preflight_record: dict[str, Any],
    embedding_index: int,
) -> dict[str, list[int]]:
    records = preflight_record["embeddings"][str(instance["bqm"].num_variables)]
    record = next(
        item["embedding"]
        for item in records
        if int(item["embedding_index"]) == embedding_index
    )
    return {str(variable): [int(qubit) for qubit in chain] for variable, chain in record.items()}


def sample_condition(
    qpu: DWaveSampler,
    instance: dict[str, Any],
    embedding: dict[str, list[int]],
    mode: str,
    condition_id: str,
    embedding_index: int,
    chain_prefactor: float,
    schedule: list[list[float]] | None,
    reads_per_gauge: int,
    gauge_count: int,
    gauge_seed: int,
    gauge_indices: list[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variables = [str(variable) for variable in instance["bqm"].variables]
    spin_bqm = instance["bqm"].change_vartype(dimod.SPIN, inplace=False)
    gauges = deterministic_gauges(len(variables), gauge_count, gauge_seed)
    common_scale, unscaled_chain_strength = common_gauge_scale(
        spin_bqm, embedding, gauges, chain_prefactor, qpu
    )
    programmed_chain_strength = unscaled_chain_strength / common_scale
    if schedule is not None:
        qpu.validate_anneal_schedule(schedule)
    rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    selected_gauge_indices = (
        list(range(len(gauges))) if gauge_indices is None else list(gauge_indices)
    )
    if not selected_gauge_indices or any(
        index < 0 or index >= len(gauges) for index in selected_gauge_indices
    ):
        raise ValueError("Stage78 gauge index selection is invalid")
    for gauge_index in selected_gauge_indices:
        gauge = gauges[gauge_index]
        transformed, initial_state = gauged_spin_bqm(spin_bqm, gauge)
        transformed.scale(1.0 / common_scale)
        sampler = FixedEmbeddingComposite(
            qpu, embedding=embedding, scale_aware=True
        )
        parameters: dict[str, Any] = {
            "num_reads": reads_per_gauge,
            "answer_mode": "raw",
            "auto_scale": False,
            "chain_strength": programmed_chain_strength,
            "return_embedding": True,
            "label": f"stage78-{instance['metadata']['instance_id']}-{condition_id}-g{gauge_index}",
        }
        if mode == "reverse":
            parameters.update(
                {
                    "initial_state": initial_state,
                    "anneal_schedule": schedule,
                    "reinitialize_state": True,
                }
            )
        sampleset = sampler.sample(transformed, **parameters)
        sampleset.resolve()
        dtype_names = set(sampleset.record.dtype.names or ())
        for read_index, datum in enumerate(sampleset.data()):
            sample = unflip_spin_sample(variables, datum.sample, gauge)
            evaluation = decode(instance, sample)
            chain_break_fraction = (
                float(datum.chain_break_fraction)
                if "chain_break_fraction" in dtype_names
                else 0.0
            )
            main_analysis_eligible = math.isclose(
                chain_break_fraction, 0.0, abs_tol=TOLERANCE
            )
            rows.append(
                {
                    "instance_id": instance["metadata"]["instance_id"],
                    "condition_id": condition_id,
                    "mode": mode,
                    "embedding_index": embedding_index,
                    "gauge_index": gauge_index,
                    "read_index": read_index,
                    "sample_bits": "".join(str(sample[name]) for name in variables),
                    "num_occurrences": int(datum.num_occurrences),
                    "chain_break_fraction": chain_break_fraction,
                    "main_analysis_eligible": main_analysis_eligible,
                    "guarded_strict_improvement": bool(
                        main_analysis_eligible and evaluation["strict_improvement"]
                    ),
                    "guarded_exact_optimum": bool(
                        main_analysis_eligible and evaluation["exact_optimum"]
                    ),
                    "energy_below_certified_optimum": bool(
                        main_analysis_eligible
                        and evaluation["feasible"]
                        and float(evaluation["energy"])
                        < float(instance["metadata"]["exact_reference"]["objective_energy"])
                        - 1e-8
                    ),
                    **evaluation,
                }
            )
        jobs.append(
            {
                "instance_id": instance["metadata"]["instance_id"],
                "condition_id": condition_id,
                "mode": mode,
                "embedding_index": embedding_index,
                "gauge_index": gauge_index,
                "chain_prefactor": chain_prefactor,
                "unscaled_chain_strength": unscaled_chain_strength,
                "common_physical_scale_factor": common_scale,
                "programmed_chain_strength": programmed_chain_strength,
                "schedule": schedule,
                "reads": reads_per_gauge,
                "timing": sampleset.info.get("timing", {}),
                "problem_id": sampleset.info.get("problem_id"),
            }
        )
    return rows, jobs


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["instance_id"], row["condition_id"], row["mode"])].append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        total = sum(int(row["num_occurrences"]) for row in group)
        weighted = lambda field: sum(
            float(row[field]) * int(row["num_occurrences"]) for row in group
        ) / total
        summaries.append(
            {
                "instance_id": key[0],
                "condition_id": key[1],
                "mode": key[2],
                "read_count": total,
                "feasible_read_fraction": weighted("feasible"),
                "strict_improvement_read_fraction": weighted("strict_improvement"),
                "exact_optimum_read_fraction": weighted("exact_optimum"),
                "intact_chain_read_fraction": weighted("main_analysis_eligible"),
                "guarded_strict_improvement_read_fraction": weighted(
                    "guarded_strict_improvement"
                ),
                "guarded_exact_optimum_read_fraction": weighted(
                    "guarded_exact_optimum"
                ),
                "mean_chain_break_fraction": weighted("chain_break_fraction"),
                "best_guarded_energy": min(
                    float(row["guarded_energy"])
                    if bool(row["main_analysis_eligible"])
                    else float("inf")
                    for row in group
                ),
                "below_certified_optimum_count": sum(
                    int(bool(row["energy_below_certified_optimum"]))
                    * int(row["num_occurrences"])
                    for row in group
                ),
                "gauge_count": len({int(row["gauge_index"]) for row in group}),
            }
        )
        if math.isinf(float(summaries[-1]["best_guarded_energy"])):
            summaries[-1]["best_guarded_energy"] = None
    return summaries


def block_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["instance_id"]),
                str(row["mode"]),
                int(row["embedding_index"]),
                int(row["gauge_index"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        total = sum(int(row["num_occurrences"]) for row in group)
        strict_count = sum(
            int(bool(row["guarded_strict_improvement"]))
            * int(row["num_occurrences"])
            for row in group
        )
        exact_count = sum(
            int(bool(row["guarded_exact_optimum"])) * int(row["num_occurrences"])
            for row in group
        )
        intact_count = sum(
            int(bool(row["main_analysis_eligible"])) * int(row["num_occurrences"])
            for row in group
        )
        output.append(
            {
                "instance_id": key[0],
                "mode": key[1],
                "embedding_index": key[2],
                "gauge_index": key[3],
                "read_count": total,
                "intact_chain_read_count": intact_count,
                "intact_chain_read_fraction": intact_count / total,
                "guarded_strict_improvement_read_count": strict_count,
                "guarded_strict_improvement_read_fraction": strict_count / total,
                "guarded_exact_optimum_read_count": exact_count,
                "guarded_exact_optimum_read_fraction": exact_count / total,
                "strict_improvement_recovered": strict_count > 0,
                "below_certified_optimum_count": sum(
                    int(bool(row["energy_below_certified_optimum"]))
                    * int(row["num_occurrences"])
                    for row in group
                ),
            }
        )
    return output


def qpu_access_time_microseconds(jobs: list[dict[str, Any]]) -> int:
    return sum(int(job.get("timing", {}).get("qpu_access_time", 0) or 0) for job in jobs)


def require_qpu_time_budget(
    jobs: list[dict[str, Any]], protocol: dict[str, Any], prior_microseconds: int = 0
) -> None:
    used = prior_microseconds + qpu_access_time_microseconds(jobs)
    allowed = int(
        float(protocol["maximum_planned_qpu_access_time_seconds"]) * 1_000_000
    )
    if used > allowed:
        raise RuntimeError(
            f"Stage78 QPU access-time hard limit exceeded: {used} > {allowed} microseconds"
        )


def exact_paired_block_pvalue(reverse_only: int, forward_only: int) -> float:
    discordant = reverse_only + forward_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(0, min(reverse_only, forward_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def calibration(
    root: Path,
    output_root: Path,
    result: dict[str, Any],
    solver_name: str | None,
) -> dict[str, Any]:
    preflight_record = read_json(output_root / "preflight.json")
    instances = load_instances(root, result)
    instance = next(
        item
        for item in instances
        if item["metadata"]["role"] == "calibration_diagnostic"
    )
    protocol = result["hardware_protocol"]
    qpu = qpu_sampler(protocol, solver_name)
    try:
        require_solver_features(qpu, protocol)
        require_preflight_identity(qpu, preflight_record)
        embedding_index = 0
        embedding = embedding_for(instance, preflight_record, embedding_index)
        rows: list[dict[str, Any]] = []
        jobs: list[dict[str, Any]] = []
        calibration_protocol = protocol["calibration"]
        condition_index = 0
        for chain_prefactor in calibration_protocol["chain_strength_prefactors"]:
            for s_minimum in calibration_protocol["reverse_s_minimum_values"]:
                for pause_us in calibration_protocol["pause_microseconds"]:
                    condition_id = f"cal{condition_index:02d}"
                    schedule = reverse_schedule(
                        float(calibration_protocol["ramp_microseconds"]),
                        float(pause_us),
                        float(s_minimum),
                    )
                    condition_rows, condition_jobs = sample_condition(
                        qpu,
                        instance,
                        embedding,
                        "reverse",
                        condition_id,
                        embedding_index,
                        float(chain_prefactor),
                        schedule,
                        int(calibration_protocol["reads_per_gauge"]),
                        int(calibration_protocol["gauge_count"]),
                        int(calibration_protocol["gauge_seed_base"]) + condition_index,
                    )
                    for row in condition_rows:
                        row.update(
                            {
                                "chain_prefactor": float(chain_prefactor),
                                "reverse_s_minimum": float(s_minimum),
                                "pause_microseconds": float(pause_us),
                            }
                        )
                    rows.extend(condition_rows)
                    jobs.extend(condition_jobs)
                    require_qpu_time_budget(jobs, protocol)
                    condition_index += 1
        summaries = summarize_rows(rows)
        for summary in summaries:
            example = next(row for row in rows if row["condition_id"] == summary["condition_id"])
            gauge_best_energies: list[float] = []
            for gauge_index in range(int(calibration_protocol["gauge_count"])):
                gauge_rows = [
                    row
                    for row in rows
                    if row["condition_id"] == summary["condition_id"]
                    and int(row["gauge_index"]) == gauge_index
                    and bool(row["main_analysis_eligible"])
                ]
                gauge_best_energies.append(
                    min(float(row["guarded_energy"]) for row in gauge_rows)
                    if gauge_rows
                    else float(instance["metadata"]["warm_energy"])
                )
            summary.update(
                {
                    "chain_prefactor": example["chain_prefactor"],
                    "reverse_s_minimum": example["reverse_s_minimum"],
                    "pause_microseconds": example["pause_microseconds"],
                    "cross_gauge_best_energy_range": max(gauge_best_energies)
                    - min(gauge_best_energies),
                }
            )
        selected = sorted(
            summaries,
            key=lambda row: (
                -float(row["intact_chain_read_fraction"]),
                float(row["best_guarded_energy"])
                if row["best_guarded_energy"] is not None
                else float("inf"),
                float(row["cross_gauge_best_energy_range"]),
                float(row["chain_prefactor"]),
                -float(row["reverse_s_minimum"]),
                float(row["pause_microseconds"]),
            ),
        )[0]
        write_csv(output_root / "calibration_reads.csv", rows)
        write_csv(output_root / "calibration_summary.csv", summaries)
        planned_jobs = int(calibration_protocol["planned_qpu_job_count"])
        planned_reads = int(calibration_protocol["planned_qpu_read_count"])
        if len(jobs) != planned_jobs or sum(int(job["reads"]) for job in jobs) != planned_reads:
            raise RuntimeError("Stage78 calibration submission count differs from preregistration")
        record = {
            "status": "stage78_advantage2_calibration_complete",
            "solver_id": str(qpu.solver.id),
            "graph_id": qpu.properties.get("graph_id"),
            "working_graph_sha256": graph_sha256(qpu),
            "calibration_instance_id": instance["metadata"]["instance_id"],
            "embedding_index": embedding_index,
            "selected_condition": selected,
            "job_count": len(jobs),
            "qpu_reads": sum(int(job["reads"]) for job in jobs),
            "qpu_access_time_microseconds": qpu_access_time_microseconds(jobs),
            "jobs": jobs,
        }
        write_json(output_root / "calibration.json", record)
        return record
    finally:
        qpu.close()


def confirmation(
    root: Path,
    output_root: Path,
    result: dict[str, Any],
    solver_name: str | None,
) -> dict[str, Any]:
    preflight_record = read_json(output_root / "preflight.json")
    calibration_record = read_json(output_root / "calibration.json")
    protocol = result["hardware_protocol"]
    selected = calibration_record["selected_condition"]
    paid_instances = [
        item
        for item in load_instances(root, result)
        if item["metadata"]["role"]
        in {"confirmation_positive", "confirmation_negative"}
    ]
    qpu = qpu_sampler(protocol, solver_name)
    try:
        require_solver_features(qpu, protocol)
        require_preflight_identity(qpu, preflight_record)
        rows: list[dict[str, Any]] = []
        jobs: list[dict[str, Any]] = []
        confirm_protocol = protocol["confirmation"]
        schedule = reverse_schedule(
            float(protocol["calibration"]["ramp_microseconds"]),
            float(selected["pause_microseconds"]),
            float(selected["reverse_s_minimum"]),
        )
        expected_instance_count = int(confirm_protocol["planned_default_instance_count"])
        if len(paid_instances) != expected_instance_count:
            raise ValueError("Stage78 confirmation instance count differs from preregistration")
        embedding_count = int(confirm_protocol["embedding_count"])
        gauge_count = int(confirm_protocol["gauge_count"])
        blocks = list(
            itertools.product(
                range(len(paid_instances)), range(embedding_count), range(gauge_count)
            )
        )
        order_rng = np.random.default_rng(
            int(confirm_protocol["gauge_seed_base"]) + 99_999
        )
        order_rng.shuffle(blocks)
        calibration_time = int(
            calibration_record.get("qpu_access_time_microseconds", 0)
        )
        for block_order, (instance_index, embedding_index, gauge_index) in enumerate(
            blocks
        ):
            instance = paid_instances[instance_index]
            embedding = embedding_for(instance, preflight_record, embedding_index)
            gauge_seed = (
                int(confirm_protocol["gauge_seed_base"])
                + 100 * instance_index
                + embedding_index
            )
            modes = ["reverse", "forward"]
            if int(order_rng.integers(0, 2)):
                modes.reverse()
            for mode in modes:
                condition_rows, condition_jobs = sample_condition(
                    qpu,
                    instance,
                    embedding,
                    mode,
                    f"confirm_{mode}",
                    embedding_index,
                    float(selected["chain_prefactor"]),
                    schedule if mode == "reverse" else None,
                    int(confirm_protocol["reads_per_gauge"]),
                    gauge_count,
                    gauge_seed,
                    gauge_indices=[gauge_index],
                )
                for row in condition_rows:
                    row["paired_block_order"] = block_order
                rows.extend(condition_rows)
                jobs.extend(condition_jobs)
                require_qpu_time_budget(jobs, protocol, calibration_time)
        summaries = summarize_rows(rows)
        blocks_summary = block_summaries(rows)
        write_csv(output_root / "confirmation_reads.csv", rows)
        write_csv(output_root / "confirmation_summary.csv", summaries)
        write_csv(output_root / "confirmation_blocks.csv", blocks_summary)

        planned_jobs = int(confirm_protocol["planned_qpu_job_count"])
        planned_reads = int(confirm_protocol["planned_qpu_read_count"])
        actual_reads = sum(int(job["reads"]) for job in jobs)
        if len(jobs) != planned_jobs or actual_reads != planned_reads:
            raise RuntimeError("Stage78 confirmation submission count differs from preregistration")

        endpoints: list[dict[str, Any]] = []
        for instance in paid_instances:
            metadata = instance["metadata"]
            instance_id = metadata["instance_id"]
            instance_blocks = [
                row for row in blocks_summary if row["instance_id"] == instance_id
            ]
            reverse_blocks = {
                (int(row["embedding_index"]), int(row["gauge_index"])): bool(
                    row["strict_improvement_recovered"]
                )
                for row in instance_blocks
                if row["mode"] == "reverse"
            }
            forward_blocks = {
                (int(row["embedding_index"]), int(row["gauge_index"])): bool(
                    row["strict_improvement_recovered"]
                )
                for row in instance_blocks
                if row["mode"] == "forward"
            }
            reverse_successes = sum(reverse_blocks.values())
            forward_successes = sum(forward_blocks.values())
            reverse_only = sum(
                reverse_blocks[key] and not forward_blocks[key]
                for key in reverse_blocks
            )
            forward_only = sum(
                forward_blocks[key] and not reverse_blocks[key]
                for key in reverse_blocks
            )
            mode_summaries = {
                row["mode"]: row
                for row in summaries
                if row["instance_id"] == instance_id
            }
            reverse_fraction = float(
                mode_summaries["reverse"][
                    "guarded_strict_improvement_read_fraction"
                ]
            )
            forward_fraction = float(
                mode_summaries["forward"][
                    "guarded_strict_improvement_read_fraction"
                ]
            )
            below_exact = sum(
                int(row["below_certified_optimum_count"]) for row in instance_blocks
            )
            if metadata["role"] == "confirmation_positive":
                passed = (
                    reverse_successes >= 13
                    and reverse_fraction - forward_fraction >= 0.05 - TOLERANCE
                    and below_exact == 0
                )
            else:
                guarded_improvements = sum(
                    int(row["guarded_strict_improvement_read_count"])
                    for row in instance_blocks
                )
                passed = guarded_improvements == 0 and below_exact == 0
            endpoints.append(
                {
                    "instance_id": instance_id,
                    "role": metadata["role"],
                    "reverse_successful_blocks": reverse_successes,
                    "forward_successful_blocks": forward_successes,
                    "total_blocks_per_mode": embedding_count * gauge_count,
                    "reverse_guarded_improvement_read_fraction": reverse_fraction,
                    "forward_guarded_improvement_read_fraction": forward_fraction,
                    "reverse_minus_forward_read_fraction": reverse_fraction
                    - forward_fraction,
                    "reverse_only_successful_blocks": reverse_only,
                    "forward_only_successful_blocks": forward_only,
                    "paired_block_exact_pvalue": exact_paired_block_pvalue(
                        reverse_only, forward_only
                    ),
                    "below_certified_optimum_count": below_exact,
                    "primary_endpoint_passed": passed,
                }
            )
        primary_passed = all(bool(row["primary_endpoint_passed"]) for row in endpoints)
        record = {
            "status": (
                "stage78_advantage2_physical_poc_passed"
                if primary_passed
                else "stage78_advantage2_physical_poc_not_passed"
            ),
            "solver_id": str(qpu.solver.id),
            "graph_id": qpu.properties.get("graph_id"),
            "working_graph_sha256": graph_sha256(qpu),
            "selected_calibration_condition": selected,
            "instance_count": len(paid_instances),
            "job_count": len(jobs),
            "qpu_reads": actual_reads,
            "qpu_access_time_microseconds": qpu_access_time_microseconds(jobs),
            "cumulative_qpu_access_time_microseconds": calibration_time
            + qpu_access_time_microseconds(jobs),
            "summaries": summaries,
            "block_summaries": blocks_summary,
            "primary_endpoints": endpoints,
            "primary_endpoint_passed": primary_passed,
            "jobs": jobs,
            "claim_boundary": "Physical-hardware PoC only; no quantum advantage or scaling claim.",
        }
        write_json(output_root / "confirmation.json", record)
        return record
    finally:
        qpu.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("local-validate", "preflight", "calibrate", "confirm"),
        required=True,
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("data/stage78_advantage2_reverse_annealing_poc_result.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("external_results/stage78_advantage2_reverse_annealing_poc"),
    )
    parser.add_argument("--solver-name")
    parser.add_argument("--authorize-paid-qpu", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result = read_json((root / args.result).resolve())
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "local-validate":
        output = local_validate(root, result)
        write_json(output_root / "local_validation.json", output)
    elif args.phase == "preflight":
        output = preflight(root, output_root, result, args.solver_name)
    elif args.phase == "calibrate":
        paid_authorized(args.authorize_paid_qpu)
        output = calibration(root, output_root, result, args.solver_name)
    else:
        paid_authorized(args.authorize_paid_qpu)
        output = confirmation(
            root,
            output_root,
            result,
            args.solver_name,
        )
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
