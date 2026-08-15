"""Freeze and validate the Stage78 Advantage2 reverse-annealing PoC inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSolver, TabuSampler
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

try:
    import scripts.run_stage75_explicit_variable_k_cqm as s75
    import scripts.run_stage77_quantum_hardware_interface_gate as s77
except ImportError:
    import run_stage75_explicit_variable_k_cqm as s75
    import run_stage77_quantum_hardware_interface_gate as s77


TOLERANCE = 1e-10


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def stage77_inputs(stage77_config: dict[str, Any], root: Path) -> dict[str, Path]:
    required = (
        "stage72_model_record",
        "stage74_workload_metrics",
        "stage74_cell_comparison",
        "stage74_solver_trials",
    )
    return {
        name: s75.verified(root, stage77_config["inputs"][name], name)
        for name in required
    }


def rebuild_local_records(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stage77_config_path = s75.verified(
        root, config["inputs"]["stage77_config"], "Stage77 config"
    )
    stage77_config = s75.read_json(stage77_config_path)
    for name in ("stage77_result", "stage77_audit", "stage77_local_metrics"):
        s75.verified(root, config["inputs"][name], name)
    stage77_result = s75.read_json(root / config["inputs"]["stage77_result"]["path"])
    stage77_audit = s75.read_json(root / config["inputs"]["stage77_audit"]["path"])
    if not stage77_result["route_gate"]["local_reverse_annealing_poc_gate_passed"]:
        raise ValueError("Stage78 requires the passing Stage77 local hardware gate")
    if stage77_audit["status"] != "stage77_quantum_hardware_interface_independent_audit_ok":
        raise ValueError("Stage78 requires the passing Stage77 independent audit")
    cells = s77.source_cells(stage77_config, stage77_inputs(stage77_config, root))
    records, _ = s77.local_subproblems(cells, stage77_config)
    return stage77_config, records


def canonical_records(
    records: list[dict[str, Any]], quantile: float
) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if math.isclose(
            float(record["row"]["reward_quantile"]), quantile, abs_tol=1e-12
        )
    ]
    keys = {
        (
            str(record["row"]["target_id"]),
            int(record["row"]["outer_fold"]),
            int(record["row"]["k"]),
        )
        for record in selected
    }
    if len(keys) != len(selected):
        raise ValueError("Stage78 canonical reward quantile produced duplicate cells")
    return selected


def negative_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    row = record["row"]
    return (
        -int(row["encoded_move_variable_count"]),
        -int(row["k"]),
        int(row["outer_fold"]),
        int(row["subproblem_index"]),
    )


def freeze_instances(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    freeze = config["instance_freeze"]
    positives = sorted(
        [
            record
            for record in records
            if bool(record["row"]["hardware_resolvable_single_move_improvement"])
        ],
        key=lambda record: (
            float(record["row"]["best_single_move_energy_delta"]),
            int(record["row"]["outer_fold"]),
        ),
    )
    if len(positives) != int(freeze["required_positive_instance_count"]):
        raise ValueError(
            f"Stage78 expected two hardware-resolvable positives, found {len(positives)}"
        )

    frozen: list[dict[str, Any]] = []
    for record in positives:
        frozen.append(
            {
                "role": "confirmation_positive",
                "include_in_paid_run": True,
                "record": record,
            }
        )

    for target_id in freeze["negative_control_targets"]:
        candidates = [
            record
            for record in records
            if record["row"]["target_id"] == target_id
            and not bool(record["row"]["improving_single_move_available"])
        ]
        if not candidates:
            raise ValueError(f"Stage78 has no canonical negative control for {target_id}")
        frozen.append(
            {
                "role": "confirmation_negative",
                "include_in_paid_run": True,
                "record": sorted(candidates, key=negative_sort_key)[0],
            }
        )

    diagnostics = sorted(
        [
            record
            for record in records
            if bool(record["row"]["improving_single_move_available"])
            and not bool(record["row"]["hardware_resolvable_single_move_improvement"])
        ],
        key=lambda record: (
            float(record["row"]["best_single_move_energy_delta"]),
            str(record["row"]["target_id"]),
            int(record["row"]["outer_fold"]),
            int(record["row"]["k"]),
        ),
    )
    if len(diagnostics) != int(freeze["required_subresolution_diagnostic_count"]):
        raise ValueError(
            "Stage78 expected one sub-resolution diagnostic improvement instance"
        )
    frozen.append(
        {
            "role": "calibration_diagnostic",
            "include_in_paid_run": True,
            "record": diagnostics[0],
        }
    )
    return frozen


def instance_id(role: str, row: dict[str, Any]) -> str:
    return (
        f"{str(row['target_id']).lower()}_of{int(row['outer_fold'])}_"
        f"k{int(row['k'])}_{role}"
    )


def move_rows(local: dict[str, Any]) -> list[dict[str, Any]]:
    model = local["cell"]["model"]
    rows: list[dict[str, Any]] = []
    for variable, move in zip(local["variable_names"], local["moves"]):
        rows.append(
            {
                "variable": variable,
                "removed_index": int(move["removed_index"]),
                "removed_receptor_id": model["receptor_ids"][move["removed_index"]],
                "added_index": int(move["added_index"]),
                "added_receptor_id": model["receptor_ids"][move["added_index"]],
                "deficit_delta": int(move["deficit_delta"]),
                "single_move_energy_delta": float(move["energy_delta"]),
            }
        )
    return rows


def conflicting_pairs(moves: list[dict[str, Any]]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for left in range(len(moves)):
        for right in range(left + 1, len(moves)):
            if (
                int(moves[left]["removed_index"])
                == int(moves[right]["removed_index"])
                or int(moves[left]["added_index"])
                == int(moves[right]["added_index"])
            ):
                pairs.add((left, right))
    return pairs


def exact_milp(local: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    bqm = local["bqm"]
    names = list(bqm.variables)
    index = {name: position for position, name in enumerate(names)}
    interactions = [
        (index[left], index[right], float(value))
        for (left, right), value in bqm.quadratic.items()
    ]
    variable_count = len(names)
    product_count = len(interactions)
    total_count = variable_count + product_count
    objective = np.zeros(total_count, dtype=float)
    for name, value in bqm.linear.items():
        objective[index[name]] = float(value)
    for product_index, (_, _, value) in enumerate(interactions):
        objective[variable_count + product_index] = value

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(coefficients: Iterable[tuple[int, float]], lb: float, ub: float) -> None:
        row = len(lower)
        for column, value in coefficients:
            row_indices.append(row)
            column_indices.append(column)
            values.append(float(value))
        lower.append(float(lb))
        upper.append(float(ub))

    for product_index, (left, right, _) in enumerate(interactions):
        product = variable_count + product_index
        add_constraint(((product, 1.0), (left, -1.0)), -np.inf, 0.0)
        add_constraint(((product, 1.0), (right, -1.0)), -np.inf, 0.0)
        add_constraint(((left, 1.0), (right, 1.0), (product, -1.0)), -np.inf, 1.0)

    conflicts = conflicting_pairs(local["moves"])
    for left, right in sorted(conflicts):
        add_constraint(((left, 1.0), (right, 1.0)), -np.inf, 1.0)

    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(lower), total_count),
    ).tocsr()
    result = milp(
        objective,
        integrality=np.ones(total_count, dtype=int),
        bounds=Bounds(np.zeros(total_count), np.ones(total_count)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={
            "presolve": True,
            "time_limit": float(protocol["time_limit_seconds"]),
            "mip_rel_gap": float(protocol["mip_relative_gap"]),
        },
    )
    if not result.success or int(result.status) != 0 or result.x is None:
        raise ValueError(f"Stage78 exact MILP failed: {result.message}")
    sample = {
        name: int(round(float(result.x[position])))
        for position, name in enumerate(names)
    }
    selected = [position for position, name in enumerate(names) if sample[name]]
    subset, conflict_free = s77.subset_after_moves(
        local["warm_subset"], local["moves"], selected
    )
    if not conflict_free:
        raise ValueError("Stage78 exact MILP returned conflicting moves")
    if sum(int(local["moves"][position]["deficit_delta"]) for position in selected) > 0:
        raise ValueError("Stage78 exact MILP returned a quality-worsening move set")
    bqm_energy = float(bqm.energy(sample))
    true_energy = float(
        s75.variable_energy(local["cell"]["model"], subset, local["cell"]["reward"])
    )
    if not math.isclose(bqm_energy, true_energy, abs_tol=1e-8):
        raise ValueError("Stage78 exact MILP objective differs from frozen objective")
    if not math.isclose(
        float(result.fun) + float(bqm.offset), bqm_energy, abs_tol=1e-8
    ):
        raise ValueError("Stage78 exact MILP linearization objective differs")
    return {
        "status": int(result.status),
        "message": str(result.message),
        "objective_energy": bqm_energy,
        "improvement_from_warm": bqm_energy - float(local["warm_energy"]),
        "selected_move_variables": [names[position] for position in selected],
        "selected_move_count": len(selected),
        "selected_subset": s75.subset_name(local["cell"]["model"], subset),
        "mip_gap": float(getattr(result, "mip_gap", 0.0)),
        "mip_node_count": int(getattr(result, "mip_node_count", 0)),
        "binary_variable_count": variable_count,
        "linearized_product_variable_count": product_count,
        "constraint_count": len(lower),
    }


def sample_evaluation(
    local: dict[str, Any], sampleset: dimod.SampleSet, exact_energy: float
) -> dict[str, Any]:
    names = list(local["bqm"].variables)
    total = 0
    feasible = 0
    improving = 0
    exact_hits = 0
    best_feasible = float(local["warm_energy"])
    for datum in sampleset.data(fields=["sample", "num_occurrences"]):
        occurrences = int(datum.num_occurrences)
        total += occurrences
        selected = [index for index, name in enumerate(names) if int(datum.sample[name])]
        _, conflict_free = s77.subset_after_moves(
            local["warm_subset"], local["moves"], selected
        )
        quality_ok = (
            sum(int(local["moves"][index]["deficit_delta"]) for index in selected)
            <= 0
        )
        if not (conflict_free and quality_ok):
            continue
        feasible += occurrences
        energy = float(local["bqm"].energy(datum.sample))
        best_feasible = min(best_feasible, energy)
        if energy < float(local["warm_energy"]) - TOLERANCE:
            improving += occurrences
        if math.isclose(energy, exact_energy, abs_tol=1e-8):
            exact_hits += occurrences
    if total <= 0:
        raise ValueError("Stage78 classical sampler returned no reads")
    return {
        "read_count": total,
        "feasible_read_fraction": feasible / total,
        "strict_improvement_read_fraction": improving / total,
        "exact_optimum_read_fraction": exact_hits / total,
        "best_guarded_energy": best_feasible,
        "best_guarded_improvement": best_feasible - float(local["warm_energy"]),
        "warm_guard_nonworse": best_feasible <= float(local["warm_energy"]) + TOLERANCE,
    }


def classical_control_rows(
    instance: dict[str, Any], config: dict[str, Any], exact: dict[str, Any]
) -> list[dict[str, Any]]:
    local = instance["record"]
    protocol = config["classical_controls"]
    names = list(local["bqm"].variables)
    zero = {name: 0 for name in names}
    rows: list[dict[str, Any]] = []
    base_seed = int(protocol["seed_base"]) + int(local["row"]["subproblem_index"]) * 100
    methods = (
        "cold_simulated_annealing",
        "warm_simulated_annealing",
        "warm_tabu",
        "warm_steepest_descent",
    )
    for method_index, method in enumerate(methods):
        repeats = (
            1
            if method == "warm_steepest_descent"
            else int(protocol["repeats"])
        )
        for repeat in range(repeats):
            seed = base_seed + method_index * 10 + repeat
            if method == "cold_simulated_annealing":
                sampleset = SimulatedAnnealingSampler().sample(
                    local["bqm"],
                    num_reads=int(protocol["sa_reads_per_repeat"]),
                    num_sweeps=int(protocol["sa_sweeps_per_read"]),
                    beta_schedule_type="geometric",
                    seed=seed,
                )
            elif method == "warm_simulated_annealing":
                sampleset = SimulatedAnnealingSampler().sample(
                    local["bqm"],
                    num_reads=int(protocol["sa_reads_per_repeat"]),
                    num_sweeps=int(protocol["sa_sweeps_per_read"]),
                    beta_schedule_type="geometric",
                    initial_states=[zero],
                    initial_states_generator="tile",
                    seed=seed,
                )
            elif method == "warm_tabu":
                sampleset = TabuSampler().sample(
                    local["bqm"],
                    num_reads=int(protocol["tabu_reads_per_repeat"]),
                    timeout=int(protocol["tabu_timeout_milliseconds"]),
                    initial_states=[zero],
                    initial_states_generator="tile",
                    seed=seed,
                )
            else:
                sampleset = SteepestDescentSolver().sample(
                    local["bqm"],
                    initial_states=[zero],
                    initial_states_generator="tile",
                    num_reads=1,
                    seed=seed,
                )
            rows.append(
                {
                    "instance_id": instance["instance_id"],
                    "role": instance["role"],
                    "method": method,
                    "repeat": repeat,
                    "seed": seed,
                    **sample_evaluation(
                        local, sampleset, float(exact["objective_energy"])
                    ),
                }
            )
    return rows


def write_instance(
    root: Path,
    output_directory: Path,
    frozen: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    local = frozen["record"]
    row = local["row"]
    identifier = instance_id(frozen["role"], row)
    bqm_path = output_directory / f"{identifier}.bqm.json"
    moves_path = output_directory / f"{identifier}.moves.csv"
    metadata_path = output_directory / f"{identifier}.metadata.json"
    bqm_serializable = local["bqm"].to_serializable(use_bytes=False)
    moves = move_rows(local)
    exact = exact_milp(local, config["exact_reference"])
    improvement = float(exact["improvement_from_warm"])
    if frozen["role"] == "confirmation_positive" and not improvement < -TOLERANCE:
        raise ValueError("Stage78 positive is not a certified multi-move improvement")
    if frozen["role"] == "confirmation_negative" and not math.isclose(
        improvement, 0.0, abs_tol=TOLERANCE
    ):
        raise ValueError(
            "Stage78 frozen negative has a certified multi-move improvement; stop without replacement"
        )
    if frozen["role"] == "calibration_diagnostic" and not improvement < -TOLERANCE:
        raise ValueError("Stage78 calibration diagnostic lacks a certified improvement")
    s75.write_json(bqm_path, bqm_serializable)
    s75.write_csv(moves_path, moves)
    metadata = {
        "schema_version": "1.0",
        "instance_id": identifier,
        "role": frozen["role"],
        "include_in_paid_run": bool(frozen["include_in_paid_run"]),
        "source_subproblem_index": int(row["subproblem_index"]),
        "target_id": str(row["target_id"]),
        "outer_fold": int(row["outer_fold"]),
        "k": int(row["k"]),
        "canonical_reward_quantile": float(row["reward_quantile"]),
        "reward_value": float(row["reward_value"]),
        "warm_subset": str(row["warm_subset"]),
        "warm_energy": float(row["warm_energy"]),
        "warm_deficit": int(row["warm_deficit"]),
        "quality_threshold": int(row["quality_threshold"]),
        "logical_variable_count": int(row["encoded_move_variable_count"]),
        "interaction_count": int(row["bqm_interaction_count"]),
        "ideal_zephyr_physical_qubit_count": int(
            row["ideal_zephyr_physical_qubit_count"]
        ),
        "ideal_zephyr_maximum_chain_length": int(
            row["ideal_zephyr_maximum_chain_length"]
        ),
        "best_single_move_energy_delta": float(
            row["best_single_move_energy_delta"]
        ),
        "hardware_resolvable_single_move_improvement": bool(
            row["hardware_resolvable_single_move_improvement"]
        ),
        "exact_reference": exact,
        "bqm": s75.descriptor(root, bqm_path),
        "moves": s75.descriptor(root, moves_path),
    }
    s75.write_json(metadata_path, metadata)
    manifest_row = {
        "instance_id": identifier,
        "role": frozen["role"],
        "include_in_paid_run": bool(frozen["include_in_paid_run"]),
        "target_id": row["target_id"],
        "outer_fold": int(row["outer_fold"]),
        "k": int(row["k"]),
        "canonical_reward_quantile": float(row["reward_quantile"]),
        "logical_variable_count": int(row["encoded_move_variable_count"]),
        "interaction_count": int(row["bqm_interaction_count"]),
        "warm_energy": float(row["warm_energy"]),
        "exact_energy": float(exact["objective_energy"]),
        "exact_improvement_from_warm": float(exact["improvement_from_warm"]),
        "best_single_move_energy_delta": float(
            row["best_single_move_energy_delta"]
        ),
        "bqm_path": bqm_path.relative_to(root).as_posix(),
        "bqm_sha256": s75.sha256(bqm_path),
        "moves_path": moves_path.relative_to(root).as_posix(),
        "moves_sha256": s75.sha256(moves_path),
        "metadata_path": metadata_path.relative_to(root).as_posix(),
        "metadata_sha256": s75.sha256(metadata_path),
    }
    instance = {
        **frozen,
        "instance_id": identifier,
        "metadata": metadata,
        "manifest_row": manifest_row,
    }
    return instance, classical_control_rows(instance, config, exact)


def report_text(result: dict[str, Any]) -> str:
    summary = result["instance_summary"]
    return f"""# Stage78 Advantage2 Reverse-Annealing PoC Freeze

## Purpose

Freeze a minimal, falsifiable hardware experiment from the passing Stage77 local-BQM gate. This stage performs no cloud query and no QPU sampling.

## Frozen Instances

- Canonical reward quantile: `{result['instance_freeze']['canonical_reward_quantile']}`.
- Paid-run primary set: `{summary['paid_run_instance_count']}` independent fixed-k BQMs.
- Hardware-resolvable positives: `{summary['hardware_resolvable_positive_count']}`.
- Cross-target hard negatives: `{summary['negative_control_count']}`.
- Sub-resolution calibration diagnostic: `{summary['calibration_diagnostic_count']}`; used for hardware-only tuning and excluded from confirmation endpoints.
- Maximum logical variables / interactions: `{summary['maximum_logical_variable_count']}` / `{summary['maximum_interaction_count']}`.

The sub-resolution PPARG diagnostic is used only for hardware calibration. Both hardware-resolvable PPARG positives and the three target-matched negatives remain untouched until confirmation. Reward quantiles are not treated as replicates.

## Local Evidence

- Every frozen BQM has an independently checkable SciPy MILP optimum and a warm all-zero state.
- Classical controls include cold and warm simulated annealing, warm tabu, and warm steepest descent.
- Wall-clock time is recorded nowhere as a scientific endpoint.

## External Stop Boundary

The next command that cannot be completed locally is the Leap preflight: it needs a D-Wave Leap account, API token, and access to an Advantage2 Zephyr QPU. Preflight queries the current working graph and creates physical embeddings but performs no QPU sampling. Calibration and confirmation additionally require two explicit paid-execution acknowledgements.

## Claim Boundary

Stage78 preregisters a physical-hardware proof of concept. It does not authorize a quantum-advantage, scaling, biological-generalization, or end-to-end speedup claim.
"""


def compute(config: dict[str, Any], root: Path) -> dict[str, Any]:
    for name in ("runner", "independent_auditor", "hardware_executor"):
        s75.verified(root, config["implementation"][name], f"Stage78 {name}")
    _, records = rebuild_local_records(config, root)
    canonical = canonical_records(
        records, float(config["instance_freeze"]["canonical_reward_quantile"])
    )
    frozen = freeze_instances(canonical, config)
    output_directory = root / config["outputs"]["instance_directory"]
    output_directory.mkdir(parents=True, exist_ok=True)
    instances: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for item in frozen:
        instance, rows = write_instance(root, output_directory, item, config)
        instances.append(instance)
        controls.extend(rows)
    manifest_rows = [instance["manifest_row"] for instance in instances]
    manifest_path = root / config["outputs"]["instance_manifest_csv"]
    controls_path = root / config["outputs"]["classical_controls_csv"]
    s75.write_csv(manifest_path, manifest_rows)
    s75.write_csv(controls_path, controls)

    roles = Counter(instance["role"] for instance in instances)
    paid = [instance for instance in instances if instance["include_in_paid_run"]]
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage78_advantage2_reverse_annealing_poc_frozen",
        "config": s75.descriptor(
            root, root / "configs/stage78_advantage2_reverse_annealing_poc.json"
        ),
        "instance_freeze": config["instance_freeze"],
        "instance_summary": {
            "frozen_instance_count": len(instances),
            "paid_run_instance_count": len(paid),
            "hardware_resolvable_positive_count": roles["confirmation_positive"],
            "negative_control_count": roles["confirmation_negative"],
            "calibration_diagnostic_count": roles["calibration_diagnostic"],
            "maximum_logical_variable_count": max(
                int(instance["manifest_row"]["logical_variable_count"])
                for instance in instances
            ),
            "maximum_interaction_count": max(
                int(instance["manifest_row"]["interaction_count"])
                for instance in instances
            ),
            "role_counts": dict(sorted(roles.items())),
        },
        "instances": [instance["metadata"] for instance in instances],
        "classical_control_summary": {
            "trial_row_count": len(controls),
            "all_guarded_nonworse": all(
                bool(row["warm_guard_nonworse"]) for row in controls
            ),
            "method_count": len({row["method"] for row in controls}),
        },
        "hardware_protocol": config["hardware_protocol"],
        "external_stop": {
            "reached": True,
            "reason": "D-Wave Leap credentials and a live Advantage2 Zephyr working graph are required for the next preflight step.",
            "cloud_queries_run": 0,
            "qpu_jobs_run": 0,
            "qpu_reads_run": 0,
        },
        "decision": {
            "ready_for_external_leap_preflight": True,
            "paid_qpu_execution_authorized": False,
            "quantum_advantage_claim_authorized": False,
            "quantum_scaling_claim_authorized": False,
        },
        "data_boundary": {
            "historical_development_targets_read": 4,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "cloud_solver_queries": 0,
            "quantum_hardware_jobs": 0,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": package_version("scipy"),
            "dimod": dimod.__version__,
            "dwave_samplers": package_version("dwave-samplers"),
        },
        "outputs": {
            "instance_manifest_csv": s75.descriptor(root, manifest_path),
            "classical_controls_csv": s75.descriptor(root, controls_path),
            "instance_files": [
                {
                    "instance_id": instance["instance_id"],
                    "bqm": instance["metadata"]["bqm"],
                    "moves": instance["metadata"]["moves"],
                    "metadata": s75.descriptor(
                        root,
                        output_directory / f"{instance['instance_id']}.metadata.json",
                    ),
                }
                for instance in instances
            ],
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    result_path = root / config["outputs"]["result_json"]
    report_path = root / config["outputs"]["report_md"]
    s75.write_json(result_path, result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text(result), encoding="ascii", newline="\n")
    result["outputs"]["report_md"] = s75.descriptor(root, report_path)
    s75.write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run(config_path: Path, root: Path, overwrite: bool = False) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = s75.read_json(config_path)
    result_path = root / config["outputs"]["result_json"]
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"Stage78 result already exists: {result_path}")
    return compute(config, root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage78_advantage2_reverse_annealing_poc.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
