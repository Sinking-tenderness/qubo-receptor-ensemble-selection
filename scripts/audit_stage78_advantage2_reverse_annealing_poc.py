"""Independently audit the frozen Stage78 Advantage2 PoC package."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import dimod
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

try:
    import scripts.run_stage75_explicit_variable_k_cqm as s75
    import scripts.run_stage77_quantum_hardware_interface_gate as s77
except ImportError:
    import run_stage75_explicit_variable_k_cqm as s75
    import run_stage77_quantum_hardware_interface_gate as s77


TOLERANCE = 1e-8


def stage77_records(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    stage77_path = s75.verified(
        root, config["inputs"]["stage77_config"], "Stage77 config"
    )
    stage77_config = s75.read_json(stage77_path)
    required = (
        "stage72_model_record",
        "stage74_workload_metrics",
        "stage74_cell_comparison",
        "stage74_solver_trials",
    )
    inputs = {
        name: s75.verified(root, stage77_config["inputs"][name], name)
        for name in required
    }
    cells = s77.source_cells(stage77_config, inputs)
    records, _ = s77.local_subproblems(cells, stage77_config)
    return [
        record
        for record in records
        if math.isclose(
            float(record["row"]["reward_quantile"]),
            float(config["instance_freeze"]["canonical_reward_quantile"]),
            abs_tol=1e-12,
        )
    ]


def independent_selection(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> list[tuple[str, dict[str, Any], bool]]:
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
    selected: list[tuple[str, dict[str, Any], bool]] = [
        ("confirmation_positive", positives[0], True),
        ("confirmation_positive", positives[1], True),
    ]
    for target in config["instance_freeze"]["negative_control_targets"]:
        candidates = [
            record
            for record in records
            if record["row"]["target_id"] == target
            and not bool(record["row"]["improving_single_move_available"])
        ]
        chosen = sorted(
            candidates,
            key=lambda record: (
                -int(record["row"]["encoded_move_variable_count"]),
                -int(record["row"]["k"]),
                int(record["row"]["outer_fold"]),
                int(record["row"]["subproblem_index"]),
            ),
        )[0]
        selected.append(("confirmation_negative", chosen, True))
    diagnostic = sorted(
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
    )[0]
    selected.append(("calibration_diagnostic", diagnostic, True))
    return selected


def identifier(role: str, row: dict[str, Any]) -> str:
    return (
        f"{str(row['target_id']).lower()}_of{int(row['outer_fold'])}_"
        f"k{int(row['k'])}_{role}"
    )


def conflicts(moves: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [
        (left, right)
        for left in range(len(moves))
        for right in range(left + 1, len(moves))
        if moves[left]["removed_index"] == moves[right]["removed_index"]
        or moves[left]["added_index"] == moves[right]["added_index"]
    ]


def independent_exact_energy(local: dict[str, Any], config: dict[str, Any]) -> float:
    bqm = local["bqm"]
    names = list(bqm.variables)
    positions = {name: index for index, name in enumerate(names)}
    products = [
        (positions[left], positions[right], float(value))
        for (left, right), value in bqm.quadratic.items()
    ]
    n = len(names)
    total = n + len(products)
    objective = np.zeros(total)
    for name, value in bqm.linear.items():
        objective[positions[name]] = float(value)
    for offset, (_, _, value) in enumerate(products):
        objective[n + offset] = value
    row_index: list[int] = []
    column_index: list[int] = []
    coefficients: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def constraint(terms: Iterable[tuple[int, float]], lb: float, ub: float) -> None:
        row = len(lower)
        for column, value in terms:
            row_index.append(row)
            column_index.append(column)
            coefficients.append(float(value))
        lower.append(lb)
        upper.append(ub)

    for offset, (left, right, _) in enumerate(products):
        product = n + offset
        constraint(((product, 1), (left, -1)), -np.inf, 0)
        constraint(((product, 1), (right, -1)), -np.inf, 0)
        constraint(((left, 1), (right, 1), (product, -1)), -np.inf, 1)
    for left, right in conflicts(local["moves"]):
        constraint(((left, 1), (right, 1)), -np.inf, 1)
    matrix = coo_matrix(
        (coefficients, (row_index, column_index)),
        shape=(len(lower), total),
    ).tocsr()
    result = milp(
        objective,
        integrality=np.ones(total, dtype=int),
        bounds=Bounds(np.zeros(total), np.ones(total)),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={
            "presolve": True,
            "time_limit": float(config["exact_reference"]["time_limit_seconds"]),
            "mip_rel_gap": 0.0,
        },
    )
    if not result.success or int(result.status) != 0:
        raise ValueError(f"Stage78 independent MILP failed: {result.message}")
    return float(result.fun) + float(bqm.offset)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = s75.read_json(config_path.resolve())
    for name in ("runner", "independent_auditor", "hardware_executor"):
        s75.verified(root, config["implementation"][name], f"Stage78 {name}")
    for name, descriptor in config["inputs"].items():
        s75.verified(root, descriptor, name)
    result_path = root / config["outputs"]["result_json"]
    result = s75.read_json(result_path)
    if result["status"] != "stage78_advantage2_reverse_annealing_poc_frozen":
        raise ValueError("Stage78 source result is not frozen")
    records = stage77_records(config, root)
    selected = independent_selection(records, config)
    manifest_path = s75.verified(
        root, result["outputs"]["instance_manifest_csv"], "Stage78 manifest"
    )
    manifest = read_csv(manifest_path)
    manifest_by_id = {row["instance_id"]: row for row in manifest}
    metadata_by_id = {row["instance_id"]: row for row in result["instances"]}
    audited = 0
    exact_match = 0
    bqm_match = 0
    move_match = 0
    for role, local, paid in selected:
        row = local["row"]
        instance_id = identifier(role, row)
        if instance_id not in manifest_by_id or instance_id not in metadata_by_id:
            raise ValueError(f"Stage78 missing independently selected {instance_id}")
        manifest_row = manifest_by_id[instance_id]
        metadata = metadata_by_id[instance_id]
        if s75.truth(manifest_row["include_in_paid_run"]) != paid:
            raise ValueError(f"Stage78 paid-run flag differs for {instance_id}")
        bqm_path = s75.verified(root, metadata["bqm"], f"{instance_id} BQM")
        serialized = s75.read_json(bqm_path)
        rebuilt = local["bqm"].to_serializable(use_bytes=False)
        if s75.canonical_sha256(serialized) != s75.canonical_sha256(rebuilt):
            raise ValueError(f"Stage78 serialized BQM differs for {instance_id}")
        bqm_match += 1
        moves_path = s75.verified(root, metadata["moves"], f"{instance_id} moves")
        moves = read_csv(moves_path)
        if len(moves) != len(local["moves"]):
            raise ValueError(f"Stage78 move count differs for {instance_id}")
        for move_row, move in zip(moves, local["moves"]):
            if (
                int(move_row["removed_index"]) != int(move["removed_index"])
                or int(move_row["added_index"]) != int(move["added_index"])
                or int(move_row["deficit_delta"]) != int(move["deficit_delta"])
                or not math.isclose(
                    float(move_row["single_move_energy_delta"]),
                    float(move["energy_delta"]),
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"Stage78 move map differs for {instance_id}")
        move_match += 1
        exact_energy = independent_exact_energy(local, config)
        if not math.isclose(
            exact_energy,
            float(metadata["exact_reference"]["objective_energy"]),
            abs_tol=TOLERANCE,
        ):
            raise ValueError(f"Stage78 exact objective differs for {instance_id}")
        exact_improvement = exact_energy - float(metadata["warm_energy"])
        if role in {"confirmation_positive", "calibration_diagnostic"}:
            if not exact_improvement < -TOLERANCE:
                raise ValueError(
                    f"Stage78 independently audited {role} is not improvable: {instance_id}"
                )
        elif role == "confirmation_negative":
            if not math.isclose(exact_improvement, 0.0, abs_tol=TOLERANCE):
                raise ValueError(
                    f"Stage78 independently audited negative is improvable: {instance_id}"
                )
        else:
            raise ValueError(f"Stage78 has an unrecognized frozen role: {role}")
        exact_match += 1
        audited += 1
    if len(manifest) != audited or len(metadata_by_id) != audited:
        raise ValueError("Stage78 has unregistered extra instances")
    controls_path = s75.verified(
        root, result["outputs"]["classical_controls_csv"], "classical controls"
    )
    controls = read_csv(controls_path)
    expected_control_rows = audited * (3 * int(config["classical_controls"]["repeats"]) + 1)
    if len(controls) != expected_control_rows:
        raise ValueError("Stage78 classical-control row count differs")
    if not all(s75.truth(row["warm_guard_nonworse"]) for row in controls):
        raise ValueError("Stage78 classical warm guard worsened a solution")
    if result["external_stop"]["qpu_jobs_run"] != 0:
        raise ValueError("Stage78 freeze unexpectedly contacted a QPU")
    required_paid = int(config["instance_freeze"]["required_paid_run_instance_count"])
    actual_paid = sum(paid for _, _, paid in selected)
    if actual_paid != required_paid:
        raise ValueError("Stage78 paid-run instance count differs from preregistration")
    audit_record = {
        "schema_version": "1.0",
        "status": "stage78_advantage2_reverse_annealing_poc_independent_audit_ok",
        "source_result": s75.descriptor(root, result_path),
        "canonical_stage77_local_records_rebuilt": len(records),
        "instances_independently_selected": audited,
        "serialized_bqms_independently_matched": bqm_match,
        "move_maps_independently_matched": move_match,
        "exact_milp_objectives_independently_matched": exact_match,
        "classical_control_rows_checked": len(controls),
        "paid_run_instance_count": actual_paid,
        "cloud_queries_observed": 0,
        "qpu_jobs_observed": 0,
        "ready_for_external_leap_preflight": True,
        "paid_qpu_execution_authorized": False,
        "quantum_advantage_claim_authorized": False,
    }
    output = root / config["outputs"]["audit_json"]
    s75.write_json(output, audit_record)
    print(json.dumps(audit_record, indent=2, sort_keys=True))
    return audit_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage78_advantage2_reverse_annealing_poc.json"),
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    audit(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
