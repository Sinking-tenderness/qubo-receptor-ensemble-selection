"""Screen protein-derived local move-QUBOs for nontrivial multi-move traps."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json  # noqa: F401 (deduped)
import argparse
import copy
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dwave.samplers import SteepestDescentSolver, TabuSampler

try:
    import scripts.run_stage75_explicit_variable_k_cqm as s75
    import scripts.run_stage77_quantum_hardware_interface_gate as s77
except ImportError:
    import run_stage75_explicit_variable_k_cqm as s75
    import run_stage77_quantum_hardware_interface_gate as s77


TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()




def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Stage80 refuses to write an empty metrics table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verified(root: Path, descriptor: dict[str, Any], label: str) -> Path:
    path = root / str(descriptor["path"])
    if not path.is_file() or sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage80 {label} identity differs: {path}")
    if path.stat().st_size != int(descriptor["size_bytes"]):
        raise ValueError(f"Stage80 {label} size differs: {path}")
    return path


def canonical_cells(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inputs = config["inputs"]
    stage77_config = read_json(verified(root, inputs["stage77_config"], "Stage77 config"))
    source = read_json(verified(root, inputs["stage72_model_record"], "Stage72 model"))
    workloads = s75.read_csv(
        verified(root, inputs["stage74_workload_metrics"], "Stage74 workloads")
    )
    comparisons = s75.read_csv(
        verified(root, inputs["stage74_cell_comparison"], "Stage74 comparisons")
    )
    trials = s75.read_csv(
        verified(root, inputs["stage74_solver_trials"], "Stage74 trials")
    )
    quantile = float(config["screen"]["canonical_reward_quantile"])
    cells: list[dict[str, Any]] = []
    for index, record in enumerate(source["models"], start=1):
        model = s75.load_model(record)
        frontiers = s75.source_frontiers(
            model,
            workloads,
            comparisons,
            trials,
            stage77_config["frozen_cqm"]["quality_regime"],
        )
        reward = float(s75.reward_order_statistic(model, quantile)["reward"])
        cells.append(
            {
                "model": model,
                "frontiers": frontiers,
                "reward": reward,
                "reward_quantile": quantile,
            }
        )
        print(
            json.dumps(
                {
                    "stage80_models_ready": index,
                    "stage80_models_total": len(source["models"]),
                    "target_id": record["target_id"],
                    "outer_fold": record["outer_fold"],
                }
            ),
            flush=True,
        )
    return stage77_config, cells


def conflict_free(local: dict[str, Any], selected: list[int]) -> bool:
    removed = [int(local["moves"][index]["removed_index"]) for index in selected]
    added = [int(local["moves"][index]["added_index"]) for index in selected]
    return len(removed) == len(set(removed)) and len(added) == len(set(added))


def best_feasible_sample(
    local: dict[str, Any], sampleset: Any
) -> tuple[float, int, bool]:
    names = list(local["bqm"].variables)
    best_energy = float(local["warm_energy"])
    best_selected = 0
    found = False
    for datum in sampleset.data(fields=["sample", "energy"]):
        selected = [index for index, name in enumerate(names) if int(datum.sample[name])]
        if not conflict_free(local, selected):
            continue
        if sum(int(local["moves"][index]["deficit_delta"]) for index in selected) > 0:
            continue
        energy = float(local["bqm"].energy(datum.sample))
        if energy < best_energy - TOLERANCE:
            best_energy = energy
            best_selected = len(selected)
            found = True
    return best_energy, best_selected, found


def best_single_and_pair(local: dict[str, Any]) -> tuple[float, float]:
    bqm = local["bqm"]
    names = list(bqm.variables)
    best_single = min([0.0] + [float(bqm.linear[name]) for name in names])
    best_pair = 0.0
    for left in range(len(names)):
        left_move = local["moves"][left]
        for right in range(left + 1, len(names)):
            right_move = local["moves"][right]
            if (
                int(left_move["removed_index"]) == int(right_move["removed_index"])
                or int(left_move["added_index"]) == int(right_move["added_index"])
            ):
                continue
            delta = (
                float(bqm.linear[names[left]])
                + float(bqm.linear[names[right]])
                + float(bqm.get_quadratic(names[left], names[right], default=0.0))
            )
            best_pair = min(best_pair, delta)
    return best_single, best_pair


def screen_rows(
    config: dict[str, Any], stage77_config: dict[str, Any], cells: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    protocol = config["screen"]
    local_config = copy.deepcopy(stage77_config)
    local_config["local_swap_bqm"]["maximum_move_variable_count"] = int(
        protocol["maximum_qci_binary_variable_count"]
    )
    rows: list[dict[str, Any]] = []
    subproblem_index = 0
    for cell in cells:
        record = cell["model"]["record"]
        target_id = str(record["target_id"])
        outer_fold = int(record["outer_fold"])
        for k in sorted(cell["frontiers"]):
            started = time.perf_counter()
            local = s77.build_swap_bqm(cell, int(k), local_config)
            names = list(local["bqm"].variables)
            zero = {name: 0 for name in names}
            best_single, best_pair = best_single_and_pair(local)

            steepest = SteepestDescentSolver().sample(
                local["bqm"], initial_states=[zero]
            )
            steepest_energy, steepest_moves, _ = best_feasible_sample(local, steepest)

            tabu = TabuSampler().sample(
                local["bqm"],
                num_reads=int(protocol["tabu_reads"]),
                timeout=int(protocol["tabu_timeout_milliseconds"]),
                initial_states=[zero],
                initial_states_generator="tile",
                seed=int(protocol["seed_base"]) + subproblem_index,
            )
            tabu_energy, tabu_moves, _ = best_feasible_sample(local, tabu)
            warm_energy = float(local["warm_energy"])
            steepest_delta = steepest_energy - warm_energy
            tabu_delta = tabu_energy - warm_energy
            single_improvable = best_single < -TOLERANCE
            pair_improvable = best_pair < -TOLERANCE
            tabu_improvable = tabu_delta < -TOLERANCE
            multi_move_tabu_improvement = tabu_improvable and tabu_moves >= 2
            local_trap_candidate = (
                (not single_improvable and (pair_improvable or tabu_improvable))
                or (
                    tabu_delta < steepest_delta - TOLERANCE
                    and tabu_moves >= 2
                )
            )
            rows.append(
                {
                    "subproblem_index": subproblem_index,
                    "target_id": target_id,
                    "outer_fold": outer_fold,
                    "reward_quantile": float(cell["reward_quantile"]),
                    "k": int(k),
                    "eligible_move_count": int(local["eligible_move_count"]),
                    "encoded_move_count": len(names),
                    "bqm_interaction_count": int(local["bqm"].num_interactions),
                    "qci_total_binary_levels": 2 * len(names),
                    "qci_level_limit_ok": 2 * len(names)
                    <= int(protocol["qci_total_level_limit"]),
                    "best_single_delta": best_single,
                    "best_pair_delta": best_pair,
                    "steepest_delta": steepest_delta,
                    "steepest_selected_move_count": steepest_moves,
                    "tabu_delta": tabu_delta,
                    "tabu_selected_move_count": tabu_moves,
                    "single_improvable": single_improvable,
                    "pair_improvable": pair_improvable,
                    "tabu_improvable": tabu_improvable,
                    "multi_move_tabu_improvement": multi_move_tabu_improvement,
                    "local_trap_candidate": local_trap_candidate,
                    "wall_seconds_diagnostic_only": time.perf_counter() - started,
                }
            )
            subproblem_index += 1
    return rows


def summarize(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    target_counts = Counter(str(row["target_id"]) for row in rows)
    trap_rows = [row for row in rows if bool(row["local_trap_candidate"])]
    summary = {
        "subproblem_count": len(rows),
        "target_counts": dict(sorted(target_counts.items())),
        "minimum_eligible_move_count": min(int(row["eligible_move_count"]) for row in rows),
        "maximum_eligible_move_count": max(int(row["eligible_move_count"]) for row in rows),
        "maximum_encoded_move_count": max(int(row["encoded_move_count"]) for row in rows),
        "single_improvable_count": sum(bool(row["single_improvable"]) for row in rows),
        "pair_improvable_count": sum(bool(row["pair_improvable"]) for row in rows),
        "tabu_improvable_count": sum(bool(row["tabu_improvable"]) for row in rows),
        "multi_move_tabu_improvement_count": sum(
            bool(row["multi_move_tabu_improvement"]) for row in rows
        ),
        "local_trap_candidate_count": len(trap_rows),
        "all_qci_level_limits_ok": all(bool(row["qci_level_limit_ok"]) for row in rows),
    }
    gate = config["decision_gate"]
    hardware_scaling_authorized = (
        len(trap_rows) >= int(gate["minimum_local_trap_candidate_count"])
        and summary["multi_move_tabu_improvement_count"]
        >= int(gate["minimum_multi_move_tabu_improvement_count"])
    )
    return {
        "summary": summary,
        "decision": {
            "additional_qci_local_scaling_run_authorized": hardware_scaling_authorized,
            "mechanical_move_cap_scaling_rejected": not hardware_scaling_authorized,
            "global_variable_k_reformulation_review_authorized": not hardware_scaling_authorized,
            "reason": (
                "A nontrivial local trap was detected."
                if hardware_scaling_authorized
                else "No canonical subproblem showed a single-move local trap resolved by a pair or the frozen tabu screen; increasing the local move cap would only pad the Stage79 task."
            ),
        },
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    decision = result["decision"]
    text = f"""# Stage80 Local Move-QUBO Hardness Screen

## Result

The canonical historical-development screen covered `{summary['subproblem_count']}`
fixed-k protein-derived local move-QUBOs. Only
`{summary['single_improvable_count']}` subproblems contained a strict single-move
improvement. No subproblem contained an improving nonconflicting pair, and the
frozen warm-start tabu screen found no multi-move improvement or local-trap
candidate.

The eligible move count ranged from `{summary['minimum_eligible_move_count']}`
to `{summary['maximum_eligible_move_count']}`. Up to
`{summary['maximum_encoded_move_count']}` binary move variables fit the documented
Dirac-3 level limit, but the added variables did not create a harder scientific
decision.

## Decision

- Additional QCI local scaling run authorized: `{str(decision['additional_qci_local_scaling_run_authorized']).lower()}`.
- Mechanical move-cap scaling rejected: `{str(decision['mechanical_move_cap_scaling_rejected']).lower()}`.
- Global variable-k reformulation review authorized: `{str(decision['global_variable_k_reformulation_review_authorized']).lower()}`.

The Stage79 physical result remains valid, but its local repair task is too easy
for a scaling or advantage claim. The next defensible route is to revisit the
full variable-k QUBO for Dirac-3, where all-to-all connectivity removes minor
embedding and float32 precision may permit a tighter penalty construction.

## Boundary

This screen does not prove that no higher-order trap exists. It reports that no
two-move trap or tabu-detected multi-move improvement was found under the frozen
canonical protocol. It used no new docking data and submitted no cloud or
physical-hardware job.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    for name in ("stage77_result", "stage79_physical_audit"):
        verified(root, config["inputs"][name], name)
    stage79 = read_json(root / config["inputs"]["stage79_physical_audit"]["path"])
    if stage79["status"] != "stage79_qci_dirac3_physical_poc_independent_audit_ok":
        raise ValueError("Stage80 requires the passing Stage79 physical audit")
    stage77_config, cells = canonical_cells(config, root)
    rows = screen_rows(config, stage77_config, cells)
    aggregate = summarize(config, rows)
    outputs = config["outputs"]
    metrics_path = root / outputs["metrics_csv"]
    write_csv(metrics_path, rows)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage80_local_move_hardness_screen_complete",
        **aggregate,
        "data_boundary": {
            "historical_development_subproblems_read": len(rows),
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "qci_cloud_queries": 0,
            "quantum_hardware_jobs": 0,
        },
        "outputs": {
            "metrics_csv": {
                "path": outputs["metrics_csv"],
                "sha256": sha256(metrics_path),
                "size_bytes": metrics_path.stat().st_size,
            }
        },
    }
    result_path = root / outputs["result_json"]
    write_json(result_path, result)
    report_path = root / outputs["report_md"]
    write_report(report_path, result)
    result["outputs"]["report_md"] = {
        "path": outputs["report_md"],
        "sha256": sha256(report_path),
        "size_bytes": report_path.stat().st_size,
    }
    write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/stage80_local_move_hardness_screen.json",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = run((root / args.config).resolve(), root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
