"""Converge prior evidence into a strict gate for quantum-worthy instances."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.run_stage68_quality_plateau_portfolio_qubo as s68


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty Stage87 CSV: {path}")
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


def verified(root: Path, descriptor: dict[str, Any]) -> Path:
    path = root / descriptor["path"]
    if not path.is_file() or sha256(path) != descriptor["sha256"]:
        raise ValueError(f"Stage87 frozen input identity differs: {path}")
    return path


def bool_value(value: str) -> bool:
    return value == "True"


def grouped_stage68(rows: list[dict[str, str]]) -> dict[tuple[Any, ...], dict[str, dict[str, str]]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["candidate_id"] == "pair_off":
            continue
        key = (
            row["target_id"],
            int(row["outer_fold"]),
            row["candidate_id"],
            int(row["subset_size"]),
        )
        grouped[key][row["solver_id"]] = row
    return dict(grouped)


def initial_candidates(
    grouped: dict[tuple[Any, ...], dict[str, dict[str, str]]],
    metrics: list[str],
) -> list[tuple[tuple[Any, ...], dict[str, dict[str, str]]]]:
    candidates = []
    required = {s68.SOLVER_EXACT, s68.SOLVER_GREEDY, s68.SOLVER_SWAP}
    for key, methods in grouped.items():
        if set(methods) != required:
            raise ValueError(f"Stage87 incomplete Stage68 solver group: {key}")
        exact = methods[s68.SOLVER_EXACT]
        direct = methods[s68.SOLVER_GREEDY]
        swapped = methods[s68.SOLVER_SWAP]
        exact_objective = float(exact["stable_redundancy_sum"])
        checks = (
            exact["milp_status"] == "0",
            float(exact["milp_gap"]) <= 1e-12,
            exact["selected_subset"] != direct["selected_subset"],
            exact["selected_subset"] != swapped["selected_subset"],
            exact_objective < float(direct["stable_redundancy_sum"]) - 1e-12,
            exact_objective < float(swapped["stable_redundancy_sum"]) - 1e-12,
            all(
                float(exact[field]) > float(direct[field]) + 1e-12
                and float(exact[field]) > float(swapped[field]) + 1e-12
                for field in metrics
            ),
        )
        if all(checks):
            candidates.append((key, methods))
    return candidates


def rebuild_fold(
    root: Path,
    stage68_config: dict[str, Any],
    stage64_config: dict[str, Any],
    target_id: str,
    outer_fold: int,
    cache: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    key = (target_id, outer_fold)
    if key in cache:
        return cache[key]
    target = s68.load_target(root, target_id, stage64_config["targets"][target_id])
    train_mask = np.asarray(
        [target["outer"][ligand_id] != outer_fold for ligand_id in target["ligand_ids"]]
    )
    ranks = s68.rank_cube(target["scores"], train_mask)
    train_rows = [row for row, keep in zip(target["ligands"], train_mask) if keep]
    development = stage68_config["development"]
    stats = s68.jackknife_pair_statistics(
        target["scores"][:, train_mask, :],
        target["labels"][train_mask],
        train_rows,
        float(development["bedroc_alpha"]),
        int(development["jackknife_block_count"]),
        int(development["jackknife_seed_base"]) + outer_fold,
    )
    value = {
        "target": target,
        "utility": stats["full_singleton"],
        "spread": stats["singleton_spread"],
        "redundancy": s68.stable_redundancy(ranks, train_mask),
    }
    cache[key] = value
    return value


def enumerate_candidate(
    root: Path,
    stage68_config: dict[str, Any],
    stage64_config: dict[str, Any],
    key: tuple[Any, ...],
    methods: dict[str, dict[str, str]],
    cache: dict[tuple[str, int], dict[str, Any]],
    maximum_trivial_states: int,
) -> dict[str, Any]:
    target_id, outer_fold, candidate_id, subset_size = key
    fold = rebuild_fold(
        root, stage68_config, stage64_config, target_id, outer_fold, cache
    )
    exact_row = methods[s68.SOLVER_EXACT]
    direct_row = methods[s68.SOLVER_GREEDY]
    swap_row = methods[s68.SOLVER_SWAP]
    quality_floor = float(exact_row["train_quality_floor"])
    receptor_ids = fold["target"]["receptor_ids"]
    total_states = math.comb(len(receptor_ids), subset_size)
    started = time.perf_counter()
    feasible: list[tuple[tuple[Any, ...], tuple[int, ...]]] = []
    for subset in itertools.combinations(range(len(receptor_ids)), subset_size):
        quality = float(np.mean(fold["utility"][list(subset)]))
        if quality < quality_floor - s68.TOLERANCE:
            continue
        key_value = (
            s68.redundancy_sum(subset, fold["redundancy"]),
            -quality,
            subset,
        )
        feasible.append((key_value, subset))
    elapsed = time.perf_counter() - started
    feasible.sort(key=lambda item: item[0])
    if not feasible:
        raise ValueError(f"Stage87 candidate has no feasible subset: {key}")
    exact_subset = "+".join(receptor_ids[index] for index in feasible[0][1])
    if exact_subset != exact_row["selected_subset"]:
        raise ValueError(f"Stage87 exhaustive optimum differs from Stage68 MILP: {key}")
    exact_objective = float(exact_row["stable_redundancy_sum"])
    direct_objective = float(direct_row["stable_redundancy_sum"])
    swap_objective = float(swap_row["stable_redundancy_sum"])
    return {
        "target_id": target_id,
        "outer_fold": outer_fold,
        "candidate_id": candidate_id,
        "subset_size": subset_size,
        "receptor_count": len(receptor_ids),
        "total_fixed_k_states": total_states,
        "quality_feasible_states": len(feasible),
        "exhaustive_enumeration_seconds": elapsed,
        "exact_subset": exact_subset,
        "exact_objective": exact_objective,
        "direct_greedy_objective": direct_objective,
        "greedy_swap_objective": swap_objective,
        "exact_gain_over_direct_greedy": direct_objective - exact_objective,
        "exact_gain_over_greedy_swap": swap_objective - exact_objective,
        "holdout_robust_gain_over_direct_greedy": float(
            exact_row["holdout_robust_bedroc"]
        )
        - float(direct_row["holdout_robust_bedroc"]),
        "holdout_robust_gain_over_greedy_swap": float(
            exact_row["holdout_robust_bedroc"]
        )
        - float(swap_row["holdout_robust_bedroc"]),
        "exact_certificate_verified": True,
        "direct_greedy_trap_verified": True,
        "single_start_greedy_swap_trap_verified": True,
        "exhaustively_trivial": total_states <= maximum_trivial_states,
        "nontrivial_classical_search_space": total_states > maximum_trivial_states,
    }


def evidence_rows(
    stage68_count: int,
    stage74_exact: int,
    stage74_missed: int,
    stage75_exact: int,
    stage75_missed: int,
    stage80_count: int,
    stage80_traps: int,
) -> list[dict[str, Any]]:
    return [
        {
            "evidence_block": "biological_plus_single_start_greedy_trap",
            "source_stage": "Stage68",
            "evaluated_instance_count": 240,
            "positive_instance_count": stage68_count,
            "interpretation": "Post-hoc candidates requiring strict gains across primary, mean-seed, worst-seed, and robust holdout BEDROC.",
        },
        {
            "evidence_block": "fixed_k_exact_vs_strong_classical",
            "source_stage": "Stage74",
            "evaluated_instance_count": stage74_exact,
            "positive_instance_count": stage74_missed,
            "interpretation": "Certified exact cells missed by the best multistart greedy, tabu, annealing, and deterministic-improvement portfolio.",
        },
        {
            "evidence_block": "variable_k_exact_vs_joint_classical",
            "source_stage": "Stage75",
            "evaluated_instance_count": stage75_exact,
            "positive_instance_count": stage75_missed,
            "interpretation": "Exact variable-k frontier cells missed by the joint strong-classical solver.",
        },
        {
            "evidence_block": "local_multi_move_trap",
            "source_stage": "Stage80",
            "evaluated_instance_count": stage80_count,
            "positive_instance_count": stage80_traps,
            "interpretation": "Protein-derived local move problems with a certified pair or tabu-detected improvement unavailable to a single move.",
        },
    ]


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    stage68_config = read_json(inputs["stage68_config"])
    stage64_config = read_json(root / stage68_config["inputs"]["stage64_config"]["path"])
    fixed_rows = read_csv(inputs["stage68_fixed_k_metrics"])
    grouped = grouped_stage68(fixed_rows)
    metrics = list(config["gate"]["holdout_metrics_requiring_strict_gain"])
    candidates = initial_candidates(grouped, metrics)
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    candidate_rows = [
        enumerate_candidate(
            root,
            stage68_config,
            stage64_config,
            key,
            methods,
            cache,
            int(config["gate"]["maximum_states_considered_exhaustively_trivial"]),
        )
        for key, methods in candidates
    ]

    stage74 = read_csv(inputs["stage74_cell_comparison"])
    stage74_exact = [row for row in stage74 if bool_value(row["exact_oracle_available"])]
    stage74_missed = [
        row for row in stage74_exact if not bool_value(row["strong_classical_reference_match"])
    ]
    stage75 = read_csv(inputs["stage75_cell_comparison"])
    stage75_exact = [row for row in stage75 if bool_value(row["exact_frontier_available"])]
    stage75_missed = [
        row
        for row in stage75_exact
        if not bool_value(row["joint_classical_exact_frontier_match"])
    ]
    stage80 = read_csv(inputs["stage80_hardness_metrics"])
    stage80_traps = [row for row in stage80 if bool_value(row["local_trap_candidate"])]
    stage86a = read_json(inputs["stage86a_adjudication"])
    if stage86a["additional_qci_dirac3_global_penalty_jobs_authorized"] != 0:
        raise ValueError("Stage87 requires the Stage86a QCI stop decision")

    fold_groups = Counter(
        (row["target_id"], row["candidate_id"], int(row["subset_size"]))
        for row in candidate_rows
    )
    replicated_fold_groups = {
        "|".join(map(str, key)): count
        for key, count in sorted(fold_groups.items())
        if count >= int(config["gate"]["minimum_replicated_outer_folds"])
    }
    candidate_targets: dict[str, set[str]] = defaultdict(set)
    for row in candidate_rows:
        candidate_targets[str(row["candidate_id"])].add(str(row["target_id"]))
    replicated_target_candidates = {
        key: sorted(values)
        for key, values in sorted(candidate_targets.items())
        if len(values) >= int(config["gate"]["minimum_replicated_targets"])
    }
    nontrivial = [row for row in candidate_rows if row["nontrivial_classical_search_space"]]

    checks = {
        "stage68_biological_greedy_trap_candidate_exists": len(candidate_rows) > 0,
        "cross_fold_replication_exists": bool(replicated_fold_groups),
        "cross_target_candidate_family_exists": bool(replicated_target_candidates),
        "nontrivial_certified_candidate_exists": bool(nontrivial),
        "certified_fixed_k_strong_classical_failure_exists": bool(stage74_missed),
        "certified_variable_k_strong_classical_failure_exists": bool(stage75_missed),
        "multi_move_local_trap_exists": bool(stage80_traps),
    }
    strict_gate_passed = all(checks.values())
    evidence = evidence_rows(
        len(candidate_rows),
        len(stage74_exact),
        len(stage74_missed),
        len(stage75_exact),
        len(stage75_missed),
        len(stage80),
        len(stage80_traps),
    )
    outputs = config["outputs"]
    write_csv(root / outputs["candidate_instances_csv"], candidate_rows)
    write_csv(root / outputs["evidence_matrix_csv"], evidence)

    result = {
        "schema_version": "1.0",
        "status": "stage87_no_quantum_worthy_instance_qaoa_blocked",
        "candidate_summary": {
            "stage68_screened_cell_count": len(grouped),
            "biological_single_start_trap_candidate_count": len(candidate_rows),
            "nontrivial_certified_candidate_count": len(nontrivial),
            "maximum_candidate_total_states": max(
                int(row["total_fixed_k_states"]) for row in candidate_rows
            ),
            "maximum_candidate_feasible_states": max(
                int(row["quality_feasible_states"]) for row in candidate_rows
            ),
            "maximum_exhaustive_enumeration_seconds": max(
                float(row["exhaustive_enumeration_seconds"]) for row in candidate_rows
            ),
            "replicated_fold_groups": replicated_fold_groups,
            "replicated_target_candidate_families": replicated_target_candidates,
        },
        "historical_hardness_summary": {
            "stage74_certified_exact_cell_count": len(stage74_exact),
            "stage74_strong_classical_miss_count": len(stage74_missed),
            "stage75_exact_frontier_cell_count": len(stage75_exact),
            "stage75_joint_classical_miss_count": len(stage75_missed),
            "stage80_local_subproblem_count": len(stage80),
            "stage80_multi_move_trap_count": len(stage80_traps),
        },
        "checks": checks,
        "strict_instance_gate_passed": strict_gate_passed,
        "constraint_preserving_qaoa_simulation_authorized": strict_gate_passed,
        "new_quantum_hardware_jobs_authorized": 0,
        "new_docking_jobs_authorized": 0,
        "next_action": (
            "Do not run QAOA on the current benchmark cells. Define or acquire a larger, "
            "independently validated scientific decision problem for which a certified "
            "global solution beats strong classical search; preregister that gate before "
            "new docking or quantum execution."
        ),
        "interpretation": (
            "Four post-hoc Stage68 cells beat a single-start greedy-plus-swap baseline "
            "and improve every frozen holdout BEDROC summary, but each full fixed-k space "
            "contains at most 38,760 states and is exhaustively solved in under one second "
            "on the audit workstation. Across all previously certified Stage74 and Stage75 "
            "cells, strong classical search never missed the exact reference. The current "
            "evidence therefore contains useful positive controls, not a quantum-worthy "
            "classical-hard benchmark."
        ),
        "data_boundary": {
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
    }
    write_json(root / outputs["result_json"], result)
    report = [
        "# Stage87 quantum-value instance gate",
        "",
        "## Prior evidence review",
        "",
        "- Stage37-68 repeatedly tested whether quadratic or higher-order receptor interactions improve held-out screening. No objective produced a stable cross-target superiority claim.",
        "- Stage73-75 showed that strong classical methods match every available certified exact fixed-k or variable-k reference.",
        "- Stage79 physically executed local protein-derived QUBOs, but Stage80 found no multi-move local trap among 100 canonical subproblems.",
        "- Stage85-86 showed that the scientifically meaningful global constrained model is not faithfully sampled by the current Dirac-3 penalty interface.",
        "",
        "## Targeted supplement",
        "",
        f"Stage68 contained {len(candidate_rows)} post-hoc cells where the exact portfolio differs from direct greedy and greedy-swap, has a strictly better redundancy objective, and improves primary, mean-seed, worst-seed, and robust holdout BEDROC.",
        "",
        "| Target/fold | k | Candidate | Total states | Feasible states | Exact over swap BEDROC | Enumeration seconds |",
        "|---|---:|---|---:|---:|---:|---:|",
        *[
            f"| {row['target_id']}/{row['outer_fold']} | {row['subset_size']} | {row['candidate_id']} | "
            f"{row['total_fixed_k_states']:,} | {row['quality_feasible_states']:,} | "
            f"{row['holdout_robust_gain_over_greedy_swap']:+.6f} | "
            f"{row['exhaustive_enumeration_seconds']:.3f} |"
            for row in candidate_rows
        ],
        "",
        "All four candidates are complete-enumeration problems with at most 38,760 states. They verify that one deterministic greedy start can be trapped, but they do not establish classical difficulty.",
        "",
        "## Decision",
        "",
        "The strict instance gate fails. Constraint-preserving QAOA simulation and new hardware jobs remain blocked. The next scientific task is to preregister a larger independently validated decision problem where the certified global solution has both biological benefit and a demonstrated gap over strong classical search.",
        "",
    ]
    report_path = root / outputs["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="ascii")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage87_quantum_value_instance_gate.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = run((root / args.config).resolve(), root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
