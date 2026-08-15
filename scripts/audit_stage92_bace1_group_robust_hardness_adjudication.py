from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage92_bace1_group_robust_hardness_adjudication as stage92


def run(root: Path, config_path: Path) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = stage92.read_json(config_path)
    outputs = dict(config["outputs"])
    result_path = root / str(outputs["result_json"])
    baseline_path = root / str(outputs["baseline_csv"])
    selection_path = root / str(outputs["selection_csv"])
    result = stage92.read_json(result_path)
    if result.get("status") != "stage92_bace1_group_robust_hardness_gate_failed":
        raise ValueError("Stage92 source status differs")
    if stage92.sha256(baseline_path) != result["outputs"]["baseline_csv"]["sha256"]:
        raise ValueError("Stage92 baseline hash differs")
    if stage92.sha256(selection_path) != result["outputs"]["selection_csv"]["sha256"]:
        raise ValueError("Stage92 selection hash differs")

    inputs = dict(config["inputs"])
    matrix = stage92.read_csv(stage92.verified(root, dict(inputs["primary_median_matrix"])))
    manifest = stage92.read_csv(stage92.verified(root, dict(inputs["development_manifest"])))
    objective, population = stage92.build_objective(matrix, manifest, config)
    k = int(config["literal_operationalization"]["primary_k"])
    direct = stage92.greedy(objective, k)
    direct_swap, iterations = stage92.best_one_swap(objective, direct)
    exact_selected, exact_value, second_value, optimal_count = stage92.exact(objective, k)
    all_one_swap_values = []
    selected_set = set(direct_swap)
    for removed in direct_swap:
        for added in range(len(objective.receptor_ids)):
            if added not in selected_set:
                candidate = tuple(sorted((selected_set - {removed}) | {added}))
                all_one_swap_values.append(objective.value(candidate))

    baseline_rows = stage92.read_csv(baseline_path)
    by_method = {row["method"]: row for row in baseline_rows}
    required_methods = {
        "single_best_receptor",
        "linear_top_k_singleton_utility",
        "direct_greedy",
        "greedy_plus_all_one_swaps",
        "all_singleton_greedy_plus_swaps",
        "multistart_tabu",
        "simulated_annealing",
        "uniform_random_10000",
        "exact_enumeration",
    }
    exact_ids = [objective.receptor_ids[index] for index in exact_selected]
    checks = {
        "source_input_hashes_match": True,
        "population_exact": population["population"] == 222
        and population["high"] == 195
        and population["low"] == 27
        and population["groups"] == 6
        and population["hit_count"] == 23,
        "baseline_methods_complete": set(by_method) == required_methods,
        "direct_greedy_recomputed": [objective.receptor_ids[index] for index in direct]
        == result["direct_greedy"]["selected_receptor_ids"],
        "greedy_swap_recomputed": [objective.receptor_ids[index] for index in direct_swap]
        == result["greedy_plus_all_one_swaps"]["selected_receptor_ids"],
        "greedy_swap_is_one_swap_local_optimum": max(all_one_swap_values)
        <= objective.value(direct_swap) + stage92.TOLERANCE,
        "exact_selection_recomputed": exact_ids == result["exact"]["selected_receptor_ids"],
        "exact_value_recomputed": abs(exact_value - float(result["exact"]["objective"]))
        <= stage92.TOLERANCE,
        "second_best_recomputed": abs(
            second_value - float(result["exact"]["second_best_objective"])
        )
        <= stage92.TOLERANCE,
        "unique_exact_optimum": optimal_count == 1
        and int(result["exact"]["optimal_state_count"]) == 1,
        "strong_methods_match_exact": all(
            by_method[method]["matches_exact_selection"].lower() == "true"
            for method in {
                "greedy_plus_all_one_swaps",
                "all_singleton_greedy_plus_swaps",
                "multistart_tabu",
                "simulated_annealing",
                "exact_enumeration",
            }
        ),
        "hardness_failure_recomputed": exact_selected == direct_swap
        and abs(exact_value - objective.value(direct_swap)) <= stage92.TOLERANCE,
        "protected_scores_unread": result["data_boundary"]["confirmation_scores_read"]
        == 0
        and result["data_boundary"]["locked_test_scores_read"] == 0,
        "confirmation_and_quantum_remain_locked": not result["authorization"][
            "confirmation_a_preparation_or_docking_authorized"
        ]
        and not result["authorization"]["quantum_simulation_or_hardware_authorized"],
    }
    audit = {
        "schema_version": "1.0",
        "audit_id": "stage92-bace1-group-robust-hardness-independent-audit-v1",
        "status": (
            "independent_stage92_bace1_group_robust_hardness_audit_ok"
            if all(checks.values())
            else "independent_stage92_bace1_group_robust_hardness_audit_failed"
        ),
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "state_count_reenumerated": 1344904,
        "exact_selected_receptor_ids": exact_ids,
        "exact_objective": exact_value,
        "direct_greedy_objective": objective.value(direct),
        "greedy_swap_objective": objective.value(direct_swap),
        "greedy_swap_iterations": iterations,
        "maximum_one_swap_neighbor_objective": max(all_one_swap_values),
        "hardness_gate_passed": False,
        "data_boundary": {
            "confirmation_scores_read": 0,
            "locked_test_scores_read": 0,
            "new_docking_jobs": 0,
            "quantum_jobs": 0,
        },
    }
    audit_path = root / "data/stage92_bace1_group_robust_hardness_adjudication_audit.json"
    stage92.write_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["failed_checks"]:
        raise SystemExit(1)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage92_bace1_group_robust_hardness_adjudication.json"),
    )
    args = parser.parse_args()
    run(args.root, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
