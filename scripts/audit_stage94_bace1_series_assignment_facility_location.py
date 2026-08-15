from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_stage92_bace1_group_robust_hardness_adjudication as common
import scripts.run_stage94_bace1_series_assignment_facility_location as stage94


def run(root: Path, config_path: Path) -> dict[str, object]:
    root = root.resolve()
    config = common.read_json(config_path.resolve())
    outputs = dict(config["outputs"])
    result = common.read_json(root / outputs["result_json"])
    baseline_path = root / outputs["baseline_csv"]
    assignment_path = root / outputs["assignment_csv"]

    scores = common.read_csv(root / config["inputs"]["amended_seed_scores"])
    manifest = common.read_csv(root / config["inputs"]["development_manifest"])
    receptor_ids, group_ids, pairs, catalog, dimensions = stage94.build_pair_catalog(
        scores, manifest, config
    )
    conditional = stage94.ConditionalAssignment(pairs, catalog)
    exact = stage94.solve_milp(
        len(receptor_ids),
        pairs,
        catalog,
        int(config["frozen_problem"]["open_receptor_count"]),
    )
    exact_value, exact_assignments, exact_threshold = conditional.solve(exact["opened"])

    baseline_rows = common.read_csv(baseline_path)
    by_method = {row["method"]: row for row in baseline_rows}
    required_methods = {
        "direct_greedy_open_set",
        "greedy_plus_one_swap",
        "all_pair_starts_plus_one_swap",
        "random256_plus_one_swap",
        "multistart_tabu",
        "simulated_annealing",
        "milp_exact",
    }
    strong_methods = {
        "all_pair_starts_plus_one_swap",
        "random256_plus_one_swap",
        "simulated_annealing",
        "milp_exact",
    }
    exact_ids = [receptor_ids[index] for index in exact["opened"]]
    exact_set = set(exact["opened"])
    neighbor_values = []
    for removed in exact["opened"]:
        for added in range(len(receptor_ids)):
            if added in exact_set:
                continue
            candidate = tuple(sorted((exact_set - {removed}) | {added}))
            neighbor_values.append(conditional.solve(candidate)[0])

    assignment_rows = common.read_csv(assignment_path)
    expected_pairs = [pairs[index] for index in exact_assignments]
    assignments_match = len(assignment_rows) == len(group_ids)
    for group_index, row in enumerate(assignment_rows):
        left, right = expected_pairs[group_index]
        assignments_match = assignments_match and (
            row["series_id"] == group_ids[group_index]
            and row["receptor_a"] == receptor_ids[left]
            and row["receptor_b"] == receptor_ids[right]
        )

    checks = {
        "source_status_is_failed_gate": result.get("status")
        == "stage94_bace1_series_assignment_gate_failed",
        "output_hashes_match": common.sha256(baseline_path)
        == result["outputs"]["baseline_csv"]["sha256"]
        and common.sha256(assignment_path)
        == result["outputs"]["assignment_csv"]["sha256"],
        "dimensions_recomputed": dimensions == result["dimensions"],
        "baseline_methods_complete": set(by_method) == required_methods,
        "milp_optimality_certificate_recomputed": exact["mip_gap"] <= 1e-12,
        "milp_open_set_recomputed": exact_ids == result["milp"]["open_receptor_ids"],
        "milp_objective_recomputed": math.isclose(
            exact_value, float(result["milp"]["objective"]), abs_tol=1e-9
        ),
        "conditional_assignments_recomputed": tuple(exact["assignments"])
        == exact_assignments,
        "assignment_table_recomputed": assignments_match,
        "worst_coverage_recomputed": math.isclose(
            exact_threshold,
            float(result["milp"]["worst_series_coverage"]),
            abs_tol=1e-9,
        ),
        "exact_is_one_swap_local_optimum": max(neighbor_values)
        <= exact_value + stage94.TOLERANCE,
        "strong_classical_methods_reach_exact": all(
            by_method[method]["matches_milp_open_set"].lower() == "true"
            and abs(float(by_method[method]["objective_gap_to_milp"])) <= 1e-9
            for method in strong_methods
        ),
        "hardness_failure_recomputed": by_method["random256_plus_one_swap"][
            "matches_milp_open_set"
        ].lower()
        == "true"
        and int(result["best_one_swap"]["replacement_distance_to_milp"]) == 0,
        "protected_data_and_hardware_unread": all(
            int(value) == 0 for value in result["data_boundary"].values()
        ),
        "all_authorizations_remain_locked": not any(
            bool(value) for value in result["authorization"].values()
        ),
    }
    audit = {
        "schema_version": "1.0",
        "audit_id": "stage94-bace1-series-assignment-facility-location-independent-audit-v1",
        "status": (
            "independent_stage94_bace1_series_assignment_audit_ok"
            if all(checks.values())
            else "independent_stage94_bace1_series_assignment_audit_failed"
        ),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "milp_objective_recomputed": exact_value,
        "milp_open_receptor_ids_recomputed": exact_ids,
        "maximum_one_swap_neighbor_objective": max(neighbor_values),
        "strong_classical_methods_matching_milp": sorted(strong_methods),
        "hardness_gate_passed": False,
        "data_boundary": result["data_boundary"],
    }
    audit_path = root / "data/stage94_bace1_series_assignment_facility_location_audit.json"
    common.write_json(audit_path, audit)
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
        default=Path(
            "configs/stage94_bace1_series_assignment_facility_location.json"
        ),
    )
    args = parser.parse_args()
    run(args.root, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
