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
import scripts.run_stage95_pparg_md96_series_routing_scaling as stage95


def run(root: Path, config_path: Path) -> dict[str, object]:
    root = root.resolve()
    config = common.read_json(config_path.resolve())
    result = common.read_json(root / config["outputs"]["result_json"])
    bound = common.read_json(
        root / "data/stage95_pparg_md96_series_routing_bound_adjudication.json"
    )
    scale_path = root / config["outputs"]["scale_csv"]
    solution_path = root / config["outputs"]["solution_csv"]
    scale_rows = common.read_csv(scale_path)
    solution_rows = common.read_csv(solution_path)
    by_scale: dict[str, list[dict[str, str]]] = {}
    for row in solution_rows:
        by_scale.setdefault(row["scale_id"], []).append(row)

    scores = common.read_csv(root / config["inputs"]["scores"]["path"])
    ligand_rows = common.read_csv(root / config["inputs"]["ligand_manifest"]["path"])
    frame_rows = common.read_csv(root / config["inputs"]["frame_manifest"]["path"])
    series_rows = common.read_csv(root / config["inputs"]["series_manifest"])
    receptor_order = stage95.build_receptor_order(frame_rows)
    active_ids, decoy_ids, hit_sets, hit_count = stage95.build_hit_sets(
        scores,
        ligand_rows,
        receptor_order,
        float(config["objective"]["receptor_hit_fraction_per_seed"]),
    )

    objective_recomputations = []
    local_optimum_checks = []
    expected_scales = [str(row["scale_id"]) for row in config["nested_scale_grid"]]
    for scale in config["nested_scale_grid"]:
        scale_id = str(scale["scale_id"])
        receptor_count = int(scale["receptor_count"])
        receptor_ids = receptor_order[:receptor_count]
        pairs, _, utility, weights = stage95.build_utility(
            receptor_ids,
            active_ids,
            decoy_ids,
            [seed[:receptor_count] for seed in hit_sets],
            series_rows,
            int(scale["series_count"]),
        )
        routing = stage95.ConditionalRouting(receptor_count, pairs, utility, weights)
        receptor_index = {value: index for index, value in enumerate(receptor_ids)}
        strong_row = max(
            (row for row in by_scale[scale_id] if not row["method"].startswith("milp")),
            key=lambda row: float(row["objective"]),
        )
        strong_state = tuple(
            sorted(receptor_index[value] for value in strong_row["open_receptor_ids"].split(";"))
        )
        recomputed = routing.solve(strong_state)[0]
        objective_recomputations.append(
            {
                "scale_id": scale_id,
                "reported": float(strong_row["objective"]),
                "recomputed": recomputed,
                "matches": math.isclose(
                    recomputed, float(strong_row["objective"]), abs_tol=1e-10
                ),
            }
        )
        neighbor, neighbor_value, _ = routing.best_neighbor(strong_state)
        local_optimum_checks.append(
            {
                "scale_id": scale_id,
                "strong_state": strong_state,
                "best_neighbor": neighbor,
                "strong_objective": recomputed,
                "best_neighbor_objective": neighbor_value,
                "is_one_swap_local_optimum": neighbor_value
                <= recomputed + stage95.TOLERANCE,
            }
        )

    exact_rows = [row for row in scale_rows if row["milp_optimal"].lower() == "true"]
    bounded_rows = [row for row in scale_rows if row["milp_optimal"].lower() != "true"]
    checks = {
        "result_status": result["status"]
        == "stage95_pparg_md96_series_routing_hardness_not_supported",
        "output_hashes_match": common.sha256(scale_path)
        == result["outputs"]["scale_csv"]["sha256"]
        and common.sha256(solution_path)
        == result["outputs"]["solution_csv"]["sha256"],
        "scale_grid_complete": [row["scale_id"] for row in scale_rows]
        == expected_scales,
        "real_score_dimensions": len(scores) == 46080
        and len(receptor_order) == 96
        and len(active_ids) == len(decoy_ids) == 80
        and hit_count == 16,
        "structure_only_series_partition": len(series_rows) == 80
        and all(row["label"] == "active" for row in series_rows),
        "all_saved_strong_objectives_recomputed": all(
            row["matches"] for row in objective_recomputations
        ),
        "all_strong_solutions_are_one_swap_local_optima": all(
            row["is_one_swap_local_optimum"] for row in local_optimum_checks
        ),
        "four_exact_scales": len(exact_rows) == 4
        and all(float(row["certified_relative_gap"]) == 0.0 for row in exact_rows),
        "one_bounded_nonexact_scale": len(bounded_rows) == 1
        and bounded_rows[0]["scale_id"] == "r96_g48_k16",
        "dual_bound_dominates_strong_solution": all(
            float(row["certified_upper_bound"])
            + stage95.TOLERANCE
            >= float(row["best_strong_classical_objective"])
            for row in scale_rows
        ),
        "all_scales_exclude_one_percent_gap": all(
            row["one_percent_gap_mathematically_excluded"].lower() == "true"
            and float(row["maximum_possible_relative_gap"]) < 0.01
            for row in scale_rows
        ),
        "bound_adjudication_consistent": bound["exact_scale_count"] == 4
        and bound["bounded_nonexact_scale_count"] == 1
        and bound["one_percent_gap_excluded_scale_count"] == 5,
        "quantum_and_retuning_locked": result["quantum_hardware_authorized"] is False
        and result["same_matrix_objective_retuning_authorized"] is False,
        "protected_data_and_new_compute_unused": all(
            int(value) == 0 for value in result["data_boundary"].values()
        ),
    }
    audit = {
        "schema_version": "1.0",
        "audit_id": "stage95-pparg-md96-series-routing-independent-audit-v1",
        "status": (
            "independent_stage95_pparg_md96_series_routing_audit_ok"
            if all(checks.values())
            else "independent_stage95_pparg_md96_series_routing_audit_failed"
        ),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "objective_recomputations": objective_recomputations,
        "local_optimum_checks": local_optimum_checks,
        "exact_scale_count": len(exact_rows),
        "bounded_nonexact_scale_count": len(bounded_rows),
        "largest_maximum_possible_relative_gap": max(
            float(row["maximum_possible_relative_gap"]) for row in scale_rows
        ),
        "quantum_hardware_authorized": False,
        "data_boundary": result["data_boundary"],
    }
    output = root / "data/stage95_pparg_md96_series_routing_scaling_audit.json"
    common.write_json(output, audit)
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
        default=Path("configs/stage95_pparg_md96_series_routing_scaling.json"),
    )
    args = parser.parse_args()
    run(args.root, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
