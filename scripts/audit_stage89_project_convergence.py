import argparse
import csv
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="ascii"))


def run(root, config_path):
    config = read_json(root / config_path)
    result = read_json(root / config["outputs"]["result_json"])
    mk14 = read_json(root / config["inputs"]["mk14_fresh_validation"])
    stage80 = read_json(root / config["inputs"]["local_move_hardness"])
    stage86a = read_json(root / config["inputs"]["qci_global_penalty_adjudication"])
    stage87 = read_json(root / config["inputs"]["quantum_value_instance_gate"])

    with (root / config["outputs"]["claim_evidence_csv"]).open(
        encoding="ascii", newline=""
    ) as handle:
        claims = {row["claim_id"]: row for row in csv.DictReader(handle)}

    source_qubo = mk14["method_metrics"]["pair_synergy_qubo"]["primary"][
        "bedroc_alpha_20"
    ]
    source_greedy = mk14["method_metrics"]["nested_greedy_final"]["primary"][
        "bedroc_alpha_20"
    ]
    checks = {
        "claim_ids_complete": set(claims) == {f"C{index}" for index in range(1, 9)},
        "mk14_qubo_equals_greedy_in_source": abs(source_qubo - source_greedy) < 1e-12,
        "mk14_values_match_result": (
            result["critical_results"]["mk14_primary_bedroc"]["qubo"] == source_qubo
            and result["critical_results"]["mk14_primary_bedroc"]["greedy"]
            == source_greedy
        ),
        "quantum_advantage_not_established": claims["C7"]["status"]
        == "not_established",
        "drug_discovery_not_tested": claims["C8"]["status"] == "not_tested",
        "local_trap_count_matches_source": (
            result["critical_results"]["stage80_multi_move_local_traps"]
            == stage80["summary"]["local_trap_candidate_count"]
            == 0
        ),
        "global_feasible_count_matches_source": (
            result["critical_results"]["stage86_fully_feasible_physical_samples"]
            == stage86a["constraint_fidelity"]["fully_feasible_count"]
            == 0
        ),
        "strict_instance_gate_remains_closed": (
            result["critical_results"]["stage87_quantum_worthy_instance_gate_passed"]
            == stage87["strict_instance_gate_passed"]
            is False
        ),
        "new_experiment_routes_closed": all(
            result["authorization"][key] is False
            for key in (
                "new_objective_search_authorized",
                "new_target_docking_authorized",
                "new_quantum_hardware_jobs_authorized",
            )
        ),
        "delivery_routes_open": all(
            result["authorization"][key] is True
            for key in (
                "manuscript_preparation_authorized",
                "reproducibility_packaging_authorized",
            )
        ),
        "zero_new_compute": result["data_boundary"]
        == {
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
    }
    audit = {
        "schema_version": "1.0",
        "status": (
            "stage89_project_convergence_independent_audit_ok"
            if all(checks.values())
            else "stage89_project_convergence_independent_audit_failed"
        ),
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }
    output = root / config["outputs"]["audit_json"]
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["failed_checks"]:
        raise SystemExit(1)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage89_project_convergence.json"),
    )
    args = parser.parse_args()
    run(args.root.resolve(), args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
