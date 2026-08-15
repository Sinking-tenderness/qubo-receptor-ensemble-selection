"""Independently audit Stage87 evidence convergence outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="ascii"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    outputs = config["outputs"]
    result = read_json(root / outputs["result_json"])
    candidates = read_csv(root / outputs["candidate_instances_csv"])
    evidence = read_csv(root / outputs["evidence_matrix_csv"])
    checks = {
        "candidate_count": len(candidates) == 4,
        "all_exact_certificates": all(
            row["exact_certificate_verified"] == "True" for row in candidates
        ),
        "all_holdout_gains_positive": all(
            float(row["holdout_robust_gain_over_direct_greedy"]) > 0
            and float(row["holdout_robust_gain_over_greedy_swap"]) > 0
            for row in candidates
        ),
        "all_candidates_exhaustively_trivial": all(
            row["exhaustively_trivial"] == "True" for row in candidates
        ),
        "state_bound": max(int(row["total_fixed_k_states"]) for row in candidates)
        == 38760,
        "evidence_block_count": len(evidence) == 4,
        "stage74_no_certified_miss": result["historical_hardness_summary"][
            "stage74_strong_classical_miss_count"
        ]
        == 0,
        "stage75_no_certified_miss": result["historical_hardness_summary"][
            "stage75_joint_classical_miss_count"
        ]
        == 0,
        "stage80_no_multi_move_trap": result["historical_hardness_summary"][
            "stage80_multi_move_trap_count"
        ]
        == 0,
        "qaoa_blocked": not result["constraint_preserving_qaoa_simulation_authorized"],
        "no_hardware": result["new_quantum_hardware_jobs_authorized"] == 0,
    }
    if not all(checks.values()):
        raise ValueError(f"Stage87 independent audit failed: {checks}")
    audit = {
        "schema_version": "1.0",
        "status": "stage87_quantum_value_instance_gate_independent_audit_ok",
        "checks": checks,
        "candidate_count": len(candidates),
        "maximum_total_states": max(
            int(row["total_fixed_k_states"]) for row in candidates
        ),
        "qaoa_authorized": False,
        "quantum_hardware_jobs_authorized": 0,
    }
    path = root / outputs["audit_json"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage87_quantum_value_instance_gate.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    audit = run((root / args.config).resolve(), root)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
