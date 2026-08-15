from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage97_project_convergence_amendment_stage96.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    stage89 = json.loads((root / config["inputs"]["stage89_result"]).read_text(encoding="utf-8"))
    stage96 = json.loads((root / config["inputs"]["stage96_result"]).read_text(encoding="utf-8"))
    stage96_audit = json.loads((root / config["inputs"]["stage96_audit"]).read_text(encoding="utf-8"))
    if stage89["status"] != "stage89_project_converged_claims_frozen":
        raise ValueError("Stage89 is not in its frozen state")
    if stage96["status"] != "stage96_replay_complete":
        raise ValueError("Stage96 replay is incomplete")
    if stage96_audit["status"] != "stage96_audit_ok":
        raise ValueError("Stage96 independent audit failed")
    if stage96["policy_gate"]["passes"] or stage96["solver_value"]["passes"]:
        raise ValueError("Stage96 negative amendment unexpectedly changed to positive")

    gate_rows = stage96["policy_gate"]["targets"]
    rows = [
        {
            "claim_id": "C1",
            "claim": stage89["frozen_thesis"],
            "status": "supported_with_scope",
            "evidence": "Stage89 frozen evidence plus Stage96 real-matrix replay audit.",
            "boundary": "Feasibility and practical-limit study; no quantum advantage claim.",
        },
        {
            "claim_id": "C9",
            "claim": "A hidden sequential docking-policy replay shows that the current QUBO batch selector recovers early-recognition performance more efficiently than strong classical policies.",
            "status": "not_supported",
            "evidence": "; ".join(f"{row['target_id']}: mean gain at 20% budget={row['mean_gain']:.6f}, policy_gate={row['passes_policy_gate']}" for row in gate_rows),
            "boundary": "This negative result is post-hoc on completed PPARG/BACE1 matrices; it does not authorize changing the objective on the same data.",
        },
        {
            "claim_id": "C10",
            "claim": "The exact QUBO solver exposes a measurable optimization gap over a strong classical one-swap solver on the replay instances.",
            "status": "not_supported",
            "evidence": f"Stage96 maximum exact-minus-one-swap objective gap={stage96['solver_value']['max_exact_minus_one_swap']:.3e}; positive comparison count={stage96['solver_value']['positive_comparison_count']}.",
            "boundary": "No quantum hardware execution is justified by these instances.",
        },
    ]
    output_dir = root / Path(config["outputs"]["claim_evidence_csv"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = root / config["outputs"]["claim_evidence_csv"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "schema_version": "1.0",
        "status": "stage97_project_convergence_amended_stage96",
        "base_freeze": "stage89_project_converged_claims_frozen",
        "positioning": "feasibility_and_boundary_study",
        "new_supported_claim_count": 0,
        "new_negative_claim_count": 2,
        "stage96_policy_gate_passed": False,
        "stage96_solver_value_passed": False,
        "authorization": config["policy"],
        "decision": "Do not spend on new QUBO tuning or quantum hardware on the current matrices. Proceed with manuscript, figures, reproducibility package, and a separately preregistered future instance only if an independently validated classical-hard gate is first met.",
        "stage96_input_hashes": {name: sha256(root / path) for name, path in config["inputs"].items() if name != "stage96_report"},
        "outputs": {"claim_evidence_csv": str(csv_path.relative_to(root)).replace("\\", "/"), "claim_count": len(rows)},
    }
    result_path = root / config["outputs"]["result_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    audit = {
        "schema_version": "1.0",
        "status": "stage97_audit_ok",
        "stage89_preserved": True,
        "stage96_audit_status": stage96_audit["status"],
        "fresh_validation_rows": 0,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
    audit_path = root / config["outputs"]["audit_json"]
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report = f"""# Stage97 Stage96 convergence amendment

Stage89 remains preserved as the original frozen claim ledger. Stage96 adds a completed hidden-matrix replay on real PPARG and BACE1 Uni-Dock matrices.

## Decision

The Stage96 adaptive docking policy gate failed. At the primary 20% task budget, QUBO exact was below the best non-QUBO policy on both targets, and exact QUBO never exceeded the classical one-swap solver on the same candidate problems.

The project therefore remains a feasibility-and-boundary study. Manuscript preparation and reproducibility packaging continue; new objective tuning, new target docking, and quantum hardware spending remain blocked until a new preregistered instance passes an independent classical-hardness gate.

## Stage96 evidence

""" + "\n".join(f"- {row['target_id']}: QUBO exact mean BEDROC={row['qubo_exact_mean_bedroc']:.6f}; best non-QUBO={row['best_nonqubo_mean_bedroc']:.6f}; mean gain={row['mean_gain']:.6f}; pass={row['passes_policy_gate']}." for row in gate_rows) + """

## Interpretation

This is a useful negative result: it rules out the current adaptive-QUBO formulation as a demonstrated advantage on the available matrices. It does not prove that every future protein or every quantum algorithm will fail, but it does define the evidence required before reopening the experimental branch.
"""
    report_path = root / config["outputs"]["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"status": result["status"], "audit": audit["status"], "policy_gate": False, "solver_value": False, "result": str(result_path), "report": str(report_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
