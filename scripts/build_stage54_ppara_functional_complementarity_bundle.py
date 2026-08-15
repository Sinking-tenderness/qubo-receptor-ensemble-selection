"""Build the deterministic Stage 54 PPARA diagnosis evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle


PATHS = (
    "configs/stage54_ppara_functional_complementarity_diagnosis.json",
    "data/stage53_ppara_large_pool_qubo_transfer_result.json",
    "data/stage53_ppara_large_pool_qubo_transfer_audit.json",
    "data/stage54_future_target_intake_criteria.json",
    "data/stage54_ppara_functional_complementarity_diagnosis_result.json",
    "data/stage54_ppara_functional_complementarity_diagnosis_audit.json",
    "data/processed/stage52a_ppara_train374_unidock_pdbqt_manifest.csv",
    "data/processed/stage52b_ppara_stage51_passing20_receptor_manifest.csv",
    "results/runs/stage52c_ppara_target_id_amendment/scores.csv",
    "results/runs/stage53_ppara_large_pool_qubo_transfer/fold_assignments.csv",
    "results/runs/stage54_ppara_functional_complementarity_diagnosis/receptor_diagnostics.csv",
    "results/runs/stage54_ppara_functional_complementarity_diagnosis/pair_diagnostics.csv",
    "results/runs/stage54_ppara_functional_complementarity_diagnosis/fold_oracle_diagnostics.csv",
    "reports/stage-54/ppara_functional_complementarity_diagnosis.md",
    "scripts/diagnose_stage54_ppara_functional_complementarity.py",
    "scripts/audit_stage54_ppara_functional_complementarity.py",
    "scripts/build_stage54_ppara_functional_complementarity_bundle.py",
    "scripts/run_stage42d_bace1_large_pool_qubo_screen.py",
    "scripts/run_stage42f_bace1_rank_sensitive_pair_qubo.py",
    "scripts/run_stage53_ppara_large_pool_qubo_transfer.py",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "tests/test_stage54_ppara_functional_complementarity.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, list(PATHS))
    result.update(
        {
            "operation": "Stage54 PPARA functional-complementarity failure diagnosis and prospective target-intake gate",
            "target_id": "PPARA",
            "receptor_count": 20,
            "ligand_count": 374,
            "pair_count": 190,
            "outer_fold_count": 4,
            "failure_mechanism_resolved": True,
            "ppara_future_intake_gate": "fail",
            "same_data_retuning_authorized": False,
            "fresh_validation_rows": 0,
            "locked_test_rows": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        }
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
