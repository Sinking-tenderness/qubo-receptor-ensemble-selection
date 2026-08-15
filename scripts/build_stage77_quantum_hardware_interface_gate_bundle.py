"""Build the audited Stage77 quantum-hardware interface core bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.build_stage76_variable_k_sampler_repair_bundle import (
        PATHS as STAGE76_PATHS,
    )
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle
    from build_stage76_variable_k_sampler_repair_bundle import (
        PATHS as STAGE76_PATHS,
    )


PATHS = tuple(
    sorted(
        set(
            STAGE76_PATHS
            + (
                "analysis/literature-search-20260807-quantum-hardware-variable-k-cqm/papers.md",
                "analysis/literature-search-20260807-quantum-hardware-variable-k-cqm/papers.csv",
                "analysis/literature-search-20260807-quantum-hardware-variable-k-cqm/search-notes.md",
                "configs/stage77_quantum_hardware_interface_gate.json",
                "scripts/run_stage77_quantum_hardware_interface_gate.py",
                "scripts/audit_stage77_quantum_hardware_interface_gate.py",
                "scripts/build_stage77_quantum_hardware_interface_gate_bundle.py",
                "tests/test_stage77_quantum_hardware_interface_gate.py",
                "results/runs/stage77_quantum_hardware_interface_gate/direct_encoding_metrics.csv",
                "results/runs/stage77_quantum_hardware_interface_gate/local_swap_bqm_metrics.csv",
                "results/runs/stage77_quantum_hardware_interface_gate/emulation_trials.csv",
                "results/runs/stage77_quantum_hardware_interface_gate/emulation_summary.csv",
                "results/runs/stage77_quantum_hardware_interface_gate/hardware_route_review.csv",
                "data/stage77_quantum_hardware_interface_gate_result.json",
                "data/stage77_quantum_hardware_interface_gate_audit.json",
                "reports/stage-77/quantum_hardware_interface_gate.md",
            )
        )
    )
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = json.loads(
        (root / "data/stage77_quantum_hardware_interface_gate_result.json").read_text(
            encoding="ascii"
        )
    )
    audit = json.loads(
        (root / "data/stage77_quantum_hardware_interface_gate_audit.json").read_text(
            encoding="ascii"
        )
    )
    if audit["status"] != "stage77_quantum_hardware_interface_independent_audit_ok":
        raise ValueError("Stage77 bundle requires the independent audit")
    result = write_bundle(root, args.output, list(PATHS))
    result.update(
        {
            "operation": "audited Stage77 quantum-hardware interface gate",
            "target_count": 4,
            "cqm_model_count": source["direct_encoding_summary"]["cqm_model_count"],
            "local_swap_bqm_subproblem_count": source["local_swap_bqm_summary"][
                "subproblem_count"
            ],
            "emulation_run_count": sum(
                row["run_count"] for row in source["emulation_summaries"]
            ),
            "maximum_local_logical_variable_count": source[
                "local_swap_bqm_summary"
            ]["maximum_encoded_move_variable_count"],
            "maximum_local_ideal_zephyr_chain_length": source[
                "local_swap_bqm_summary"
            ]["maximum_ideal_zephyr_chain_length"],
            "hardware_resolvable_improvement_subproblem_count": source[
                "local_swap_bqm_summary"
            ]["hardware_resolvable_improvement_subproblem_count"],
            "unique_hardware_resolvable_fixed_k_instance_count": source[
                "local_swap_bqm_summary"
            ]["unique_hardware_resolvable_fixed_k_instance_count"],
            "full_direct_qpu_route_authorized": False,
            "local_reverse_annealing_poc_ready_for_budget_request": source[
                "decision"
            ]["advantage2_local_reverse_annealing_poc_ready_for_budget_request"],
            "paid_cloud_execution_authorized": False,
            "paid_qpu_execution_authorized": False,
            "quantum_scaling_claim_authorized": False,
            "quantum_advantage_claim_authorized": False,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "cloud_cqm_jobs": 0,
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
