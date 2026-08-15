"""Build audited Stage78 core and external-execution bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
    from scripts.build_stage77_quantum_hardware_interface_gate_bundle import (
        PATHS as STAGE77_PATHS,
    )
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle
    from build_stage77_quantum_hardware_interface_gate_bundle import (
        PATHS as STAGE77_PATHS,
    )


COMMON_PATHS = (
    "configs/stage78_advantage2_reverse_annealing_poc.json",
    "environment/stage78_dwave_advantage2.yml",
    "scripts/prepare_stage78_advantage2_reverse_annealing_poc.py",
    "scripts/audit_stage78_advantage2_reverse_annealing_poc.py",
    "scripts/build_stage78_advantage2_reverse_annealing_poc_bundles.py",
    "scripts/experimental/quantum/run_stage78_advantage2_reverse_annealing_poc.py",
    "scripts/experimental/quantum/run_stage78_advantage2_preflight_remote.sh",
    "tests/test_stage78_advantage2_reverse_annealing_poc.py",
    "data/stage78_advantage2_reverse_annealing_poc_result.json",
    "data/stage78_advantage2_reverse_annealing_poc_audit.json",
    "reports/stage-78/advantage2_reverse_annealing_poc_freeze.md",
    "reports/stage-78/advantage2_external_execution.md",
    "results/runs/stage78_advantage2_reverse_annealing_poc/instance_manifest.csv",
    "results/runs/stage78_advantage2_reverse_annealing_poc/classical_controls.csv",
)


def instance_paths(result: dict) -> tuple[str, ...]:
    paths: list[str] = []
    for item in result["outputs"]["instance_files"]:
        for key in ("bqm", "moves", "metadata"):
            paths.append(str(item[key]["path"]))
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--core-output", type=Path, required=True)
    parser.add_argument("--external-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = json.loads(
        (root / "data/stage78_advantage2_reverse_annealing_poc_result.json").read_text(
            encoding="ascii"
        )
    )
    audit = json.loads(
        (root / "data/stage78_advantage2_reverse_annealing_poc_audit.json").read_text(
            encoding="ascii"
        )
    )
    if audit["status"] != (
        "stage78_advantage2_reverse_annealing_poc_independent_audit_ok"
    ):
        raise ValueError("Stage78 bundles require the independent audit")
    instances = instance_paths(result)
    core_paths = tuple(sorted(set(STAGE77_PATHS + COMMON_PATHS + instances)))
    external_exclusions = {
        "scripts/prepare_stage78_advantage2_reverse_annealing_poc.py",
        "scripts/audit_stage78_advantage2_reverse_annealing_poc.py",
        "scripts/build_stage78_advantage2_reverse_annealing_poc_bundles.py",
        "tests/test_stage78_advantage2_reverse_annealing_poc.py",
    }
    external_paths = tuple(
        sorted(
            path
            for path in set(COMMON_PATHS + instances)
            if path not in external_exclusions
        )
    )
    core = write_bundle(root, args.core_output, list(core_paths))
    external = write_bundle(root, args.external_output, list(external_paths))
    summary = {
        "schema_version": "1.0",
        "status": "stage78_advantage2_reverse_annealing_poc_bundles_ok",
        "operation": "audited local freeze plus no-sampling external preflight package",
        "frozen_instance_count": result["instance_summary"]["frozen_instance_count"],
        "confirmation_positive_count": result["instance_summary"][
            "hardware_resolvable_positive_count"
        ],
        "confirmation_negative_count": result["instance_summary"][
            "negative_control_count"
        ],
        "calibration_diagnostic_count": result["instance_summary"][
            "calibration_diagnostic_count"
        ],
        "planned_qpu_jobs": result["hardware_protocol"][
            "planned_default_total_qpu_jobs"
        ],
        "planned_qpu_reads": result["hardware_protocol"][
            "planned_default_total_qpu_reads"
        ],
        "maximum_qpu_access_time_seconds": result["hardware_protocol"][
            "maximum_planned_qpu_access_time_seconds"
        ],
        "cloud_queries_run_while_building": 0,
        "qpu_jobs_run_while_building": 0,
        "paid_qpu_execution_authorized": False,
        "core_bundle": core,
        "external_execution_bundle": external,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
