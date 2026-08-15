"""Build audited Stage79 core and external QCI execution bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import write_bundle
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle


COMMON_PATHS = (
    "configs/stage79_qci_dirac3_local_move_qubo_poc.json",
    "environment/stage79_qci_dirac3.yml",
    "scripts/prepare_stage79_qci_dirac3_poc.py",
    "scripts/audit_stage79_qci_dirac3_poc.py",
    "scripts/build_stage79_qci_dirac3_poc_bundles.py",
    "scripts/experimental/quantum/run_stage79_qci_dirac3_poc.py",
    "scripts/experimental/quantum/run_stage79_qci_preflight_remote.sh",
    "scripts/experimental/quantum/run_stage79_qci_device_remote.sh",
    "tests/test_stage79_qci_dirac3_poc.py",
    "data/stage78_advantage2_reverse_annealing_poc_result.json",
    "data/stage78_advantage2_reverse_annealing_poc_audit.json",
    "data/stage79_qci_dirac3_poc_result.json",
    "data/stage79_qci_dirac3_poc_audit.json",
    "reports/stage-79/qci_dirac3_poc_freeze.md",
    "reports/stage-79/qci_dirac3_external_execution.md",
    "results/runs/stage79_qci_dirac3_poc/instance_manifest.csv",
)


def instance_paths(result: dict) -> tuple[str, ...]:
    paths: list[str] = []
    for item in result["outputs"]["instance_files"]:
        for key in (
            "qci_polynomial",
            "qci_mapping",
            "source_bqm",
            "source_moves",
            "source_metadata",
        ):
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
        (root / "data/stage79_qci_dirac3_poc_result.json").read_text(
            encoding="ascii"
        )
    )
    audit = json.loads(
        (root / "data/stage79_qci_dirac3_poc_audit.json").read_text(
            encoding="ascii"
        )
    )
    if audit["status"] != (
        "stage79_qci_dirac3_local_move_qubo_poc_independent_audit_ok"
    ):
        raise ValueError("Stage79 bundles require the independent audit")
    instances = instance_paths(result)
    core_paths = tuple(sorted(set(COMMON_PATHS + instances)))
    external_exclusions = {
        "scripts/prepare_stage79_qci_dirac3_poc.py",
        "scripts/audit_stage79_qci_dirac3_poc.py",
        "scripts/build_stage79_qci_dirac3_poc_bundles.py",
        "tests/test_stage79_qci_dirac3_poc.py",
        "reports/stage-79/qci_dirac3_poc_freeze.md",
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
    protocol = result["hardware_protocol"]
    summary = {
        "schema_version": "1.0",
        "status": "stage79_qci_dirac3_poc_bundles_ok",
        "operation": "audited local translation plus allocation-only preflight package",
        "instance_count": result["instance_summary"]["instance_count"],
        "planned_device_jobs": protocol["planned_total_job_count"],
        "planned_device_samples": protocol["planned_total_sample_count"],
        "maximum_recorded_device_usage_seconds": protocol[
            "maximum_recorded_device_usage_seconds"
        ],
        "cloud_queries_run_while_building": 0,
        "qci_device_jobs_run_while_building": 0,
        "qci_device_execution_authorized": False,
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
