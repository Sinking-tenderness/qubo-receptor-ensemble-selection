"""Build the Stage61b metadata-only recovery bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_stage05_mk14_remote_bundle import write_bundle


PATHS = (
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/repair_stage61b_progress_descriptor.py",
    "scripts/experimental/unidock/package_stage61b_recovered_results.py",
    "scripts/experimental/unidock/run_stage61b_metadata_recovery_remote.sh",
    "reports/stage-61b/ppard_progress_descriptor_recovery.md",
    "tests/test_stage61b_progress_descriptor_recovery.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, sorted(PATHS))
    result.update(
        {
            "operation": "Stage61b metadata-only progress descriptor repair, independent audit, and result packaging",
            "target_id": "PPARD",
            "gpu_required": False,
            "docking_jobs_reexecuted": 0,
            "approved_executed_config_sha256": "644A4A1B42FA4526A4A297FB775F37519AB4CEF9E6BF14C19D5DC1EFB1764019",
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
