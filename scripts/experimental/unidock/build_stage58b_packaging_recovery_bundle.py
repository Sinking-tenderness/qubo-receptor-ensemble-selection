"""Build the Stage58b packaging-only recovery amendment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle
from scripts.experimental.unidock.repair_stage58b_ppard_pilot96_packaging import (
    FROZEN_ADAPTER_SHA256,
    FROZEN_CONFIG_SHA256,
)
from scripts.experimental.unidock.run_unidock_gpu_equivalence import file_sha256


PATHS = (
    "scripts/experimental/unidock/repair_stage58b_ppard_pilot96_packaging.py",
    "scripts/experimental/unidock/run_stage58b_ppard_packaging_recovery_remote.sh",
    "scripts/experimental/unidock/build_stage58b_packaging_recovery_bundle.py",
    "reports/stage-58/ppard_pilot96_packaging_recovery_amendment01.md",
)


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = root / "configs/stage58b_ppard_pilot96_unidock113_production.json"
    adapter = root / "scripts/experimental/unidock/run_stage58b_ppard_pilot96_production.py"
    if file_sha256(config) != FROZEN_CONFIG_SHA256:
        raise ValueError("Stage58b frozen config changed")
    if file_sha256(adapter) != FROZEN_ADAPTER_SHA256:
        raise ValueError("Stage58b frozen adapter changed")
    if any(not (root / path).is_file() for path in PATHS):
        raise ValueError("Stage58b recovery amendment is incomplete")
    return sorted(PATHS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, bundle_paths(root))
    result.update(
        {
            "operation": "Stage58b packaging-only metadata recovery amendment",
            "docking_batches_rerun": 0,
            "docking_scores_changed": 0,
            "pose_files_changed": 0,
            "gpu_required": False,
            "expected_completed_batch_count": 87,
            "expected_completed_pair_count": 8352,
            "frozen_config_sha256": FROZEN_CONFIG_SHA256,
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
