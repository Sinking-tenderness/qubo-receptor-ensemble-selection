"""Build the deterministic Stage25 BACE1 replication bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads(
        (root / "configs/stage25_bace1_prospective_structure_replication.json").read_text(encoding="ascii")
    )
    paths = {
        "configs/stage25_bace1_prospective_structure_replication.json",
        "scripts/run_stage25_bace1_prospective_structure_replication.py",
        "scripts/audit_stage25_bace1_prospective_structure_replication.py",
        "scripts/build_stage25_bace1_prospective_structure_replication_bundle.py",
        "scripts/diagnose_stage22_beam_baseline.py",
        "scripts/run_stage23_qubo_sampler_stability.py",
        "scripts/run_stage22_structural_state_coverage_qubo.py",
        "scripts/run_stage21_structure_aware_qubo.py",
        "scripts/__init__.py",
        "tests/test_stage25_bace1_prospective_structure_replication.py",
        "data/stage25_bace1_prospective_structure_replication_result.json",
        "data/stage25_bace1_prospective_structure_replication_audit.json",
        "data/stage25_bace1_prospective_structure_replication_model_record.json",
        "reports/stage-25/bace1_prospective_structure_replication.md",
        "pyproject.toml",
    }
    paths.update(
        str(value).replace("\\", "/")
        for value in config["target"]["inputs"].values()
    )
    paths.update(
        str(value).replace("\\", "/")
        for value in config["outputs"].values()
    )
    result = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
