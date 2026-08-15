"""Build the combined Stage36/36b/36c PPARG objective-redesign core bundle."""

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
    configs = [
        root / "configs/stage36_pparg_consensus_objective_landscape.json",
        root / "configs/stage36b_pparg_start_centered_consensus_landscape.json",
        root / "configs/stage36c_pparg_consensus_blend_replication.json",
    ]
    expected = [
        ("stage36_pparg_consensus_objective_landscape_complete", "stage36_pparg_consensus_objective_landscape_audit_ok"),
        ("stage36b_pparg_start_centered_consensus_landscape_complete", "stage36b_pparg_start_centered_consensus_landscape_audit_ok"),
        ("stage36c_pparg_consensus_blend_replication_complete", "stage36c_pparg_consensus_blend_replication_audit_ok"),
    ]
    paths = {
        "scripts/run_stage36_pparg_consensus_objective_landscape.py",
        "scripts/audit_stage36_pparg_consensus_objective_landscape.py",
        "scripts/run_stage36b_pparg_start_centered_consensus_landscape.py",
        "scripts/audit_stage36b_pparg_start_centered_consensus_landscape.py",
        "scripts/run_stage36c_pparg_consensus_blend_replication.py",
        "scripts/audit_stage36c_pparg_consensus_blend_replication.py",
        "scripts/build_stage36_pparg_objective_redesign_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/run_stage21_structure_aware_qubo.py",
        "scripts/run_stage29_pparg_md_qubo_scaling.py",
        "scripts/run_stage30_pparg_group_balanced_state_qubo.py",
        "scripts/run_stage31_pparg_objective_landscape_screen.py",
        "scripts/__init__.py",
        "tests/test_stage36_pparg_consensus_objective_landscape.py",
        "tests/test_stage36b_c_pparg_consensus_landscapes.py",
        "pyproject.toml",
    }
    for config_path, (result_status, audit_status) in zip(configs, expected):
        config = json.loads(config_path.read_text(encoding="ascii"))
        result = json.loads((root / config["outputs"]["result_json"]).read_text(encoding="ascii"))
        audit = json.loads((root / config["outputs"]["audit_json"]).read_text(encoding="ascii"))
        if result.get("status") != result_status or audit.get("status") != audit_status:
            raise ValueError(f"incomplete Stage36 record: {config_path.name}")
        paths.add(config_path.relative_to(root).as_posix())
        paths.update(str(value).replace("\\", "/") for value in config["inputs"].values())
        paths.update(str(value).replace("\\", "/") for value in config["outputs"].values())
    bundle = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
