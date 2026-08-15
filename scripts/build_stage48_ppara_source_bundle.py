"""Build the deterministic Stage48 PPARA source and preregistration bundle."""

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
    result = json.loads((root / "data/stage48_ppara_source_audit.json").read_text(encoding="ascii"))
    audit = json.loads((root / "data/stage48_ppara_source_audit_independent.json").read_text(encoding="ascii"))
    if result.get("status") != "stage48_ppara_source_audit_ok":
        raise ValueError("Stage48 source result is incomplete")
    if audit.get("status") != "stage48_ppara_source_independent_audit_ok":
        raise ValueError("Stage48 independent audit is incomplete")
    config_path = root / result["config"]["path"]
    config = json.loads(config_path.read_text(encoding="ascii"))
    paths = {
        result["config"]["path"],
        "configs/stage47_new_target_feasibility_screen.json",
        "configs/stage47b_expanded_new_target_feasibility_screen.json",
        "configs/stage46_src_independent_k6_preregistration.json",
        "configs/stage46_src_independent_k6_preregistration_amendment01.json",
        "scripts/audit_stage48_ppara_source.py",
        "scripts/audit_stage48_ppara_source_result.py",
        "scripts/screen_stage47_new_target_feasibility.py",
        "scripts/build_stage48_ppara_source_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/audit_external_target_intake.py",
        "scripts/discover_mk14_rcsb_receptor_candidates.py",
        "scripts/discover_stage13_external_target_rcsb_candidates.py",
        "scripts/__init__.py",
        "tests/test_stage47_48_target_intake.py",
        "data/stage46a_src_metadata_discovery_failure_adjudication.json",
        "data/stage46b_src_human_metadata_pool_failure_adjudication.json",
        "data/stage47_new_target_feasibility_screen_result.json",
        "data/stage47b_expanded_new_target_feasibility_screen_result.json",
        "data/stage48_ppara_source_audit.json",
        "data/stage48_ppara_source_audit_independent.json",
        "data/processed/stage47_new_target_feasibility_screen.csv",
        "data/processed/stage47b_expanded_new_target_feasibility_screen.csv",
        "pyproject.toml",
    }
    paths.update(value["path"] for value in config["inputs"].values())
    bundle = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
