"""Build the deterministic Stage31 PPARG objective-landscape core bundle."""

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
    config_path = root / "configs/stage31_pparg_objective_landscape_screen.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    paths = {
        "configs/stage31_pparg_objective_landscape_screen.json",
        "scripts/run_stage31_pparg_objective_landscape_screen.py",
        "scripts/audit_stage31_pparg_objective_landscape_screen.py",
        "scripts/build_stage31_pparg_objective_landscape_screen_bundle.py",
        "scripts/run_stage29_pparg_md_qubo_scaling.py",
        "scripts/run_stage21_structure_aware_qubo.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/__init__.py",
        "tests/test_stage31_pparg_objective_landscape_screen.py",
        "pyproject.toml",
        "data/stage31_pparg_objective_landscape_screen_result.json",
        "data/stage31_pparg_objective_landscape_screen_audit.json",
    }
    paths.update(str(value).replace("\\", "/") for value in config["inputs"].values())
    paths.update(str(value).replace("\\", "/") for value in config["outputs"].values())
    result = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
