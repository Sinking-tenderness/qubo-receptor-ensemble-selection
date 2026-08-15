"""Build the deterministic Stage29 PPARG MD QUBO scaling core bundle."""

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
    config_path = root / "configs/stage29_pparg_md_qubo_solver_scaling.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    paths = {
        "configs/stage29_pparg_md_qubo_solver_scaling.json",
        "scripts/run_stage29_pparg_md_qubo_scaling.py",
        "scripts/audit_stage29_pparg_md_qubo_scaling.py",
        "scripts/build_stage29_pparg_md_qubo_scaling_bundle.py",
        "scripts/run_stage21_structure_aware_qubo.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/__init__.py",
        "tests/test_stage29_pparg_md_qubo_scaling.py",
        "pyproject.toml",
        "data/stage29_pparg_md_qubo_solver_scaling_result.json",
        "data/stage29_pparg_md_qubo_solver_scaling_audit.json",
    }
    paths.update(str(value).replace("\\", "/") for value in config["inputs"].values())
    paths.update(str(value).replace("\\", "/") for value in config["outputs"].values())
    result = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
