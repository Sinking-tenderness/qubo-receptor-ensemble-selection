"""Build the deterministic Stage41a BACE1 large-pool freeze core bundle."""

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
    config_path = root / "configs/stage41a_bace1_large_pool_freeze.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    result = json.loads((root / config["outputs"]["result_json"]).read_text(encoding="ascii"))
    audit = json.loads((root / config["outputs"]["audit_json"]).read_text(encoding="ascii"))
    if result.get("status") != "stage41a_bace1_large_pool_frozen":
        raise ValueError("Stage41a result is incomplete")
    if audit.get("status") != "stage41a_bace1_large_pool_freeze_audit_ok":
        raise ValueError("Stage41a audit is incomplete")

    paths = {
        "configs/stage41a_bace1_large_pool_freeze.json",
        "scripts/freeze_stage41a_bace1_large_pool.py",
        "scripts/audit_stage41a_bace1_large_pool.py",
        "scripts/build_stage41a_bace1_large_pool_freeze_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/__init__.py",
        "tests/test_stage41a_bace1_large_pool_freeze.py",
        "pyproject.toml",
    }
    paths.update(str(value["path"]).replace("\\", "/") for value in config["inputs"].values())
    paths.update(str(value).replace("\\", "/") for value in config["outputs"].values())
    bundle = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
