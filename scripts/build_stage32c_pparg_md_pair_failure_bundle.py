"""Build the deterministic Stage32c PPARG MD-pair failure diagnostic bundle."""

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
    config_path = root / "configs/stage32c_pparg_md_pair_failure_diagnostic.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    result = json.loads((root / config["outputs"]["result_json"]).read_text(encoding="ascii"))
    audit = json.loads((root / config["outputs"]["audit_json"]).read_text(encoding="ascii"))
    if result.get("status") != "stage32c_pparg_md_pair_failure_diagnostic_complete" or audit.get("status") != "stage32c_pparg_md_pair_failure_diagnostic_audit_ok":
        raise ValueError("Stage32c result or audit is incomplete")
    paths = {
        "configs/stage32c_pparg_md_pair_failure_diagnostic.json",
        "scripts/diagnose_stage32c_pparg_md_pair_failure.py",
        "scripts/audit_stage32c_pparg_md_pair_failure.py",
        "scripts/build_stage32c_pparg_md_pair_failure_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/stage32b_common.py",
        "scripts/__init__.py",
        "tests/test_stage32c_pparg_md_pair_failure.py",
        "pyproject.toml",
    }
    paths.update(str(value).replace("\\", "/") for value in config["inputs"].values())
    paths.update(str(value).replace("\\", "/") for value in config["outputs"].values())
    bundle = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
