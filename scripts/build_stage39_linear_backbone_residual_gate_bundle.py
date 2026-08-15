"""Build the deterministic Stage39 gated-residual core bundle."""

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
    config_path = root / "configs/stage39_linear_backbone_residual_gate.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    result = json.loads((root / config["outputs"]["result_json"]).read_text(encoding="ascii"))
    audit = json.loads((root / config["outputs"]["audit_json"]).read_text(encoding="ascii"))
    if result.get("status") != "stage39_linear_backbone_residual_gate_complete":
        raise ValueError("Stage39 result is incomplete")
    if audit.get("status") != "stage39_linear_backbone_residual_gate_audit_ok":
        raise ValueError("Stage39 audit is incomplete")
    paths = {
        "configs/stage39_linear_backbone_residual_gate.json",
        "scripts/run_stage39_linear_backbone_residual_gate.py",
        "scripts/audit_stage39_linear_backbone_residual_gate.py",
        "scripts/build_stage39_linear_backbone_residual_gate_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/diagnose_stage19e_cross_target_qubo_v2.py",
        "scripts/run_stage05_mk14_method_gate.py",
        "scripts/run_stage37_cross_target_robust_functional_qubo.py",
        "scripts/run_stage38_cross_target_stable_triplet_hubo.py",
        "scripts/screen_stage10_mk14_expanded16_qubo_greedy.py",
        "scripts/prepare_receptor.py",
        "scripts/__init__.py",
        "tests/test_stage39_linear_backbone_residual_gate.py",
        "pyproject.toml",
    }
    paths.update(str(value["path"]).replace("\\", "/") for value in config["inputs"].values())
    paths.update(str(value).replace("\\", "/") for value in config["outputs"].values())
    bundle = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
