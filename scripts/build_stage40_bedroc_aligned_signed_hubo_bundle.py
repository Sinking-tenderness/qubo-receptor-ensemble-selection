"""Build the deterministic Stage40 BEDROC-aligned signed-HUBO core bundle."""

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
    config_path = root / "configs/stage40_bedroc_aligned_signed_hubo.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    result = json.loads((root / config["outputs"]["result_json"]).read_text(encoding="ascii"))
    audit = json.loads((root / config["outputs"]["audit_json"]).read_text(encoding="ascii"))
    if result.get("status") != "stage40_bedroc_aligned_signed_hubo_complete":
        raise ValueError("Stage40 result is incomplete")
    if audit.get("status") != "stage40_bedroc_aligned_signed_hubo_audit_ok":
        raise ValueError("Stage40 audit is incomplete")
    paths = {
        "configs/stage40_bedroc_aligned_signed_hubo.json",
        "scripts/run_stage40_bedroc_aligned_signed_hubo.py",
        "scripts/audit_stage40_bedroc_aligned_signed_hubo.py",
        "scripts/build_stage40_bedroc_aligned_signed_hubo_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/diagnose_stage19e_cross_target_qubo_v2.py",
        "scripts/run_stage05_mk14_method_gate.py",
        "scripts/run_stage37_cross_target_robust_functional_qubo.py",
        "scripts/run_stage38_cross_target_stable_triplet_hubo.py",
        "scripts/run_stage39_linear_backbone_residual_gate.py",
        "scripts/screen_stage10_mk14_expanded16_qubo_greedy.py",
        "scripts/prepare_receptor.py",
        "scripts/__init__.py",
        "tests/test_stage40_bedroc_aligned_signed_hubo.py",
        "pyproject.toml",
    }
    paths.update(str(value["path"]).replace("\\", "/") for value in config["inputs"].values())
    paths.update(str(value).replace("\\", "/") for value in config["outputs"].values())
    bundle = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
