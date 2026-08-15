"""Build the deterministic Stage45 PPARG generalization diagnosis bundle."""

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
    config_path = root / "configs/stage45_pparg_md96_generalization_diagnosis.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    result = json.loads((root / config["outputs"]["result_json"]).read_text(encoding="ascii"))
    audit_path = root / "data/stage45_pparg_md96_generalization_diagnosis_audit.json"
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    if result.get("status") != "stage45_pparg_md96_generalization_diagnosis_complete":
        raise ValueError("Stage45 result is incomplete")
    if audit.get("status") != "stage45_pparg_md96_generalization_diagnosis_independent_audit_ok":
        raise ValueError("Stage45 audit is incomplete")
    paths = {
        "configs/stage45_pparg_md96_generalization_diagnosis.json",
        "scripts/diagnose_stage45_pparg_md96_generalization.py",
        "scripts/audit_stage45_pparg_md96_generalization.py",
        "scripts/build_stage45_pparg_md96_generalization_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/run_stage42d_bace1_large_pool_qubo_screen.py",
        "scripts/run_stage42f_bace1_rank_sensitive_pair_qubo.py",
        "scripts/run_stage44_pparg_md96_rank_sensitive_qubo.py",
        "scripts/__init__.py",
        "tests/test_stage45_pparg_md96_generalization.py",
        "data/stage45_pparg_md96_generalization_diagnosis_audit.json",
        "pyproject.toml",
    }
    paths.update(str(value["path"]).replace("\\", "/") for value in config["inputs"].values())
    paths.update(str(value).replace("\\", "/") for value in config["outputs"].values())
    bundle = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
