"""Build the pose-free Stage32a PPARG MD functional-landscape bundle."""

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
    config_path = root / "configs/stage32a_pparg_md_functional_landscape_analysis.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    result_path = root / config["outputs"]["result_json"]
    audit_path = root / config["outputs"]["audit_json"]
    if json.loads(result_path.read_text(encoding="ascii")).get("status") != "stage32a_pparg_md_functional_landscape_analysis_complete":
        raise ValueError("Stage32a analysis is not complete")
    if json.loads(audit_path.read_text(encoding="ascii")).get("status") != "stage32a_pparg_md_functional_landscape_audit_ok":
        raise ValueError("Stage32a audit is not complete")
    paths = {
        "configs/stage32_pparg_md_functional_complementarity_pilot_executed_v2.json",
        "configs/stage32a_pparg_md_functional_landscape_analysis.json",
        "scripts/analyze_stage32a_pparg_md_functional_landscape.py",
        "scripts/audit_stage32_pparg_md_functional_pilot_matrix.py",
        "scripts/audit_stage32a_pparg_md_functional_landscape.py",
        "scripts/build_stage32a_pparg_md_functional_landscape_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/diagnose_stage19e_cross_target_qubo_v2.py",
        "scripts/__init__.py",
        "tests/test_stage32a_pparg_md_functional_landscape.py",
        "pyproject.toml",
        "data/stage32_pparg_md_functional_pilot_input_preparation_result.json",
        "data/stage32_pparg_md_functional_pilot_matrix_audit.json",
        "data/stage32a_pparg_md_functional_landscape_result.json",
        "data/stage32a_pparg_md_functional_landscape_audit.json",
        "data/processed/stage32_pparg_md_selected16_prepared_receptor_manifest.csv",
        "data/processed/stage32_pparg_train160_ligand_manifest.csv",
    }
    paths.update(str(value).replace("\\", "/") for key, value in config["inputs"].items() if key not in {"executed_config", "input_preparation_result", "prepared_receptor_manifest", "ligand_manifest"})
    paths.update(str(value).replace("\\", "/") for value in config["outputs"].values())
    batch_root = root / "results/runs/stage32_pparg_md_functional_complementarity_pilot/batches"
    paths.update(path.relative_to(root).as_posix() for path in batch_root.glob("*/*/batch_summary.json"))
    result = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
