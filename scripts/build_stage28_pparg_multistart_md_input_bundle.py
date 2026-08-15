"""Build the deterministic Stage28 remote MD input bundle."""

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
    parser.add_argument("--config", type=Path, default=Path("configs/stage28_pparg_multistart_md_ensemble.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = root / args.config
    config = json.loads(config_path.read_text(encoding="ascii"))
    preparation = json.loads((root / config["outputs"]["preparation_result_json"]).read_text(encoding="ascii"))
    starts = []
    import csv
    with (root / config["runtime"]["start_manifest"]).open("r", encoding="utf-8", newline="") as handle:
        starts = list(csv.DictReader(handle))
    paths = {
        config_path.relative_to(root).as_posix(),
        config["target"]["source_selection_manifest"],
        config["runtime"]["start_manifest"],
        config["outputs"]["preparation_result_json"],
        "environment/stage03_openmm.yml",
        "scripts/prepare_stage28_pparg_multistart_md_inputs.py",
        "scripts/run_stage28_pparg_multistart_md.py",
        "scripts/collect_stage28_pparg_md_ensemble.py",
        "scripts/audit_stage28_pparg_multistart_md_ensemble.py",
        "scripts/build_stage28_pparg_multistart_md_input_bundle.py",
        "scripts/build_stage28_pparg_multistart_md_result_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/build_openmm_system.py",
        "scripts/run_openmm_equilibration_smoke.py",
        "scripts/run_openmm_equilibration.py",
        "scripts/run_openmm_production.py",
        "scripts/analyze_md_trajectory.py",
        "scripts/cluster_md_pocket_frames.py",
        "scripts/run_stage21_structure_aware_qubo.py",
        "scripts/run_stage28_pparg_multistart_md_remote.sh",
        "scripts/run_stage28b_pparg_md_ready_remote.sh",
        "scripts/__init__.py",
        "tests/test_stage28_pparg_multistart_md_inputs.py",
        "tests/test_stage28b_pparg_md_ready_starts.py",
        "reports/stage-28/pparg_multistart_md_remote_execution.md",
        "pyproject.toml",
    }
    for value in config.get("selection_provenance_files", []):
        paths.add(value)
    for item in preparation["generated_configs"]:
        paths.add(item["path"])
    for row in starts:
        paths.add(row["starting_pdb"])
    result = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
