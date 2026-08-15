"""Build the Stage32 PPARG MD functional-pilot input bundle."""

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
    config_path = root / "configs/stage32_pparg_md_functional_complementarity_pilot.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    selected_ligands = root / config["outputs"]["selected_ligand_manifest"]
    import csv
    with selected_ligands.open("r", encoding="utf-8", newline="") as handle:
        ligand_rows = list(csv.DictReader(handle))
    paths = {
        "configs/stage32_pparg_md_functional_complementarity_pilot.json",
        "scripts/prepare_stage32_pparg_md_functional_pilot.py",
        "scripts/experimental/unidock/run_stage32_pparg_md_functional_pilot.py",
        "scripts/experimental/unidock/run_stage09_mk14_train696_production.py",
        "scripts/experimental/unidock/run_unidock_batch_targeted.py",
        "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
        "scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py",
        "scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py",
        "scripts/run_stage32_pparg_md_functional_pilot_remote.sh",
        "scripts/build_stage32_pparg_md_functional_pilot_input_bundle.py",
        "scripts/build_stage32_pparg_md_functional_pilot_result_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/prepare_receptor.py",
        "scripts/__init__.py",
        "scripts/experimental/__init__.py",
        "scripts/experimental/unidock/__init__.py",
        "tests/test_stage32_pparg_md_functional_pilot.py",
        "data/processed/stage32_pparg_md_selected16_frame_manifest.csv",
        "data/processed/stage32_pparg_train160_ligand_manifest.csv",
        "data/stage32_pparg_md_functional_pilot_input_preparation_result.json",
    }
    paths.update(str(value).replace("\\", "/") for value in config["inputs"].values())
    paths.update(row["pdbqt_path"].replace("\\", "/") for row in ligand_rows)
    result = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
