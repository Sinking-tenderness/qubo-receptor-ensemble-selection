"""Build the Stage43 PPARG MD-96 remote input bundle."""

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
    config_path = root / "configs/stage43_pparg_md96_rank_sensitive_replication.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    paths = {
        "configs/stage43_pparg_md96_rank_sensitive_replication.json",
        "scripts/prepare_stage43_pparg_md96_inputs.py",
        "scripts/experimental/unidock/run_stage43_pparg_md96_production.py",
        "scripts/experimental/unidock/run_stage43_pparg_md96_production_remote.sh",
        "scripts/audit_stage43_pparg_md96_production.py",
        "scripts/build_stage43_pparg_md96_input_bundle.py",
        "scripts/build_stage43_pparg_md96_result_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/prepare_receptor.py",
        "scripts/__init__.py",
        "scripts/experimental/__init__.py",
        "scripts/experimental/unidock/__init__.py",
        "scripts/experimental/unidock/run_stage09_mk14_train696_production.py",
        "scripts/experimental/unidock/run_unidock_batch_targeted.py",
        "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
        "scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py",
        "scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py",
        "tests/test_stage43_pparg_md96.py",
        "data/processed/stage43_pparg_md96_frame_manifest.csv",
        "data/processed/stage43_pparg_md96_prepared_receptor_manifest.csv",
        "data/stage43_pparg_md96_input_preparation_result.json",
        "reports/stage-43/pparg_md96_execution.md",
    }
    for key, value in config["inputs"].items():
        if key == "seeds":
            continue
        paths.add(str(value).replace("\\", "/"))
    ligand_rows = []
    import csv
    with (root / config["inputs"]["stage32_ligand_manifest"]).open(newline="", encoding="utf-8") as handle:
        ligand_rows = list(csv.DictReader(handle))
    paths.update(row["pdbqt_path"].replace("\\", "/") for row in ligand_rows)
    result = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
