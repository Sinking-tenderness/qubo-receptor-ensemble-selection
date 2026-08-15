"""Build the locked-test-free Stage32b PPARG fresh-validation input bundle."""

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
    config_path = root / "configs/stage32b_pparg_md_pair_fresh_validation.json"
    config = json.loads(config_path.read_text(encoding="ascii"))
    selection = json.loads((root / config["outputs"]["train_selection_json"]).read_text(encoding="ascii"))
    preparation = json.loads((root / config["outputs"]["preparation_result"]).read_text(encoding="ascii"))
    if selection.get("status") != "stage32b_pparg_md_pair_train_selection_frozen":
        raise ValueError("Stage32b train selection is not frozen")
    if preparation.get("status") != "stage32b_validation_inputs_frozen_awaiting_remote_preparation":
        raise ValueError("Stage32b validation identities are not frozen")
    paths = {
        "configs/stage32b_pparg_md_pair_fresh_validation.json",
        "scripts/select_stage32b_pparg_md_pair.py",
        "scripts/freeze_stage32b_pparg_fresh_validation_inputs.py",
        "scripts/prepare_stage32b_pparg_fresh_validation.py",
        "scripts/evaluate_stage32b_pparg_md_pair_fresh_validation.py",
        "scripts/experimental/unidock/run_stage32b_pparg_md_pair_fresh_validation.py",
        "scripts/experimental/unidock/run_stage09_mk14_train696_production.py",
        "scripts/experimental/unidock/run_unidock_batch_targeted.py",
        "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
        "scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py",
        "scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py",
        "scripts/run_stage32b_pparg_md_pair_fresh_validation_remote.sh",
        "scripts/build_stage32b_pparg_md_pair_fresh_validation_input_bundle.py",
        "scripts/build_stage32b_pparg_md_pair_fresh_validation_result_bundle.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/stage32b_common.py",
        "scripts/batch_prepare_ligand_pdbqt.py",
        "scripts/prepare_ligand_3d_sdf.py",
        "scripts/__init__.py",
        "scripts/experimental/__init__.py",
        "scripts/experimental/unidock/__init__.py",
        "tests/test_stage32b_pparg_md_pair_fresh_validation.py",
        "pyproject.toml",
        config["inputs"]["stage32a_audit"],
        config["inputs"]["stage32_scores_csv"],
        config["inputs"]["stage32_median_matrix_csv"],
        config["inputs"]["stage32_minimum_matrix_csv"],
        config["inputs"]["stage32_train_ligand_manifest"],
        config["inputs"]["stage32_prepared_receptor_manifest"],
        config["inputs"]["stage19a_allocation_summary"],
        config["inputs"]["stage07c_profile_freeze"],
        config["outputs"]["train_selection_json"],
        config["outputs"]["fresh_validation_source_manifest"],
        config["outputs"]["selected_receptor_manifest"],
        config["outputs"]["preparation_result"],
    }
    if any("locked" in path.lower() or "test" in path.lower() and not path.startswith("tests/") for path in paths):
        raise ValueError("Stage32b input bundle path set crossed the locked-test boundary")
    result = write_bundle(root, args.output, sorted(paths))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
