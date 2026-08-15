"""Build the Stage52a PPARA Train-374 ligand-preparation input bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle
from scripts.experimental.unidock.prepare_development_ligand_inputs import (
    read_json,
    validate_source,
)


CONFIG = "configs/stage52a_ppara_train374_unidock_input_preparation.json"
PATHS = (
    "configs/stage52_ppara_posthoc_exploratory_development_preregistration.json",
    CONFIG,
    "data/stage51b_ppara_redocking_bias_diagnostic_result.json",
    "data/stage51b_ppara_redocking_bias_diagnostic_audit.json",
    "data/stage49_ppara_ligand_panel_allocation_summary.json",
    "data/processed/stage49_ppara_train374_ligand_manifest.csv",
    "environment/stage08_unidock_gpu.yml",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/prepare_ligand_3d_sdf.py",
    "scripts/batch_prepare_ligand_pdbqt.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/prepare_development_ligand_inputs.py",
    "scripts/experimental/unidock/run_stage52a_ppara_train374_input_preparation_remote.sh",
    "scripts/experimental/unidock/build_stage52a_ppara_train374_input_bundle.py",
    "reports/stage-52/ppara_train374_input_preparation.md",
    "tests/test_stage52a_ppara_train374_input_preparation.py",
)


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = read_json(root / CONFIG)
    rows, _, labels = validate_source(root, config)
    if len(rows) != 374 or labels != {"active": 187, "decoy": 187}:
        raise ValueError("Stage52a frozen development panel differs")
    if any(
        marker in path.lower()
        for path in PATHS
        for marker in ("fresh_validation", "locked_test", "protected/")
    ):
        raise ValueError("Stage52a bundle crossed a protected split boundary")
    return sorted(set(PATHS))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, bundle_paths(root))
    result.update(
        {
            "operation": "Stage52a PPARA Train-374 checkpointed Uni-Dock ligand input preparation",
            "target_id": "PPARA",
            "analysis_class": "post-hoc exploratory development-only",
            "development_active_count": 187,
            "development_decoy_count": 187,
            "development_ligand_count": 374,
            "future_receptor_count": 20,
            "future_seed_count": 3,
            "future_pair_count": 22440,
            "gpu_docking_jobs_in_this_bundle": 0,
            "fresh_validation_structures_prepared": 0,
            "locked_test_structures_prepared": 0,
            "resume_supported": True,
            "confirmatory_status_changed": False,
        }
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
