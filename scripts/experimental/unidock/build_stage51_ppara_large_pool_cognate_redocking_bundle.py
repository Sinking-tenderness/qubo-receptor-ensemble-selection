"""Build the deterministic PPARA large-pool cognate-redocking GPU bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle
from scripts.experimental.unidock.run_stage51_ppara_large_pool_cognate_redocking import (
    common,
    validate_inputs,
)


CONFIG = "configs/stage51_ppara_large_pool_cognate_redocking.json"
FIXED_PATHS = (
    CONFIG,
    "configs/stage50_ppara_large_pool_redocking_input_preparation.json",
    "data/stage50a_ppara_input_preparation_runtime_adjudication.json",
    "data/stage50_ppara_large_pool_inputs_independent_audit.json",
    "data/stage51a_ppara_bundle_import_closure_adjudication.json",
    "environment/stage08_unidock_gpu.yml",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/audit_stage50_ppara_large_pool_inputs.py",
    "scripts/evaluate_redocking_rmsd.py",
    "scripts/prepare_receptor.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py",
    "scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py",
    "scripts/experimental/unidock/run_stage14d_fa10_cognate_redocking.py",
    "scripts/experimental/unidock/run_stage51_ppara_large_pool_cognate_redocking.py",
    "scripts/experimental/unidock/run_stage51_ppara_large_pool_cognate_redocking_remote.sh",
    "scripts/experimental/unidock/build_stage51_ppara_large_pool_cognate_redocking_bundle.py",
    "reports/stage-50/ppara_large_pool_input_preparation_remote.md",
    "reports/stage-51/ppara_large_pool_cognate_redocking_execution.md",
    "tests/test_stage50_51_ppara_large_pool.py",
)


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = common.read_json(root / CONFIG)
    receptors, cases, audit = validate_inputs(root, config)
    if audit["expected_redocking_pair_count"] != 180:
        raise ValueError("Stage51 bundle pair count differs")
    if any(
        int(audit[key]) != 0
        for key in (
            "ligand_labels_read",
            "benchmark_docking_scores_read",
            "fresh_validation_rows_read",
            "test_rows_read",
        )
    ):
        raise ValueError("Stage51 bundle crossed a data boundary")
    paths = list(FIXED_PATHS)
    for value in dict(config["inputs"]).values():
        if isinstance(value, dict) and "path" in value:
            paths.append(str(value["path"]))
    paths.extend(row["receptor_pdbqt"] for row in receptors)
    for row in cases:
        paths.extend((row["ligand_pdbqt"], row["reference_sdf"]))
    normalized = sorted(set(path.replace("\\", "/") for path in paths))
    forbidden = ("fresh_validation", "locked_test", "stage09_mk14", "stage11_mk14")
    if any(marker in path.lower() for path in normalized for marker in forbidden):
        raise ValueError("Stage51 bundle contains a forbidden result path")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, bundle_paths(root))
    result.update({
        "operation": "Stage51 PPARA 60-prepared-receptor three-seed cognate redocking bundle",
        "target_id": "PPARA",
        "frozen_receptor_count": 64,
        "prepared_receptor_count": 60,
        "technical_preparation_failure_count": 4,
        "cognate_ligand_count": 60,
        "seed_count": 3,
        "gpu_batch_count": 180,
        "gpu_pair_count": 180,
        "profile_id": "enhanced",
        "exhaustiveness": 1024,
        "max_step": 80,
        "ligand_labels_read": 0,
        "benchmark_docking_scores_read": 0,
        "fresh_validation_rows_read": 0,
        "test_rows_read": 0,
        "gpu_required_for_execution": True,
    })
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
