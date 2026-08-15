"""Build a self-contained Stage 08b replacement-redocking recovery bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from .run_stage08b_mk14_replacement_redocking import (
        file_sha256,
        read_json,
        validate_inputs,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from scripts.experimental.unidock.run_stage08b_mk14_replacement_redocking import (
        file_sha256,
        read_json,
        validate_inputs,
    )


CONFIG = "configs/stage08b_mk14_expanded16_replacement_redocking.json"
FIXED_PATHS = (
    CONFIG,
    "configs/stage08b_mk14_expanded16_replacement_preregistration.json",
    "configs/stage08_mk14_expanded16_redocking_failure_adjudication.json",
    "configs/stage08_mk14_expanded16_unidock_redocking.json",
    "environment/stage08_unidock_gpu.yml",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/batch_prepare_ligand_pdbqt.py",
    "scripts/prepare_receptor.py",
    "scripts/transform_sdf_coordinates.py",
    "scripts/evaluate_redocking_rmsd.py",
    "scripts/select_mk14_rcsb_coordinate_pool.py",
    "scripts/run_mk14_expanded_redocking_gate.py",
    "scripts/select_stage08_mk14_expanded16_pool.py",
    "scripts/select_stage08b_mk14_replacements.py",
    "scripts/adjudicate_stage08_mk14_expanded16_redocking_failure.py",
    "scripts/experimental/unidock/__init__.py",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/run_stage07b_unidock_enhanced_confirmation.py",
    "scripts/experimental/unidock/run_stage07c_unidock_warning_adjudication.py",
    "scripts/experimental/unidock/run_stage08_mk14_expanded16_redocking.py",
    "scripts/experimental/unidock/run_stage08b_mk14_replacement_redocking.py",
    "scripts/experimental/unidock/audit_stage08b_mk14_replacement_redocking.py",
    "scripts/experimental/unidock/run_stage08b_mk14_replacement_remote.sh",
    "reports/stage-08b/mk14_expanded16_replacement_redocking_execution.md",
)


def bundle_paths(root: Path) -> list[str]:
    root = root.resolve()
    config = read_json(root / CONFIG)
    _, replacement_rows, current_rows, _, audit = validate_inputs(root, config)
    if audit["validation_rows"] != 0 or audit["test_rows"] != 0:
        raise ValueError("Stage 08b bundle crossed a data boundary")
    if audit["expected_redocking_pair_count"] != 6:
        raise ValueError("Stage 08b replacement pair count differs")
    paths = list(FIXED_PATHS)
    for descriptor in config["implementation"].values():
        path = str(descriptor["path"]).replace("\\", "/")
        if file_sha256(root / path) != str(descriptor["sha256"]).upper():
            raise ValueError(f"Stage 08b implementation hash differs: {path}")
        paths.append(path)
    for value in config["inputs"].values():
        if isinstance(value, dict) and "path" in value:
            paths.append(str(value["path"]))

    existing_manifest = str(config["inputs"]["existing8_receptor_manifest"]["path"])
    paths.extend(manifest_paths(root, existing_manifest, "receptor_pdbqt"))
    for row in replacement_rows:
        paths.extend((row["pdb_path"], row["aligned_pdb_path"]))
    for row in current_rows[8:]:
        paths.extend((str(row["input_structure"]), str(row["receptor_pdbqt"])))
        receptor_path = root / str(row["receptor_pdbqt"])
        paths.append(
            (receptor_path.parent / "preparation_summary.json")
            .relative_to(root)
            .as_posix()
        )
    return sorted(set(path.replace("\\", "/") for path in paths))


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
            "operation": "Stage 08b deterministic structural-replacement and Uni-Dock recovery bundle",
            "current_admitted_receptor_count": 14,
            "replacement_receptor_ids": [
                "MK14_3ITZ_aligned",
                "MK14_2BAK_aligned",
            ],
            "replacement_receptor_count": 2,
            "seed_count": 3,
            "gpu_redocking_pair_count": 6,
            "final_receptor_count_if_both_pass": 16,
            "validation_rows": 0,
            "test_rows": 0,
            "gpu_required_for_execution": True,
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
