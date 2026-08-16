"""Build the deterministic Stage 6 Vina-GPU batch-bridge input bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from .run_vinagpu_deterministic_batch import read_json, validate_bridge_inputs
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from run_vinagpu_deterministic_batch import read_json, validate_bridge_inputs


CONFIG = "configs/stage06_mk14_vinagpu21_deterministic_batch_bridge.json"
FIXED_PATHS = (
    CONFIG,
    "data/stage06_mk14_vinagpu21_train160_v1_gpu_reference.csv",
    "data/stage06_mk14_vinagpu21_train160_v1_result_summary.json",
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/experimental/__init__.py",
    "scripts/experimental/README.md",
    "scripts/experimental/vinagpu/__init__.py",
    "scripts/experimental/vinagpu/README.md",
    "scripts/experimental/vinagpu/run_vinagpu_equivalence.py",
    "scripts/experimental/vinagpu/apply_deterministic_batch_patch.py",
    "scripts/experimental/vinagpu/run_vinagpu_deterministic_batch.py",
    "scripts/experimental/vinagpu/package_vinagpu_deterministic_batch_results.py",
    "scripts/experimental/vinagpu/run_vinagpu_deterministic_batch_remote.sh",
    "reports/stage-06/mk14_vinagpu21_deterministic_batch_bridge.md",
    "reports/stage-06/mk14_vinagpu21_train160_result.md",
)


def bundle_paths(root: Path) -> list[str]:
    config = read_json(root / CONFIG)
    _, _, _, audit = validate_bridge_inputs(root.resolve(), config)
    if audit["validation_rows"] != 0 or audit["test_rows"] != 0:
        raise ValueError("deterministic batch bridge crossed a data boundary")
    paths = list(FIXED_PATHS)
    receptor_manifest = str(config["inputs"]["receptor_manifest"]["path"])
    ligand_manifest = str(config["inputs"]["ligand_manifest"]["path"])
    paths.extend(manifest_paths(root, receptor_manifest, "receptor_pdbqt"))
    paths.extend(manifest_paths(root, ligand_manifest, "pdbqt_path"))
    paths.extend((receptor_manifest, ligand_manifest))
    paths.extend(
        str(seed[key])
        for seed in config["inputs"]["cpu_seed_runs"]
        for key in ("summary_path", "scores_path")
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
            "operation": "consumed Train-160 deterministic Vina-GPU batch bridge input bundle",
            "receptor_count": 5,
            "ligand_count": 160,
            "seed_count": 3,
            "pair_count": 2400,
            "chunk_size": 8,
            "chunk_count": 300,
            "validation_rows": 0,
            "test_rows": 0,
            "vinagpu_source_or_binary_included": False,
        }
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
