"""Build the deterministic Stage 6 Vina-GPU Train-160 input bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from .run_vinagpu_equivalence import read_json, validate_inputs
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from run_vinagpu_equivalence import read_json, validate_inputs


CONFIG = "configs/stage06_mk14_vinagpu21_train160_equivalence.json"
FIXED_PATHS = (
    CONFIG,
    "scripts/build_stage05_mk14_remote_bundle.py",
    "scripts/experimental/__init__.py",
    "scripts/experimental/README.md",
    "scripts/experimental/vinagpu/__init__.py",
    "scripts/experimental/vinagpu/README.md",
    "scripts/experimental/vinagpu/run_vinagpu_equivalence.py",
    "scripts/experimental/vinagpu/audit_vinagpu_equivalence.py",
    "scripts/experimental/vinagpu/package_vinagpu_results.py",
    "scripts/experimental/vinagpu/run_vinagpu_train160_remote.sh",
    "reports/stage-06/mk14_vinagpu21_train160_preregistration.md",
)


def bundle_paths(root: Path) -> list[str]:
    config = read_json(root / CONFIG)
    _, _, audit = validate_inputs(root.resolve(), config)
    if audit["validation_rows"] != 0 or audit["test_rows"] != 0:
        raise ValueError("Vina-GPU input bundle crossed a data boundary")
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
            "operation": "consumed Train-160 AutoDock Vina-GPU 2.1 equivalence input bundle",
            "receptor_count": 5,
            "ligand_count": 160,
            "seed_count": 3,
            "gpu_pair_count": 2400,
            "seed_policy": "base_seed plus ligand seed_offset; one process per pair",
            "validation_rows": 0,
            "test_rows": 0,
            "fresh_validation_scores_included": False,
            "vinagpu_binary_included": False,
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
