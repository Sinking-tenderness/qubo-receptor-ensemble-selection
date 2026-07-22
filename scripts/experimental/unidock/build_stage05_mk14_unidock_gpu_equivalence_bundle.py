"""Build the deterministic train-only MAPK14 Uni-Dock GPU pilot bundle."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    from scripts.build_stage05_mk14_remote_bundle import (
        manifest_paths,
        write_bundle,
    )
    from .run_unidock_gpu_equivalence import read_json, validate_inputs
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_stage05_mk14_remote_bundle import manifest_paths, write_bundle
    from run_unidock_gpu_equivalence import read_json, validate_inputs


CONFIG = "configs/stage05_mk14_unidock_train160_gpu_equivalence.json"
RECEPTOR_MANIFEST = (
    "data/processed/stage05_mk14_unidock_train160_receptor_manifest.csv"
)
LIGAND_MANIFEST = (
    "data/processed/stage05_mk14_expanded_train_80a80d_pdbqt_manifest.csv"
)
FIXED_PATHS = (
    CONFIG,
    RECEPTOR_MANIFEST,
    LIGAND_MANIFEST,
    "environment/stage05_unidock_gpu.yml",
    "scripts/experimental/unidock/run_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/audit_unidock_gpu_equivalence.py",
    "scripts/experimental/unidock/run_stage05_mk14_unidock_gpu_equivalence_remote.sh",
    "reports/stage-05/mk14_unidock_gpu_equivalence_pilot.md",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def bundle_paths(root: Path) -> list[str]:
    config = read_json(root / CONFIG)
    _, _, audit = validate_inputs(root.resolve(), config)
    if audit["validation_rows"] != 0 or audit["test_rows"] != 0:
        raise ValueError("GPU pilot input audit crossed a data boundary")
    paths = list(FIXED_PATHS)
    paths.extend(manifest_paths(root, RECEPTOR_MANIFEST, "receptor_pdbqt"))
    paths.extend(manifest_paths(root, LIGAND_MANIFEST, "pdbqt_path"))
    for seed in config["inputs"]["cpu_seed_runs"]:
        paths.extend([str(seed["summary_path"]), str(seed["scores_path"])])
    return sorted(set(path.replace("\\", "/") for path in paths))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    config = read_json(root / CONFIG)
    _, _, audit = validate_inputs(root, config)
    result = write_bundle(root, args.output, bundle_paths(root))
    result.update(
        {
            "experiment_id": config["experiment_id"],
            "operation": "train-only Uni-Dock GPU equivalence pilot bundle",
            "receptor_count": audit["receptor_count"],
            "ligand_count": audit["ligand_count"],
            "seed_count": audit["seed_count"],
            "gpu_pair_count": audit["gpu_pair_count"],
            "validation_rows": 0,
            "test_rows": 0,
            "fresh_validation_scores_included": False,
            "gpu_required_for_execution": True,
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
