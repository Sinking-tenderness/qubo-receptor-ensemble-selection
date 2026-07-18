"""Build a deterministic Linux bundle for expanded-matrix e32 diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .build_stage05_mk14_remote_bundle import write_bundle
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle


CONFIG_PATH = "configs/stage05_mk14_expanded_train_matrix_e32_diagnostics.json"
FIXED_PATHS = (
    CONFIG_PATH,
    "configs/stage05_mk14_expanded_train_docking_preregistration.json",
    "configs/stage05_mk14_expanded_train_e16_cpu2.txt",
    "configs/stage05_mk14_expanded_train_e32_cpu2.txt",
    "configs/stage05_mk14_search_ladder_multiseed_followup.json",
    "data/processed/stage05_mk14_expanded8_receptor_manifest.csv",
    "data/processed/stage05_mk14_expanded_train_80a80d_pdbqt_manifest.csv",
    "data/processed/stage05_mk14_expanded_train_matrix_flagged_pairs.csv",
    "data/stage05_mk14_expanded_train_matrix_admission_summary.json",
    "results/runs/stage05_mk14_expanded8_train160_e16_aggregated/aggregated_seed_scores.csv",
    "results/runs/stage05_mk14_expanded8_train160_e16_aggregated/summary.json",
    "environment/bin/vina_1.2.7_linux_x86_64",
    "scripts/audit_stage05_e32_matrix_rescue.py",
    "scripts/batch_vina_docking.py",
    "scripts/prepare_receptor.py",
    "scripts/run_vina_warning_diagnostics.py",
    "scripts/run_stage05_mk14_expanded_matrix_diagnostics.py",
    "scripts/run_stage05_mk14_expanded_matrix_diagnostics_remote.sh",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    return rows


def selected_manifest_paths(root: Path, config_relative: str = CONFIG_PATH) -> list[str]:
    config = json.loads((root / config_relative).read_text(encoding="ascii"))
    expected_cases = config["expected_cases"]
    if not isinstance(expected_cases, list) or not expected_cases:
        raise ValueError("diagnostic config has no expected cases")
    inputs = config["inputs"]
    receptor_rows = read_csv(root / str(inputs["receptor_manifest"]))
    ligand_rows = read_csv(root / str(inputs["ligand_manifest"]))
    receptors = {row["conformer_id"]: row for row in receptor_rows}
    ligands = {row["ligand_id"]: row for row in ligand_rows}
    paths: list[str] = []
    pair_keys: set[tuple[str, str]] = set()
    for case in expected_cases:
        ligand_id = str(case["ligand_id"])
        receptor_id = str(case["receptor_id"])
        key = (ligand_id, receptor_id)
        if key in pair_keys:
            raise ValueError("diagnostic cases contain a duplicate pair")
        pair_keys.add(key)
        if receptor_id not in receptors or ligand_id not in ligands:
            raise ValueError(f"diagnostic case is absent from a manifest: {key}")
        receptor = receptors[receptor_id]
        ligand = ligands[ligand_id]
        if ligand.get("selection_role") != "development_train":
            raise ValueError(f"non-train ligand entered diagnostic bundle: {ligand_id}")
        paths.append(receptor["receptor_pdbqt"].replace("\\", "/"))
        paths.append(ligand["pdbqt_path"].replace("\\", "/"))
    return sorted(set(paths))


def diagnostic_bundle_paths(root: Path) -> list[str]:
    return sorted(set(FIXED_PATHS) | set(selected_manifest_paths(root)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_bundle(
        args.root,
        args.output,
        diagnostic_bundle_paths(args.root),
    )
    result["diagnostic_case_count"] = 7
    result["expected_vina_runs"] = 21
    result["validation_rows"] = 0
    result["test_rows"] = 0
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
