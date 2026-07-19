"""Build a deterministic Linux bundle for targeted MAPK14 e64 diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .build_stage05_mk14_remote_bundle import write_bundle
except ImportError:
    from build_stage05_mk14_remote_bundle import write_bundle


CONFIG_PATH = "configs/stage05_mk14_expanded_e64_consensus_diagnostics.json"
FIXED_PATHS = (
    CONFIG_PATH,
    "configs/stage05_mk14_expanded_train_search_protocol_amendment01.json",
    "configs/stage05_mk14_expanded_train_e32_cpu2.txt",
    "configs/stage05_mk14_expanded_train_e64_cpu2.txt",
    "data/processed/stage05_mk14_expanded8_receptor_manifest.csv",
    "data/processed/stage05_mk14_expanded_train_80a80d_pdbqt_manifest.csv",
    "data/processed/stage05_mk14_expanded_e32_matrix_flagged_pairs.csv",
    "data/stage05_mk14_expanded_e32_matrix_admission_summary.json",
    "results/runs/stage05_mk14_expanded8_train160_e32_aggregated/aggregated_seed_scores.csv",
    "results/runs/stage05_mk14_expanded8_train160_e32_aggregated/summary.json",
    "environment/bin/vina_1.2.7_linux_x86_64",
    "scripts/aggregate_seed_replicates.py",
    "scripts/audit_stage05_development_matrix.py",
    "scripts/audit_stage05_expanded_train_matrix.py",
    "scripts/audit_stage05_expanded_e32_matrix.py",
    "scripts/batch_vina_docking.py",
    "scripts/prepare_receptor.py",
    "scripts/run_vina_warning_diagnostics.py",
    "scripts/run_stage05_mk14_expanded_e64_consensus_diagnostics.py",
    "scripts/run_stage05_mk14_expanded_e64_consensus_remote.sh",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def selected_input_paths(root: Path) -> list[str]:
    config = json.loads((root / CONFIG_PATH).read_text(encoding="ascii"))
    inputs = config["inputs"]
    flagged = read_csv(root / str(inputs["e32_flagged_pairs"]))
    if len(flagged) != int(config["selection"]["expected_case_count"]):
        raise ValueError("e64 bundle flagged case count differs")
    receptors = {
        row["conformer_id"]: row
        for row in read_csv(root / str(inputs["receptor_manifest"]))
    }
    ligands = {
        row["ligand_id"]: row
        for row in read_csv(root / str(inputs["ligand_manifest"]))
    }
    paths = []
    seen: set[tuple[str, str]] = set()
    for row in flagged:
        key = (row["ligand_id"], row["receptor_id"])
        if key in seen:
            raise ValueError("e64 bundle flags contain a duplicate pair")
        seen.add(key)
        if key[0] not in ligands or key[1] not in receptors:
            raise ValueError(f"e64 bundle flag is absent from a manifest: {key}")
        ligand = ligands[key[0]]
        if ligand.get("selection_role") != "development_train":
            raise ValueError(f"non-train ligand entered e64 bundle: {key[0]}")
        paths.append(ligand["pdbqt_path"].replace("\\", "/"))
        paths.append(receptors[key[1]]["receptor_pdbqt"].replace("\\", "/"))
    return sorted(set(paths))


def bundle_paths(root: Path) -> list[str]:
    return sorted(set(FIXED_PATHS) | set(selected_input_paths(root)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_bundle(args.root, args.output, bundle_paths(args.root))
    result["diagnostic_case_count"] = 14
    result["expected_vina_runs"] = 42
    result["validation_rows"] = 0
    result["test_rows"] = 0
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
