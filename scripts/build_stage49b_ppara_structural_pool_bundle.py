"""Build the deterministic Stage49b PPARA structural-pool core bundle."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_stage05_mk14_remote_bundle import write_bundle


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bundle_paths(root: Path) -> list[str]:
    summary = json.loads(
        (root / "data/stage49b_ppara_structural_selection_summary.json").read_text(
            encoding="ascii"
        )
    )
    audit = json.loads(
        (
            root
            / "data/stage49b_ppara_structural_selection_independent_audit.json"
        ).read_text(encoding="ascii")
    )
    if summary.get("status") != "stage49b_ppara_structural_pool_ok":
        raise ValueError("Stage49b structural-pool result is incomplete")
    if audit.get("status") != "stage49b_ppara_structural_pool_independent_audit_ok":
        raise ValueError("Stage49b independent audit is incomplete")
    selected_path = root / summary["artifacts"]["selected_redocking_manifest_csv"]["path"]
    selected = read_csv(selected_path)
    if len(selected) != 64:
        raise ValueError("Stage49b bundle selected count differs")

    paths = {
        "configs/stage49_ppara_ligand_panel_allocation.json",
        "configs/stage49b_ppara_structural_selection.json",
        "data/stage48_ppara_source_audit.json",
        "data/stage48_ppara_source_audit_independent.json",
        "data/stage49_ppara_ligand_panel_allocation_summary.json",
        "data/stage49b_ppara_structural_selection_summary.json",
        "data/stage49b_ppara_structural_selection_independent_audit.json",
        "data/stage47b_expanded_new_target_feasibility_screen_result.json",
        "data/processed/stage47b_expanded_new_target_candidate_metadata.csv",
        "data/processed/stage49_ppara_selected_ligand_panel_manifest.csv",
        "data/processed/stage49_ppara_train374_ligand_manifest.csv",
        "reports/stage-49b/ppara_structural_selection.md",
        "scripts/__init__.py",
        "scripts/allocate_stage49_ppara_ligand_panels.py",
        "scripts/audit_stage49b_ppara_structural_pool.py",
        "scripts/build_stage05_mk14_remote_bundle.py",
        "scripts/build_stage49b_ppara_structural_pool_bundle.py",
        "scripts/select_mk14_rcsb_coordinate_pool.py",
        "scripts/select_stage13_egfr_coordinate_pool.py",
        "scripts/select_stage49b_ppara_structural_pool.py",
        "tests/test_stage49_ppara_intake.py",
        "pyproject.toml",
    }
    paths.update(record["path"] for record in summary["artifacts"].values())
    for row in selected:
        paths.add(row["mmcif_path"])
        paths.add(row["aligned_protein_pdb_path"])
    normalized = sorted(path.replace("\\", "/") for path in paths)
    forbidden = ("fresh_validation", "locked_test", "stage11_mk14")
    if any(marker in path.lower() for path in normalized for marker in forbidden):
        raise ValueError("Stage49b bundle contains a protected path")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = write_bundle(root, args.output, bundle_paths(root))
    result.update(
        {
            "operation": "Stage49 PPARA ligand allocation and Stage49b structural-pool freeze",
            "target_id": "PPARA",
            "coordinate_eligible_count": 66,
            "selected_receptor_count": 64,
            "new_docking_jobs": 0,
            "protected_validation_files_included": 0,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
