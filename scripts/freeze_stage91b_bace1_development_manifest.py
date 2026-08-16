from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import file_sha256  # noqa: F401 (deduped)
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path




def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def run(root: Path) -> dict[str, object]:
    root = root.resolve()
    result_path = root / "data/stage91_bace1_group_robust_rescue_preregistration_result.json"
    source_path = root / "data/processed/stage91_bace1_chembl_assay_role_ligand_manifest.csv"
    output_path = root / "data/processed/stage91b_bace1_chembl365_development_ligand_manifest.csv"
    summary_path = root / "data/stage91b_bace1_development_manifest_freeze.json"

    result = read_json(result_path)
    if result.get("status") != "stage91_bace1_group_robust_rescue_preregistered":
        raise ValueError("Stage91 preregistration did not pass")
    if file_sha256(source_path) != str(
        result["outputs"]["ligand_manifest_csv"]["sha256"]
    ).upper():
        raise ValueError("Stage91 source manifest identity differs")

    with source_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = [row for row in source_rows if row["role"] == "development"]
    labels = Counter(row["potency_label"] for row in rows)
    expected = result["role_summary"]["development"]
    if len(rows) != int(expected["molecule_count"]):
        raise ValueError("development molecule count differs")
    if labels != Counter(high=248, low=52, gray=65):
        raise ValueError("development potency-label counts differ")
    if any(str(row["docking_authorized"]).lower() != "true" for row in rows):
        raise ValueError("development manifest contains an unauthorized row")
    if len({row["ligand_id"] for row in rows}) != len(rows):
        raise ValueError("development ligand IDs are not unique")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": "1.0",
        "status": "stage91b_bace1_development_manifest_frozen",
        "source": {
            "path": source_path.relative_to(root).as_posix(),
            "sha256": file_sha256(source_path),
            "row_count": len(source_rows),
        },
        "output": {
            "path": output_path.relative_to(root).as_posix(),
            "sha256": file_sha256(output_path),
            "row_count": len(rows),
        },
        "potency_label_counts": dict(sorted(labels.items())),
        "roles_present": sorted({row["role"] for row in rows}),
        "excluded_nondevelopment_row_count": len(source_rows) - len(rows),
        "data_boundary": {
            "confirmation_rows_exported": 0,
            "locked_test_rows_exported": 0,
            "docking_scores_read": 0,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
