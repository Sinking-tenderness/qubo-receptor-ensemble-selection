"""Create a non-destructive PPARD target-id amendment for Stage61b scores."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json  # noqa: F401 (deduped)
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()




def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verified(root: Path, value: dict[str, Any]) -> Path:
    path = root / str(value["path"])
    if not path.is_file() or sha256(path) != str(value["sha256"]).upper():
        raise ValueError(f"Stage61c input identity differs: {path}")
    return path


def row_fingerprint(row: dict[str, str]) -> str:
    payload = json.dumps(
        {key: value for key, value in row.items() if key != "target_id"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = verified(root, dict(config["implementation"])["runner"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage61c implementation path differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    matrix_audit = read_json(inputs["matrix_audit"])
    summary = read_json(inputs["production_summary"])
    amendment = read_json(inputs["progress_descriptor_amendment"])
    if matrix_audit.get("status") != (
        "independent_stage61b_ppard_remaining144_unidock_matrix_audit_ok"
    ):
        raise ValueError("Stage61b independent audit did not pass")
    if summary.get("status") != "stage61b_ppard_remaining144_unidock_matrix_ok":
        raise ValueError("Stage61b production summary did not pass")
    if amendment.get("status") != "stage61b_progress_descriptor_amendment01_ok":
        raise ValueError("Stage61b progress descriptor recovery did not pass")
    if (
        int(matrix_audit["pair_count"]) != 12528
        or int(matrix_audit["pose_integrity_failure_count"]) != 0
        or int(matrix_audit["unresolved_warning_event_count"]) != 0
    ):
        raise ValueError("Stage61b technical matrix dimensions differ")

    rows = read_csv(inputs["raw_scores"])
    receptors = read_csv(inputs["receptor_manifest"])
    ligands = read_csv(inputs["ligand_manifest"])
    expected = dict(config["expected"])
    if len(rows) != int(expected["row_count"]):
        raise ValueError("Stage61c row count differs")
    if {row["target_id"] for row in rows} != {expected["observed_target_id"]}:
        raise ValueError("Stage61c observed target-id defect differs")
    receptor_ids = [row["conformer_id"] for row in receptors]
    ligand_ids = [row["ligand_id"] for row in ligands]
    if set(row["receptor_id"] for row in rows) != set(receptor_ids) or not all(
        value.startswith("PPARD_") for value in receptor_ids
    ):
        raise ValueError("Stage61c receptor identities differ")
    if set(row["ligand_id"] for row in rows) != set(ligand_ids) or not all(
        value.startswith("PPARD_") for value in ligand_ids
    ):
        raise ValueError("Stage61c ligand identities differ")
    keys = {(row["seed_id"], row["receptor_id"], row["ligand_id"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("Stage61c source keys are not unique")

    before = [row_fingerprint(row) for row in rows]
    amended = [dict(row, target_id=str(expected["corrected_target_id"])) for row in rows]
    if [row_fingerprint(row) for row in amended] != before:
        raise ValueError("Stage61c changed non-target metadata")
    changed_fields = sum(
        sum(source[key] != target[key] for key in source)
        for source, target in zip(rows, amended, strict=True)
    )
    if changed_fields != len(rows):
        raise ValueError("Stage61c did not change exactly one field per row")

    outputs = dict(config["outputs"])
    output_path = root / str(outputs["corrected_scores_csv"])
    result_path = root / str(outputs["result_json"])
    if not overwrite and (output_path.exists() or result_path.exists()):
        raise FileExistsError("Stage61c outputs exist; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(amended[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(amended)
    reread = read_csv(output_path)
    if [row_fingerprint(row) for row in reread] != before:
        raise ValueError("Stage61c output fingerprints differ")
    if {row["target_id"] for row in reread} != {expected["corrected_target_id"]}:
        raise ValueError("Stage61c output target ID differs")

    result = {
        "schema_version": "1.0",
        "status": "stage61c_ppard_target_id_amendment_ok",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation),
        "operation": "metadata-only derived table; no docking score or pose was changed",
        "source": descriptor(root, inputs["raw_scores"]),
        "output": descriptor(root, output_path),
        "row_count": len(rows),
        "unique_key_count": len(keys),
        "target_id_before": sorted({row["target_id"] for row in rows}),
        "target_id_after": sorted({row["target_id"] for row in reread}),
        "changed_field_count": changed_fields,
        "non_target_row_fingerprints_exact": True,
        "label_counts": dict(sorted(Counter(row["label"] for row in reread).items())),
        "receptor_count": len(receptor_ids),
        "ligand_count": len(ligand_ids),
        "seed_count": len({row["seed_id"] for row in reread}),
        "data_boundary": {
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "decision": {
            "stage62_frozen_train240_analysis_authorized": True,
            "gpu_redocking_required": False,
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
