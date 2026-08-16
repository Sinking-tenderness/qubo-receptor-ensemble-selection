"""Freeze the complete preparation-ready BACE1 large receptor pool."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from qubo_receptor_ensemble.io import read_json, write_json  # noqa: F401 (deduped)
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_receptor import file_sha256




def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows




def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rooted(root: Path, value: str) -> Path:
    path = (root / value.replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path leaves repository root: {value}") from error
    return path


def verified(root: Path, descriptor: dict[str, Any]) -> Path:
    path = rooted(root, str(descriptor["path"]))
    if not path.is_file() or file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"input identity differs: {path}")
    return path


def portable_source_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    normalized = value.replace("\\", "/")
    marker = "qubo-receptor-ensemble-selection/"
    if marker not in normalized:
        raise ValueError(f"cannot rebase historical source path: {value}")
    rebased = (root / normalized.split(marker, 1)[1]).resolve()
    try:
        rebased.relative_to(root)
    except ValueError as error:
        raise ValueError(f"rebased path leaves repository: {value}") from error
    if not rebased.is_file():
        raise FileNotFoundError(rebased)
    return rebased


def descriptor(root: Path, path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def write_report(path: Path, result: dict[str, Any]) -> None:
    counts = result["counts"]
    lines = [
        "# Stage41a BACE1 large-pool freeze",
        "",
        "All preparation-ready structures are retained. No max-min or activity-driven structural downselection is applied.",
        "",
        f"- Frozen receptors: {counts['frozen_receptor_count']}",
        f"- k=3 states: {counts['state_count_by_k']['3']}",
        f"- k=6 states: {counts['state_count_by_k']['6']}",
        f"- total k=1..6 states: {counts['total_state_count_k1_to_k6']}",
        f"- protected benchmark rows read: {result['data_boundary']['fresh_validation_rows_read'] + result['data_boundary']['locked_test_rows_read']}",
        "",
        result["interpretation_boundary"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path)
    implementation = verified(root, config["implementation"])
    if implementation.resolve() != Path(__file__).resolve():
        raise ValueError("Stage41a implementation identity differs")
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    structural = read_json(inputs["stage21c_summary"])
    source = read_json(inputs["stage21a_source_summary"])
    stage40 = read_json(inputs["stage40_audit"])
    if structural.get("status") != "stage21c_bace1_preparation_ready_structural_pool_ok":
        raise ValueError("BACE1 structural pool did not pass")
    if source.get("status") != "stage21a_bace1_source_and_active_allocation_ok":
        raise ValueError("BACE1 source audit did not pass")
    if stage40.get("status") != "stage40_bedroc_aligned_signed_hubo_audit_ok":
        raise ValueError("Stage40 boundary record did not pass")
    if any(int(value) != 0 for value in structural["data_boundary"].values()):
        raise ValueError("BACE1 structural evidence crossed a protected boundary")

    rows = read_csv(inputs["preparation_ready_pool"])
    expected = int(config["pool_freeze"]["expected_preparation_ready_count"])
    if len(rows) != expected or int(structural["counts"]["preparation_ready_count"]) != expected:
        raise ValueError("BACE1 preparation-ready count differs")
    if len({row["conformer_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate BACE1 conformer ID")
    if sum(row["pdb_id"] == config["target"]["reference_pdb_id"] for row in rows) != 1:
        raise ValueError("BACE1 reference occurrence differs")

    frozen_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["status"] != "coordinate_eligible":
            raise ValueError(f"noneligible BACE1 row: {row['conformer_id']}")
        if int(row["global_incomplete_standard_amino_acid_residue_count"]) != 0:
            raise ValueError(f"incomplete BACE1 structure: {row['conformer_id']}")
        if row["explicit_covalent_connections"].strip():
            raise ValueError(f"covalent BACE1 structure: {row['conformer_id']}")
        mmcif = portable_source_path(root, row["mmcif_path"])
        aligned = portable_source_path(root, row["aligned_protein_pdb_path"])
        if file_sha256(mmcif) != row["mmcif_sha256"].upper():
            raise ValueError(f"mmCIF hash differs: {row['conformer_id']}")
        if file_sha256(aligned) != row["aligned_protein_pdb_sha256"].upper():
            raise ValueError(f"aligned PDB hash differs: {row['conformer_id']}")
        frozen_rows.append(
            {
                "pool_order": 0,
                "conformer_id": row["conformer_id"],
                "pdb_id": row["pdb_id"],
                "chain": row["chain"],
                "is_reference": row["pdb_id"] == config["target"]["reference_pdb_id"],
                "mmcif_path": mmcif.relative_to(root).as_posix(),
                "mmcif_sha256": file_sha256(mmcif),
                "aligned_protein_pdb_path": aligned.relative_to(root).as_posix(),
                "aligned_protein_pdb_sha256": file_sha256(aligned),
                "selected_ligand_resname": row["selected_ligand_resname"],
                "selected_ligand_resseq": row["selected_ligand_resseq"],
                "selected_ligand_icode": row["selected_ligand_icode"],
                "selected_ligand_heavy_atom_count": int(row["selected_ligand_heavy_atom_count"]),
                "resolution_angstrom": float(row["resolution_angstrom"]),
                "pool_role": "complete_preparation_ready_large_pool",
            }
        )
    frozen_rows.sort(key=lambda row: (not bool(row["is_reference"]), str(row["pdb_id"]), str(row["conformer_id"])))
    for index, row in enumerate(frozen_rows):
        row["pool_order"] = index

    outputs = {key: rooted(root, value) for key, value in config["outputs"].items()}
    if not overwrite and any(path.exists() for path in outputs.values()):
        raise FileExistsError("Stage41a outputs exist; pass --overwrite")
    write_csv(outputs["large_pool_manifest_csv"], frozen_rows)
    state_count_by_k = {str(size): math.comb(expected, size) for size in range(1, int(config["algorithmic_benchmark"]["maximum_subset_size"]) + 1)}
    result = {
        "schema_version": "1.0",
        "status": "stage41a_bace1_large_pool_frozen",
        "config": descriptor(root, config_path),
        "implementation": descriptor(root, implementation),
        "inputs": {key: descriptor(root, value) for key, value in inputs.items()},
        "target": config["target"],
        "counts": {
            "metadata_eligible_count": int(structural["counts"]["audited_count"]),
            "coordinate_eligible_count": int(structural["counts"]["coordinate_eligible_count"]),
            "preparation_ready_count": int(structural["counts"]["preparation_ready_count"]),
            "frozen_receptor_count": len(frozen_rows),
            "state_count_by_k": state_count_by_k,
            "total_state_count_k1_to_k6": sum(state_count_by_k.values()),
        },
        "redocking_gate": config["redocking_gate"],
        "development_ligand_protocol": config["development_ligand_protocol"],
        "algorithmic_benchmark": config["algorithmic_benchmark"],
        "data_boundary": {
            "structural_rows_read": len(rows),
            "development_ligand_rows_read": 0,
            "fresh_validation_rows_read": 0,
            "locked_test_rows_read": 0,
            "docking_scores_read": 0,
            "new_docking_jobs": 0,
            "quantum_hardware_jobs": 0,
        },
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_report(outputs["report_md"], result)
    result["outputs"] = {
        "large_pool_manifest_csv": descriptor(root, outputs["large_pool_manifest_csv"]),
        "report_md": descriptor(root, outputs["report_md"]),
    }
    write_json(outputs["result_json"], result)
    print(json.dumps({"status": result["status"], "counts": result["counts"], "data_boundary": result["data_boundary"]}, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage41a_bace1_large_pool_freeze.json")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run(rooted(root, args.config), root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
