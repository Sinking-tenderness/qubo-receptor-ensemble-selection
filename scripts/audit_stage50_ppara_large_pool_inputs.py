"""Independently audit the final Stage50 PPARA large-pool input result."""

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
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()




def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rooted(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def audit(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    summary_path = root / "data/stage50_ppara_large_pool_redocking_input_preparation_summary.json"
    summary = read_json(summary_path)
    if summary.get("status") != "stage50_ppara_large_pool_redocking_inputs_partial_gate_ready":
        raise ValueError("Stage50 final result did not pass its partial technical gate")
    if summary.get("technical_gate_ready") is not True:
        raise ValueError("Stage50 technical gate is false")

    config_path = root / summary["config"]["path"]
    if sha256(config_path) != summary["config"]["sha256"]:
        raise ValueError("Stage50 config identity differs")
    config = read_json(config_path)
    implementation = root / config["implementation"]["path"]
    if sha256(implementation) != config["implementation"]["sha256"]:
        raise ValueError("Stage50 implementation identity differs")
    for dependency in config["dependencies"]:
        if sha256(root / dependency["path"]) != dependency["sha256"]:
            raise ValueError(f"Stage50 dependency identity differs: {dependency['path']}")
    for descriptor in config["inputs"].values():
        if sha256(root / descriptor["path"]) != descriptor["sha256"]:
            raise ValueError(f"Stage50 input identity differs: {descriptor['path']}")
    for descriptor in summary["outputs"].values():
        if sha256(root / descriptor["path"]) != descriptor["sha256"]:
            raise ValueError(f"Stage50 output identity differs: {descriptor['path']}")

    receptors = read_csv(
        root / summary["outputs"]["receptor_manifest_csv"]["path"]
    )
    cases = read_csv(
        root / summary["outputs"]["redocking_case_manifest_csv"]["path"]
    )
    selected = read_csv(root / config["inputs"]["selected_receptor_manifest"]["path"])
    failure_ids = [row["conformer_id"] for row in summary["failed_cases"]]
    expected_failure_ids = [
        "PPARA_4BCR_aligned",
        "PPARA_3KDU_aligned",
        "PPARA_3KDT_aligned",
        "PPARA_7E5G_aligned",
    ]
    if failure_ids != expected_failure_ids:
        raise ValueError("Stage50 failure ledger differs")
    expected_success_ids = [
        row["conformer_id"]
        for row in selected
        if row["conformer_id"] not in set(failure_ids)
    ]
    receptor_ids = [row["conformer_id"] for row in receptors]
    case_ids = [row["conformer_id"] for row in cases]
    if receptor_ids != expected_success_ids or case_ids != expected_success_ids:
        raise ValueError("Stage50 successful receptor order differs")
    if len(receptors) != 60 or len(cases) != 60:
        raise ValueError("Stage50 successful case count differs")
    if len(set(receptor_ids)) != 60 or len({row["case_id"] for row in cases}) != 60:
        raise ValueError("Stage50 successful identities are not unique")

    completed_atom_count = 0
    maximum_displacement = 0.0
    for receptor in receptors:
        for path_key, hash_key in (
            ("receptor_pdbqt", "receptor_pdbqt_sha256"),
            ("completed_receptor_pdb", "completed_receptor_pdb_sha256"),
            ("receptor_preparation_summary", "receptor_preparation_summary_sha256"),
        ):
            path = rooted(root, receptor[path_key])
            if sha256(path) != receptor[hash_key].upper():
                raise ValueError(f"Stage50 receptor artifact differs: {receptor['conformer_id']}")
        evidence = read_json(rooted(root, receptor["receptor_preparation_summary"]))
        if evidence.get("status") != "ok" or evidence.get("allow_bad_res") is not False:
            raise ValueError(f"Stage50 receptor evidence failed: {receptor['conformer_id']}")
        counts = dict(evidence["residue_count_change"])
        if len(set(int(value) for value in counts.values())) != 1:
            raise ValueError(f"Stage50 receptor residue count changed: {receptor['conformer_id']}")
        displacement = float(evidence["maximum_existing_atom_displacement_angstrom"])
        maximum_displacement = max(maximum_displacement, displacement)
        if displacement > 0.001:
            raise ValueError(f"Stage50 existing atom moved: {receptor['conformer_id']}")
        completed_atom_count += int(evidence["completed_heavy_atom_count"])

    for case in cases:
        for path_key, hash_key in (
            ("ligand_pdbqt", "ligand_pdbqt_sha256"),
            ("reference_sdf", "reference_sdf_sha256"),
            ("alignment_summary", "alignment_summary_sha256"),
        ):
            path = rooted(root, case[path_key])
            if sha256(path) != case[hash_key].upper():
                raise ValueError(f"Stage50 ligand artifact differs: {case['case_id']}")

    box = read_json(root / summary["outputs"]["common_box_json"]["path"])
    if box.get("status") != "stage50_ppara_large_pool_common_box_ok":
        raise ValueError("Stage50 common box failed")
    if box["receptor_count"] != 60 or box["size"] != {"x": 28.0, "y": 30.0, "z": 24.0}:
        raise ValueError("Stage50 common box differs")
    if float(box["minimum_observed_margin_angstrom"]) < 3.5:
        raise ValueError("Stage50 common box margin is insufficient")
    if completed_atom_count != 201 or summary["counts"]["completed_missing_heavy_atom_count"] != 201:
        raise ValueError("Stage50 completed-heavy-atom count differs")
    if any(int(value) != 0 for value in summary["data_boundary"].values()):
        raise ValueError("Stage50 crossed a protected data boundary")

    result = {
        "schema_version": "1.0",
        "status": "stage50_ppara_large_pool_inputs_independent_audit_ok",
        "audited_result": {
            "path": summary_path.relative_to(root).as_posix(),
            "sha256": sha256(summary_path),
        },
        "frozen_receptor_count": 64,
        "prepared_receptor_count": 60,
        "technical_preparation_failure_count": 4,
        "technical_preparation_failure_ids": failure_ids,
        "prepared_receptor_order_reproduced": True,
        "completed_missing_heavy_atom_count": completed_atom_count,
        "maximum_existing_atom_displacement_angstrom": maximum_displacement,
        "common_box": box,
        "artifact_identities_ok": True,
        "evidence_boundary_ok": True,
        "cognate_redocking_authorized": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage50_ppara_large_pool_inputs_independent_audit.json"),
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    audit(args.root, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
