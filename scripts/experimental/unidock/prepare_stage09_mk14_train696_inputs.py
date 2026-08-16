"""Build the macrocycle-safe MAPK14 Train-696 Uni-Dock input manifest."""

from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[3] / "src"))
from qubo_receptor_ensemble.io import read_json, write_json  # noqa: F401 (deduped)
import argparse
import csv
import importlib.metadata
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

try:
    from scripts.batch_prepare_ligand_pdbqt import (
        file_sha256,
        find_meeko_script,
        parse_pdbqt,
        run_meeko,
        safe_filename,
    )
    from .run_unidock_gpu_equivalence import macrocycle_closure_atom_types
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.batch_prepare_ligand_pdbqt import (
        file_sha256,
        find_meeko_script,
        parse_pdbqt,
        run_meeko,
        safe_filename,
    )
    from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
        macrocycle_closure_atom_types,
    )


PDBQT_FIELDS = (
    "pdbqt_status",
    "pdbqt_message",
    "pdbqt_path",
    "pdbqt_atom_count",
    "pdbqt_atom_types",
    "pdbqt_charge_min",
    "pdbqt_charge_max",
    "torsdof",
    "pdbqt_sha256",
)




def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)




def rooted(root: Path, value: str) -> Path:
    path = (root / value.replace("\\", "/")).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path leaves repository root: {value}") from error
    return path


def checked_path(root: Path, descriptor: dict[str, object]) -> Path:
    path = rooted(root, str(descriptor["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"input identity differs: {path}")
    return path


def expected_macrocycles(
    rows: list[dict[str, str]], descriptors: list[dict[str, object]]
) -> list[tuple[int, dict[str, str], dict[str, object]]]:
    by_id = {row["ligand_id"]: (index, row) for index, row in enumerate(rows)}
    if len(by_id) != len(rows):
        raise ValueError("source manifest contains duplicate ligand IDs")
    selected: list[tuple[int, dict[str, str], dict[str, object]]] = []
    for descriptor in descriptors:
        ligand_id = str(descriptor["ligand_id"])
        if ligand_id not in by_id:
            raise ValueError(f"expected macrocycle is absent: {ligand_id}")
        index, row = by_id[ligand_id]
        checks = {
            "source_manifest_index": index,
            "label": row["label"],
            "source_pdbqt_sha256": row["pdbqt_sha256"].upper(),
        }
        for key, observed in checks.items():
            expected = descriptor[key]
            if str(observed) != str(expected):
                raise ValueError(f"macrocycle descriptor differs: {ligand_id} {key}")
        selected.append((index, row, descriptor))
    return selected


def prepare_rigid_pdbqt(
    meeko_script: Path,
    source_sdf: Path,
    destination: Path,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage09_macrocycle_") as temporary:
        staging = Path(temporary)
        staged_sdf = staging / "ligand.sdf"
        staged_pdbqt = staging / "ligand.pdbqt"
        shutil.copyfile(source_sdf, staged_sdf)
        completed = run_meeko(
            meeko_script,
            staged_sdf,
            staged_pdbqt,
            rigid_macrocycles=True,
        )
        if completed.returncode != 0 or not staged_pdbqt.is_file():
            message = "\n".join(
                part.strip()
                for part in (completed.stdout, completed.stderr)
                if part.strip()
            )
            raise RuntimeError(f"Meeko rigid-macrocycle preparation failed: {message[-500:]}")
        shutil.copyfile(staged_pdbqt, destination)
    pseudoatoms = macrocycle_closure_atom_types(destination)
    if pseudoatoms:
        raise ValueError(f"rigid PDBQT retained closure pseudoatoms: {pseudoatoms}")
    return {
        "pdbqt_status": "ok",
        "pdbqt_message": "meeko_rigid_macrocycles_ok",
        "pdbqt_path": destination.as_posix(),
        "pdbqt_sha256": file_sha256(destination),
        **parse_pdbqt(destination),
    }


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    preparer = checked_path(root, dict(implementation["preparer"]))
    if preparer.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 09 preparer path differs")
    checked_path(root, dict(implementation["batch_prepare_helper"]))
    checked_path(root, dict(implementation["pseudoatom_audit_helper"]))

    boundary = dict(config["data_boundary"])
    if int(boundary["validation_rows_permitted"]) != 0 or int(
        boundary["test_rows_permitted"]
    ) != 0:
        raise ValueError("Stage 09 input preparation crossed a data boundary")
    source_path = checked_path(root, dict(config["input_manifest"]))
    source_rows = read_csv(source_path)
    expected = dict(config["expected"])
    if len(source_rows) != int(expected["ligand_count"]):
        raise ValueError("Train-696 source count differs")
    labels = Counter(row["label"] for row in source_rows)
    expected_labels = Counter(
        {key: int(value) for key, value in dict(expected["label_counts"]).items()}
    )
    if labels != expected_labels:
        raise ValueError("Train-696 source labels differ")
    if {row["split"] for row in source_rows} != {boundary["allowed_split"]}:
        raise ValueError("a non-train ligand is visible")
    if {row["selection_role"] for row in source_rows} != {
        boundary["allowed_selection_role"]
    }:
        raise ValueError("Train-696 selection role differs")

    for row in source_rows:
        pdbqt = rooted(root, row["pdbqt_path"])
        if not pdbqt.is_file() or file_sha256(pdbqt) != row["pdbqt_sha256"].upper():
            raise ValueError(f"source PDBQT identity differs: {row['ligand_id']}")
    descriptors = list(config["macrocycle_ligands"])
    selected = expected_macrocycles(source_rows, descriptors)
    detected = [
        row["ligand_id"]
        for row in source_rows
        if macrocycle_closure_atom_types(rooted(root, row["pdbqt_path"]))
    ]
    selected_ids = [row[1]["ligand_id"] for row in selected]
    if detected != selected_ids:
        raise ValueError("detected macrocycle list differs from preregistration")
    if len(selected) != int(expected["replacement_count"]):
        raise ValueError("macrocycle replacement count differs")

    meeko_version = importlib.metadata.version("meeko")
    if meeko_version != str(config["preparation"]["meeko_version"]):
        raise ValueError(f"Meeko version differs: {meeko_version}")
    meeko_script = find_meeko_script()
    outputs = dict(config["outputs"])
    output_directory = rooted(root, str(outputs["pdbqt_directory"]))
    output_manifest = rooted(root, str(outputs["manifest_csv"]))
    output_summary = rooted(root, str(outputs["summary_json"]))
    if not overwrite and (output_manifest.exists() or output_summary.exists()):
        raise FileExistsError("Stage 09 input outputs exist; pass --overwrite")

    rigid_by_id: dict[str, dict[str, object]] = {}
    for index, row, descriptor in selected:
        sdf = rooted(root, row["sdf_path"])
        if not sdf.is_file() or file_sha256(sdf) != str(
            descriptor["sdf_sha256"]
        ).upper():
            raise ValueError(f"macrocycle SDF identity differs: {row['ligand_id']}")
        destination = output_directory / f"{safe_filename(row['ligand_id'])}.pdbqt"
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        prepared = prepare_rigid_pdbqt(meeko_script, sdf, destination)
        prepared["pdbqt_path"] = destination.relative_to(root).as_posix()
        rigid_by_id[row["ligand_id"]] = {
            **prepared,
            "source_manifest_index": index,
            "source_pdbqt_path": row["pdbqt_path"],
            "source_pdbqt_sha256": row["pdbqt_sha256"],
            "sdf_sha256": file_sha256(sdf),
        }

    merged_rows: list[dict[str, object]] = []
    for index, source in enumerate(source_rows):
        ligand_id = source["ligand_id"]
        row: dict[str, object] = {
            **source,
            "source_manifest_index": index,
            "seed_offset": index,
            "preparation_variant": "original_meeko_flexible",
            "source_pdbqt_path": source["pdbqt_path"],
            "source_pdbqt_sha256": source["pdbqt_sha256"],
        }
        if ligand_id in rigid_by_id:
            rigid = rigid_by_id[ligand_id]
            for field in PDBQT_FIELDS:
                row[field] = rigid[field]
            row["preparation_variant"] = "meeko_rigid_macrocycles"
            row["sdf_sha256"] = rigid["sdf_sha256"]
        merged_rows.append(row)

    remaining: list[str] = []
    for row in merged_rows:
        path = rooted(root, str(row["pdbqt_path"]))
        if file_sha256(path) != str(row["pdbqt_sha256"]).upper():
            raise ValueError(f"merged PDBQT identity differs: {row['ligand_id']}")
        if macrocycle_closure_atom_types(path):
            remaining.append(str(row["ligand_id"]))
    if remaining:
        raise ValueError(f"closure pseudoatoms remain: {remaining}")
    variants = Counter(str(row["preparation_variant"]) for row in merged_rows)
    expected_variants = Counter(
        {
            "original_meeko_flexible": len(merged_rows) - len(selected),
            "meeko_rigid_macrocycles": len(selected),
        }
    )
    if variants != expected_variants:
        raise ValueError("Stage 09 preparation variants differ")

    write_csv(output_manifest, merged_rows)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage09_train696_unidock_inputs_ok",
        "operation": "train-only Uni-Dock compatibility substitution",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "source_manifest": {
            "path": source_path.relative_to(root).as_posix(),
            "sha256": file_sha256(source_path),
        },
        "ligand_count": len(merged_rows),
        "label_counts": dict(sorted(labels.items())),
        "replacement_count": len(selected),
        "replacement_ligand_ids": selected_ids,
        "preparation_variant_counts": dict(sorted(variants.items())),
        "nonreplacement_pdbqt_identity_preserved": all(
            row["pdbqt_sha256"] == row["source_pdbqt_sha256"]
            for row in merged_rows
            if row["preparation_variant"] == "original_meeko_flexible"
        ),
        "closure_pseudoatom_ligand_count": 0,
        "order_preserved": [row["ligand_id"] for row in merged_rows]
        == [row["ligand_id"] for row in source_rows],
        "meeko": {
            "version": meeko_version,
            "script": meeko_script.as_posix(),
            "rigid_macrocycles": True,
        },
        "data_boundary": {
            "validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "output": {
            "path": output_manifest.relative_to(root).as_posix(),
            "sha256": file_sha256(output_manifest),
        },
        "interpretation_note": config["interpretation_boundary"],
    }
    write_json(output_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


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
