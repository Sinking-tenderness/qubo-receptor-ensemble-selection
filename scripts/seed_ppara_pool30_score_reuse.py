"""Verify and hard-link reusable PPARA score tables into a pool30 run.

Run this only on the Linux host after the pool30 ``prepare`` stage has
completed. It rejects any mismatch in the ligand PDBQT bytes, docking box,
docking configuration, or overlapping receptor PDBQT bytes before linking a
source score table into the destination run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/qubo_data_root")


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _read_csv_rows(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV missing required columns {sorted(required_columns)}: {path}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def _resolve_artifact_path(value: str, data_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def _read_ligands(run_directory: Path, data_root: Path) -> list[dict[str, str]]:
    rows = _read_csv_rows(
        run_directory / "prepared_ligands.csv",
        {"ligand_id", "label", "selection_role", "pdbqt_path"},
    )
    ids = [row["ligand_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"prepared ligand manifest has duplicate IDs: {run_directory}")
    return rows


def _verify_ligands(
    source_run: Path, destination_run: Path, data_root: Path
) -> list[str]:
    source_rows = _read_ligands(source_run, data_root)
    destination_rows = _read_ligands(destination_run, data_root)
    source_by_id = {row["ligand_id"]: row for row in source_rows}
    destination_by_id = {row["ligand_id"]: row for row in destination_rows}
    if list(source_by_id) != list(destination_by_id):
        raise ValueError("prepared ligand IDs or order differ")
    for ligand_id, source_row in source_by_id.items():
        destination_row = destination_by_id[ligand_id]
        for field in ("label", "selection_role"):
            if source_row[field] != destination_row[field]:
                raise ValueError(f"ligand {field} differs for {ligand_id}")
        source_hash = _file_sha256(
            _resolve_artifact_path(source_row["pdbqt_path"], data_root)
        )
        destination_hash = _file_sha256(
            _resolve_artifact_path(destination_row["pdbqt_path"], data_root)
        )
        if source_hash != destination_hash:
            raise ValueError(f"ligand PDBQT hash differs for {ligand_id}")
    return list(source_by_id)


def _read_selected_receptors(run_directory: Path) -> list[dict[str, str]]:
    payload = _read_json_object(run_directory / "receptor_preparation_audit.json")
    selected = payload.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"receptor audit has no selected receptors: {run_directory}")
    records: list[dict[str, str]] = []
    for row in selected:
        if not isinstance(row, dict):
            raise ValueError(f"invalid receptor audit row: {run_directory}")
        receptor_id = str(row.get("conformer_id", "")).strip()
        receptor_pdbqt = str(row.get("receptor_pdbqt", "")).strip()
        if not receptor_id or not receptor_pdbqt or str(row.get("status", "")) != "ok":
            raise ValueError(f"invalid selected receptor record: {run_directory}")
        records.append({"conformer_id": receptor_id, "receptor_pdbqt": receptor_pdbqt})
    return records


def _verify_receptor_prefix(
    source_run: Path, destination_run: Path, data_root: Path
) -> list[str]:
    source_rows = _read_selected_receptors(source_run)
    destination_rows = _read_selected_receptors(destination_run)
    if len(destination_rows) <= len(source_rows):
        raise ValueError("destination receptor pool is not larger than source pool")
    source_ids = [row["conformer_id"] for row in source_rows]
    destination_prefix = [row["conformer_id"] for row in destination_rows[: len(source_rows)]]
    if source_ids != destination_prefix:
        raise ValueError("source receptor IDs are not the destination prefix")
    for source_row, destination_row in zip(
        source_rows, destination_rows[: len(source_rows)], strict=True
    ):
        source_hash = _file_sha256(
            _resolve_artifact_path(source_row["receptor_pdbqt"], data_root)
        )
        destination_hash = _file_sha256(
            _resolve_artifact_path(destination_row["receptor_pdbqt"], data_root)
        )
        if source_hash != destination_hash:
            raise ValueError(
                f"receptor PDBQT hash differs for {source_row['conformer_id']}"
            )
    return source_ids


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _verify_box(source_run: Path, destination_run: Path) -> None:
    source_box = _read_json_object(source_run / "docking_box.json")
    destination_box = _read_json_object(destination_run / "docking_box.json")
    if _canonical_json(source_box) != _canonical_json(destination_box):
        raise ValueError("docking box differs")


def _read_docking_config(run_directory: Path) -> tuple[str, dict[str, Any], list[int]]:
    snapshot = _read_json_object(run_directory / "config.snapshot.json")
    target_id = str(snapshot.get("target_id", "")).strip()
    docking = snapshot.get("docking")
    if not target_id or not isinstance(docking, dict):
        raise ValueError(f"config snapshot lacks target or docking config: {run_directory}")
    raw_seeds = docking.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError(f"docking configuration lacks seeds: {run_directory}")
    seeds = [int(value) for value in raw_seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"docking configuration has duplicate seeds: {run_directory}")
    return target_id, docking, seeds


def _verify_docking_config(source_run: Path, destination_run: Path) -> list[int]:
    source_target, source_docking, source_seeds = _read_docking_config(source_run)
    destination_target, destination_docking, destination_seeds = _read_docking_config(
        destination_run
    )
    if source_target != destination_target:
        raise ValueError("target IDs differ")
    if _canonical_json(source_docking) != _canonical_json(destination_docking):
        raise ValueError("docking configuration differs")
    if source_seeds != destination_seeds:
        raise ValueError("docking seed order differs")
    return source_seeds


def _require_complete_score_table(
    path: Path, ligand_ids: list[str], receptor_id: str, seed: int
) -> None:
    rows = _read_csv_rows(
        path,
        {"ligand_id", "receptor_id", "seed", "status"},
    )
    expected = set(ligand_ids)
    observed = {row["ligand_id"] for row in rows}
    if (
        len(rows) != len(ligand_ids)
        or observed != expected
        or any(row["receptor_id"] != receptor_id for row in rows)
        or any(int(float(row["seed"])) != seed for row in rows)
        or any(row["status"] != "ok" for row in rows)
    ):
        raise ValueError(f"score table is not complete for seed={seed}, receptor={receptor_id}: {path}")


def seed_verified_score_tables(
    source_run: Path, destination_run: Path, *, data_root: Path = DEFAULT_DATA_ROOT
) -> dict[str, object]:
    """Hard-link complete, verified score tables for the shared receptor prefix."""
    source_run = source_run.resolve()
    destination_run = destination_run.resolve()
    data_root = data_root.resolve()
    if source_run == destination_run:
        raise ValueError("source and destination run directories must differ")
    ligand_ids = _verify_ligands(source_run, destination_run, data_root)
    shared_receptor_ids = _verify_receptor_prefix(source_run, destination_run, data_root)
    _verify_box(source_run, destination_run)
    seeds = _verify_docking_config(source_run, destination_run)

    source_scores = source_run / "score_tables"
    destination_scores = destination_run / "score_tables"
    destination_scores.mkdir(parents=True, exist_ok=True)
    linked_tables: list[dict[str, str]] = []
    for seed in seeds:
        for receptor_id in shared_receptor_ids:
            filename = f"seed_{seed}__{receptor_id}.csv"
            source_table = source_scores / filename
            destination_table = destination_scores / filename
            _require_complete_score_table(source_table, ligand_ids, receptor_id, seed)
            action = "linked"
            if destination_table.exists():
                if not os.path.samefile(source_table, destination_table):
                    raise FileExistsError(
                        f"destination score table already exists and is not the verified link: "
                        f"{destination_table}"
                    )
                action = "already_linked"
            else:
                os.link(source_table, destination_table)
            linked_tables.append(
                {
                    "action": action,
                    "filename": filename,
                    "sha256": _file_sha256(source_table),
                    "source": str(source_table),
                    "destination": str(destination_table),
                }
            )

    audit = {
        "status": "ok",
        "source_run": str(source_run),
        "destination_run": str(destination_run),
        "data_root": str(data_root),
        "ligand_count": len(ligand_ids),
        "seeds": seeds,
        "shared_receptor_ids": shared_receptor_ids,
        "linked_table_count": len(linked_tables),
        "linked_tables": linked_tables,
    }
    audit_path = destination_run / "score_table_reuse_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--destination-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    audit = seed_verified_score_tables(
        args.source_run,
        args.destination_run,
        data_root=args.data_root,
    )
    print(json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
