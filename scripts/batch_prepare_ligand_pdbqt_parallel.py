"""Prepare ligand PDBQT files with controlled parallel Meeko workers."""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .batch_prepare_ligand_pdbqt import (
        find_meeko_script,
        file_sha256,
        parse_pdbqt,
        read_rows,
        run_meeko,
        safe_filename,
        write_manifest,
    )
except ImportError:
    from batch_prepare_ligand_pdbqt import file_sha256, find_meeko_script, parse_pdbqt, read_rows, run_meeko, safe_filename, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--pdbqt-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-warning-sdf", action="store_true")
    return parser


def prepare_one(
    row: dict[str, str],
    pdbqt_dir: Path,
    meeko_script: Path,
    include_warning_sdf: bool,
) -> dict[str, object]:
    ligand_id = row["ligand_id"]
    sdf_status = row["prep_status"]
    sdf_path = Path(row["sdf_path"]) if row["sdf_path"] else Path()
    pdbqt_path = pdbqt_dir / f"{safe_filename(ligand_id)}.pdbqt"
    if sdf_status == "failed":
        return {**row, "pdbqt_status": "skipped", "pdbqt_message": "input_sdf_failed", "pdbqt_path": ""}
    if sdf_status == "warning" and not include_warning_sdf:
        return {**row, "pdbqt_status": "skipped", "pdbqt_message": "input_sdf_warning", "pdbqt_path": ""}
    if not sdf_path.exists():
        return {**row, "pdbqt_status": "failed", "pdbqt_message": f"missing_sdf:{sdf_path}", "pdbqt_path": ""}
    completed = run_meeko(meeko_script, sdf_path, pdbqt_path)
    combined = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    if completed.returncode == 0 and pdbqt_path.exists():
        return {
            **row,
            "pdbqt_status": "ok",
            "pdbqt_message": "meeko_ok",
            "pdbqt_path": pdbqt_path.as_posix(),
            "pdbqt_sha256": file_sha256(pdbqt_path),
            **parse_pdbqt(pdbqt_path),
        }
    return {**row, "pdbqt_status": "failed", "pdbqt_message": combined[-500:], "pdbqt_path": pdbqt_path.as_posix() if pdbqt_path.exists() else ""}


def read_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["ligand_id"]: row for row in csv.DictReader(handle)}


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    rows = read_rows(args.input_manifest)
    meeko_script = find_meeko_script()
    args.pdbqt_dir.mkdir(parents=True, exist_ok=True)
    existing = read_existing(args.output_manifest) if args.resume else {}
    pending = []
    output: dict[str, dict[str, object]] = {}
    for row in rows:
        old = existing.get(row["ligand_id"])
        if args.resume and old and old.get("pdbqt_status") == "ok" and Path(old.get("pdbqt_path", "")).exists():
            old["pdbqt_sha256"] = file_sha256(Path(old["pdbqt_path"]))
            output[row["ligand_id"]] = old
        else:
            pending.append(row)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(prepare_one, row, args.pdbqt_dir, meeko_script, args.include_warning_sdf): row["ligand_id"]
            for row in pending
        }
        for future in as_completed(futures):
            output[futures[future]] = future.result()
            write_manifest(args.output_manifest, [output[row["ligand_id"]] for row in rows if row["ligand_id"] in output])
    final = [output[row["ligand_id"]] for row in rows]
    write_manifest(args.output_manifest, final)
    counts: dict[str, int] = {}
    for row in final:
        counts[str(row["pdbqt_status"])] = counts.get(str(row["pdbqt_status"]), 0) + 1
    print(f"input_rows={len(rows)}")
    for status, count in sorted(counts.items()):
        print(f"{status}={count}")
    print(f"workers={args.workers}")
    print(f"manifest={args.output_manifest}")
    return 0 if counts.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
