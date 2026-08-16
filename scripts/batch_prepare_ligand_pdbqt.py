"""Batch-convert 3D ligand SDF files to Meeko/Vina PDBQT files.

Thin CLI wrapper; the core logic lives in ``qubo_receptor_ensemble.preparation``.
"""



from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import argparse
from pathlib import Path

from qubo_receptor_ensemble.io import file_sha256, safe_filename
from qubo_receptor_ensemble.preparation import (
    PDBQT_REQUIRED_COLUMNS,
    find_meeko_script,
    parse_pdbqt,
    read_rows as _read_rows,
    run_meeko,
    validate_columns as _validate_columns,
    validated_existing_pdbqt,
    write_manifest,
)

REQUIRED_COLUMNS = PDBQT_REQUIRED_COLUMNS

__all__ = [
    "REQUIRED_COLUMNS",
    "file_sha256",
    "validate_columns",
    "safe_filename",
    "find_meeko_script",
    "read_rows",
    "parse_pdbqt",
    "validated_existing_pdbqt",
    "run_meeko",
    "write_manifest",
]


def validate_columns(fieldnames: list[str] | None) -> None:
    _validate_columns(fieldnames, PDBQT_REQUIRED_COLUMNS, "manifest")


def read_rows(input_manifest: Path) -> list[dict[str, str]]:
    return _read_rows(input_manifest, PDBQT_REQUIRED_COLUMNS, "manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--pdbqt-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--include-warning-sdf",
        action="store_true",
        help="Prepare SDF rows marked warning in the 3D manifest.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Validate and reuse complete existing PDBQT files.",
    )
    parser.add_argument(
        "--rigid-macrocycles",
        action="store_true",
        help="Pass --rigid_macrocycles to Meeko ligand preparation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_rows(args.input_manifest)
    meeko_script = find_meeko_script()
    args.pdbqt_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []

    for row in rows:
        ligand_id = row["ligand_id"]
        sdf_status = row["prep_status"]
        sdf_path = Path(row["sdf_path"]) if row["sdf_path"] else Path()
        pdbqt_path = args.pdbqt_dir / f"{safe_filename(ligand_id)}.pdbqt"

        if sdf_status == "failed":
            output_rows.append(
                {
                    **row,
                    "pdbqt_status": "skipped",
                    "pdbqt_message": "input_sdf_failed",
                    "pdbqt_path": "",
                }
            )
            continue
        if sdf_status == "warning" and not args.include_warning_sdf:
            output_rows.append(
                {
                    **row,
                    "pdbqt_status": "skipped",
                    "pdbqt_message": "input_sdf_warning",
                    "pdbqt_path": "",
                }
            )
            continue
        if not sdf_path.exists():
            output_rows.append(
                {
                    **row,
                    "pdbqt_status": "failed",
                    "pdbqt_message": f"missing_sdf:{sdf_path}",
                    "pdbqt_path": "",
                }
            )
            continue

        if args.resume_existing:
            existing = validated_existing_pdbqt(pdbqt_path)
            if existing is not None:
                output_rows.append({**row, **existing})
                continue

        completed = run_meeko(
            meeko_script,
            sdf_path,
            pdbqt_path,
            rigid_macrocycles=args.rigid_macrocycles,
        )
        combined_output = "\n".join(
            part.strip() for part in [completed.stdout, completed.stderr] if part.strip()
        )
        if completed.returncode == 0 and pdbqt_path.exists():
            parsed = parse_pdbqt(pdbqt_path)
            output_rows.append(
                {
                    **row,
                    "pdbqt_status": "ok",
                    "pdbqt_message": (
                        "meeko_rigid_macrocycles_ok"
                        if args.rigid_macrocycles
                        else "meeko_ok"
                    ),
                    "pdbqt_path": pdbqt_path.as_posix(),
                    "pdbqt_sha256": file_sha256(pdbqt_path),
                    **parsed,
                }
            )
        else:
            output_rows.append(
                {
                    **row,
                    "pdbqt_status": "failed",
                    "pdbqt_message": combined_output[-500:],
                    "pdbqt_path": pdbqt_path.as_posix() if pdbqt_path.exists() else "",
                }
            )

    write_manifest(args.output_manifest, output_rows)
    counts: dict[str, int] = {}
    for row in output_rows:
        status = str(row["pdbqt_status"])
        counts[status] = counts.get(status, 0) + 1
    print(f"input_rows={len(rows)}")
    for status, count in sorted(counts.items()):
        print(f"{status}={count}")
    print(
        "reused_existing="
        f"{sum(row.get('pdbqt_message') == 'meeko_existing_validated' for row in output_rows)}"
    )
    print(f"meeko_script={meeko_script}")
    print(f"rigid_macrocycles={args.rigid_macrocycles}")
    print(f"pdbqt_dir={args.pdbqt_dir}")
    print(f"manifest={args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
