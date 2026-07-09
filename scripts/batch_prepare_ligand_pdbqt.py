"""Batch-convert 3D ligand SDF files to Meeko/Vina PDBQT files."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


REQUIRED_COLUMNS = {"ligand_id", "label", "sdf_path", "prep_status"}


def validate_columns(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("input CSV has no header")
    missing = REQUIRED_COLUMNS.difference(fieldnames)
    if missing:
        raise ValueError(f"input manifest is missing required columns: {sorted(missing)}")


def safe_filename(text: str) -> str:
    keep = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def find_meeko_script() -> Path:
    candidates = [
        Path(sys.prefix) / "Scripts" / "mk_prepare_ligand.py",
        Path(sys.prefix) / "bin" / "mk_prepare_ligand.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find mk_prepare_ligand.py under the active Python environment"
    )


def read_rows(input_manifest: Path) -> list[dict[str, str]]:
    with input_manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames)
        return list(reader)


def parse_pdbqt(pdbqt_path: Path) -> dict[str, object]:
    atom_count = 0
    charges: list[float] = []
    atom_types: set[str] = set()
    torsdof = ""
    with pdbqt_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                atom_count += 1
                if len(line) >= 76:
                    try:
                        charges.append(float(line[70:76].strip()))
                    except ValueError:
                        pass
                if len(line) >= 78:
                    atom_types.add(line[77:].strip())
            elif line.startswith("TORSDOF"):
                parts = line.split()
                torsdof = parts[1] if len(parts) > 1 else ""
    return {
        "pdbqt_atom_count": atom_count,
        "pdbqt_atom_types": ";".join(sorted(atom_types)),
        "pdbqt_charge_min": min(charges) if charges else "",
        "pdbqt_charge_max": max(charges) if charges else "",
        "torsdof": torsdof,
    }


def run_meeko(meeko_script: Path, sdf_path: Path, pdbqt_path: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(meeko_script),
        "-i",
        str(sdf_path),
        "-o",
        str(pdbqt_path),
    ]
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def write_manifest(output_manifest: Path, rows: list[dict[str, object]]) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

        completed = run_meeko(meeko_script, sdf_path, pdbqt_path)
        combined_output = "\n".join(
            part.strip() for part in [completed.stdout, completed.stderr] if part.strip()
        )
        if completed.returncode == 0 and pdbqt_path.exists():
            parsed = parse_pdbqt(pdbqt_path)
            output_rows.append(
                {
                    **row,
                    "pdbqt_status": "ok",
                    "pdbqt_message": "meeko_ok",
                    "pdbqt_path": pdbqt_path.as_posix(),
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
    print(f"meeko_script={meeko_script}")
    print(f"pdbqt_dir={args.pdbqt_dir}")
    print(f"manifest={args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
