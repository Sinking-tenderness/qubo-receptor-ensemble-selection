"""Run pocket feature extraction for every conformer in a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REQUIRED_COLUMNS = {"conformer_id", "pdb_path", "chain", "source_type"}


def safe_filename(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("conformer manifest is empty")
    ids = [row["conformer_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("conformer_id values must be unique")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conformer-manifest", type=Path, required=True)
    parser.add_argument("--reference-pdb", type=Path, required=True)
    parser.add_argument("--reference-chain", default="A")
    parser.add_argument("--ligand-resname", default="STU")
    parser.add_argument("--ligand-chain", default="A")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = read_manifest(args.conformer_manifest)
    script = Path(__file__).with_name("extract_pocket_features.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, object]] = []
    for row in rows:
        conformer = Path(row["pdb_path"])
        if not conformer.is_file():
            outputs.append({**row, "status": "failed", "message": f"missing_pdb:{conformer}", "residue_csv": "", "summary_json": ""})
            continue
        stem = safe_filename(row["conformer_id"])
        residue_csv = args.output_dir / f"{stem}_residues.csv"
        summary_json = args.output_dir / f"{stem}_summary.json"
        command = [
            sys.executable, str(script), "--reference-pdb", str(args.reference_pdb),
            "--reference-chain", args.reference_chain, "--ligand-resname", args.ligand_resname,
            "--ligand-chain", args.ligand_chain, "--cutoff", str(args.cutoff),
            "--conformer-pdb", str(conformer), "--conformer-id", row["conformer_id"],
            "--conformer-chain", row["chain"], "--residue-output", str(residue_csv),
            "--summary-output", str(summary_json),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode == 0 and residue_csv.is_file() and summary_json.is_file():
            outputs.append({**row, "status": "ok", "message": "pocket_features_ok", "residue_csv": residue_csv.as_posix(), "summary_json": summary_json.as_posix()})
        else:
            message = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
            outputs.append({**row, "status": "failed", "message": message[-500:], "residue_csv": "", "summary_json": ""})

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = list(outputs[0])
    with args.output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(outputs)
    status_counts: dict[str, int] = {}
    for output in outputs:
        status_counts[str(output["status"])] = status_counts.get(str(output["status"]), 0) + 1
    print(json.dumps({"input_conformers": len(rows), "status_counts": status_counts, "output_manifest": args.output_manifest.as_posix()}, indent=2))
    return 0 if status_counts.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
