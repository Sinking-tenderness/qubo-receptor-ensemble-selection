"""Run AutoDock Vina for a ligand PDBQT manifest and write a long score table.

Thin CLI wrapper; the shared docking helpers live in
``qubo_receptor_ensemble.docking``.
"""



from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import argparse
import csv
import subprocess
import time
from pathlib import Path

from qubo_receptor_ensemble.docking import (
    REQUIRED_COLUMNS,
    VINA_CONFIG_KEYS,
    build_vina_command,
    get_vina_version,
    parse_vina_modes,
    read_manifest,
    read_vina_config,
    result_rows_for_modes,
    safe_filename,
    select_rows,
    validate_columns,
    write_csv,
)

__all__ = [
    "REQUIRED_COLUMNS",
    "VINA_CONFIG_KEYS",
    "validate_columns",
    "read_manifest",
    "read_vina_config",
    "safe_filename",
    "select_rows",
    "get_vina_version",
    "parse_vina_modes",
    "result_rows_for_modes",
    "build_vina_command",
    "write_csv",
    "read_checkpoint",
]


def read_checkpoint(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Ligand PDBQT manifest CSV")
    parser.add_argument("--vina-exe", type=Path, required=True, help="AutoDock Vina executable")
    parser.add_argument("--receptor", type=Path, required=True, help="Prepared receptor PDBQT")
    parser.add_argument("--receptor-id", required=True, help="Stable receptor ID for output table")
    parser.add_argument("--config", type=Path, required=True, help="Vina box and search config")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for docked poses")
    parser.add_argument("--log-dir", type=Path, required=True, help="Directory for per-ligand Vina logs")
    parser.add_argument("--score-table", type=Path, required=True, help="Output long score table CSV")
    parser.add_argument(
        "--checkpoint-table",
        type=Path,
        default=None,
        help="Per-ligand checkpoint CSV; defaults to score-table stem + .checkpoint.csv",
    )
    parser.add_argument("--base-seed", type=int, default=20260709)
    parser.add_argument("--max-ligands", type=int, default=None)
    parser.add_argument("--sample-per-label", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=20260709)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing pose/log files when possible and parse their score tables.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.vina_exe.is_file():
        raise FileNotFoundError(f"Vina executable not found: {args.vina_exe}")
    if not args.receptor.is_file():
        raise FileNotFoundError(f"receptor PDBQT not found: {args.receptor}")
    rows = read_manifest(args.manifest)
    config = read_vina_config(args.config)
    vina_version = get_vina_version(args.vina_exe)
    selected_rows = select_rows(rows, args.max_ligands, args.sample_per_label, args.sample_seed)
    selected_ids = {row["ligand_id"] for row in selected_rows}
    checkpoint_table = args.checkpoint_table or args.score_table.with_name(
        f"{args.score_table.stem}.checkpoint.csv"
    )
    if checkpoint_table.exists() and not args.resume:
        raise FileExistsError(
            f"checkpoint already exists; use --resume or choose a new path: {checkpoint_table}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []
    successful_checkpoint_ids: set[str] = set()
    if args.resume and checkpoint_table.exists():
        checkpoint_rows = read_checkpoint(checkpoint_table)
        invalid_ids = {row.get("ligand_id", "") for row in checkpoint_rows} - selected_ids
        if invalid_ids:
            raise ValueError(
                f"checkpoint contains ligand IDs not selected for this run: {sorted(invalid_ids)}"
            )
        invalid_receptors = {
            row.get("receptor_id", "")
            for row in checkpoint_rows
            if row.get("receptor_id", "") != args.receptor_id
        }
        if invalid_receptors:
            raise ValueError(
                f"checkpoint receptor IDs do not match {args.receptor_id}: {sorted(invalid_receptors)}"
            )
        failed_checkpoint_ids = {
            row["ligand_id"] for row in checkpoint_rows if row.get("status") == "failed"
        }
        output_rows = [
            row for row in checkpoint_rows if row.get("ligand_id") not in failed_checkpoint_ids
        ]
        successful_checkpoint_ids = {
            row["ligand_id"] for row in output_rows if row.get("status") == "ok"
        }

    def save_checkpoint() -> None:
        write_csv(checkpoint_table, output_rows)

    for index, row in enumerate(selected_rows):
        ligand_id = row["ligand_id"]
        ligand_path = Path(row["pdbqt_path"])
        ligand_seed = args.base_seed + index
        safe_id = safe_filename(ligand_id)
        output_pose = args.output_dir / f"{safe_id}_docked.pdbqt"
        log_path = args.log_dir / f"{safe_id}_vina.log"
        start = time.perf_counter()

        if ligand_id in successful_checkpoint_ids and output_pose.exists() and log_path.exists():
            continue
        if ligand_id in successful_checkpoint_ids:
            output_rows = [item for item in output_rows if item.get("ligand_id") != ligand_id]
            successful_checkpoint_ids.discard(ligand_id)

        if not ligand_path.exists():
            runtime = time.perf_counter() - start
            output_rows.append(
                {
                    "target_id": row["target_id"],
                    "receptor_id": args.receptor_id,
                    "ligand_id": ligand_id,
                    "label": row["label"],
                    "pose_rank": "",
                    "docking_score": "",
                    "status": "failed",
                    "message": f"missing_ligand_pdbqt:{ligand_path}",
                    "runtime_seconds": round(runtime, 3),
                    "seed": ligand_seed,
                    "software_version": vina_version,
                    "pose_path": "",
                    "log_path": "",
                }
            )
            save_checkpoint()
            continue

        if args.resume and output_pose.exists() and log_path.exists():
            modes = parse_vina_modes(log_path.read_text(encoding="utf-8", errors="ignore"))
            if modes:
                output_rows.extend(
                    result_rows_for_modes(
                        row=row,
                        receptor_id=args.receptor_id,
                        modes=modes,
                        status="ok",
                        message="vina_ok_resumed",
                        runtime_seconds="",
                        seed=ligand_seed,
                        software_version=vina_version,
                        output_pose=output_pose,
                        log_path=log_path,
                    )
                )
                save_checkpoint()
                continue

        cmd = build_vina_command(args.vina_exe, args.receptor, ligand_path, output_pose, config, ligand_seed)
        completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
        runtime = time.perf_counter() - start
        combined_log = "\n".join(
            part.strip() for part in [completed.stdout, completed.stderr] if part.strip()
        )
        log_path.write_text(combined_log, encoding="utf-8")

        modes = parse_vina_modes(completed.stdout)
        if completed.returncode == 0 and output_pose.exists() and modes:
            output_rows.extend(
                result_rows_for_modes(
                    row=row,
                    receptor_id=args.receptor_id,
                    modes=modes,
                    status="ok",
                    message="vina_ok",
                    runtime_seconds=round(runtime, 3),
                    seed=ligand_seed,
                    software_version=vina_version,
                    output_pose=output_pose,
                    log_path=log_path,
                )
            )
        else:
            output_rows.append(
                {
                    "target_id": row["target_id"],
                    "receptor_id": args.receptor_id,
                    "ligand_id": ligand_id,
                    "label": row["label"],
                    "pose_rank": "",
                    "docking_score": "",
                    "status": "failed",
                    "message": combined_log[-500:],
                    "runtime_seconds": round(runtime, 3),
                    "seed": ligand_seed,
                    "software_version": vina_version,
                    "pose_path": output_pose.as_posix() if output_pose.exists() else "",
                    "log_path": log_path.as_posix(),
                }
            )
        save_checkpoint()

    write_csv(args.score_table, output_rows)
    ok_ligands = sorted({row["ligand_id"] for row in output_rows if row["status"] == "ok"})
    failed_ligands = sorted({row["ligand_id"] for row in output_rows if row["status"] == "failed"})
    print(f"input_manifest_rows={len(rows)}")
    print(f"selected_ligands={len(selected_rows)}")
    print(f"ok_ligands={len(ok_ligands)}")
    print(f"failed_ligands={len(failed_ligands)}")
    print(f"score_rows={len(output_rows)}")
    print(f"score_table={args.score_table}")
    print(f"checkpoint_table={checkpoint_table}")
    print(f"output_dir={args.output_dir}")
    print(f"log_dir={args.log_dir}")
    return 0 if not failed_ligands else 1


if __name__ == "__main__":
    raise SystemExit(main())
