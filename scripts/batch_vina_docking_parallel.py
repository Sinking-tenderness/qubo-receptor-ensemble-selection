"""Run controlled parallel AutoDock Vina jobs with resumable checkpoints."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from .batch_vina_docking import (
        build_vina_command,
        get_vina_version,
        parse_vina_modes,
        read_manifest,
        read_vina_config,
        result_rows_for_modes,
        safe_filename,
        select_rows,
        write_csv,
    )
except ImportError:
    from batch_vina_docking import (
        build_vina_command,
        get_vina_version,
        parse_vina_modes,
        read_manifest,
        read_vina_config,
        result_rows_for_modes,
        safe_filename,
        select_rows,
        write_csv,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vina-exe", type=Path, required=True)
    parser.add_argument("--receptor", type=Path, required=True)
    parser.add_argument("--receptor-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--score-table", type=Path, required=True)
    parser.add_argument("--checkpoint-table", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-total-cpu", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=20260709)
    parser.add_argument("--max-ligands", type=int, default=None)
    parser.add_argument("--sample-per-label", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=20260709)
    parser.add_argument("--resume", action="store_true")
    return parser


def read_checkpoint(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def replace_ligand_rows(
    rows: list[dict[str, object]], ligand_id: str, replacement: list[dict[str, object]]
) -> list[dict[str, object]]:
    return [row for row in rows if row.get("ligand_id") != ligand_id] + replacement


def ligand_seed(row: dict[str, str], index: int, base_seed: int) -> int:
    value = row.get("seed_offset", "").strip()
    offset = int(value) if value else index
    if offset < 0:
        raise ValueError(f"negative seed offset for {row['ligand_id']}")
    return base_seed + offset


def dock_one(
    row: dict[str, str],
    index: int,
    args: argparse.Namespace,
    config: dict[str, str],
    vina_version: str,
) -> list[dict[str, object]]:
    ligand_id = row["ligand_id"]
    ligand_path = Path(row["pdbqt_path"])
    seed = ligand_seed(row, index, args.base_seed)
    output_pose = args.output_dir / f"{safe_filename(ligand_id)}_docked.pdbqt"
    log_path = args.log_dir / f"{safe_filename(ligand_id)}_vina.log"
    start = time.perf_counter()
    if not ligand_path.exists():
        return [{
            "target_id": row["target_id"], "receptor_id": args.receptor_id,
            "ligand_id": ligand_id, "label": row["label"], "pose_rank": "",
            "docking_score": "", "status": "failed",
            "message": f"missing_ligand_pdbqt:{ligand_path}",
            "runtime_seconds": round(time.perf_counter() - start, 3),
            "seed": seed, "software_version": vina_version,
            "pose_path": "", "log_path": "",
        }]

    command = build_vina_command(
        args.vina_exe, args.receptor, ligand_path, output_pose, config, seed
    )
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    runtime = time.perf_counter() - start
    combined_log = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    log_path.write_text(combined_log, encoding="utf-8")
    modes = parse_vina_modes(completed.stdout)
    if completed.returncode == 0 and output_pose.exists() and modes:
        return result_rows_for_modes(
            row, args.receptor_id, modes, "ok", "vina_ok_parallel",
            round(runtime, 3), seed, vina_version, output_pose, log_path,
        )
    return [{
        "target_id": row["target_id"], "receptor_id": args.receptor_id,
        "ligand_id": ligand_id, "label": row["label"], "pose_rank": "",
        "docking_score": "", "status": "failed", "message": combined_log[-500:],
        "runtime_seconds": round(runtime, 3), "seed": seed,
        "software_version": vina_version,
        "pose_path": output_pose.as_posix() if output_pose.exists() else "",
        "log_path": log_path.as_posix(),
    }]


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if not args.vina_exe.is_file():
        raise FileNotFoundError(args.vina_exe)
    if not args.receptor.is_file():
        raise FileNotFoundError(args.receptor)

    rows = read_manifest(args.manifest)
    config = read_vina_config(args.config)
    config_cpu = int(config.get("cpu", "0"))
    if args.max_total_cpu and config_cpu and args.workers * config_cpu > args.max_total_cpu:
        raise ValueError(
            f"workers*cpu={args.workers * config_cpu} exceeds max-total-cpu={args.max_total_cpu}"
        )
    vina_version = get_vina_version(args.vina_exe)
    selected = select_rows(rows, args.max_ligands, args.sample_per_label, args.sample_seed)
    selected_ids = {row["ligand_id"] for row in selected}
    checkpoint = args.checkpoint_table or args.score_table.with_name(
        f"{args.score_table.stem}.checkpoint.csv"
    )
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(f"checkpoint exists; use --resume: {checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, object]] = []
    done_ids: set[str] = set()
    if args.resume and checkpoint.exists():
        checkpoint_rows = read_checkpoint(checkpoint)
        output_rows = [row for row in checkpoint_rows if row.get("status") != "failed"]
        done_ids = {row["ligand_id"] for row in output_rows if row.get("status") == "ok"}

    tasks = [
        (index, row) for index, row in enumerate(selected)
        if row["ligand_id"] not in done_ids
    ]

    def save() -> None:
        write_csv(checkpoint, sorted(output_rows, key=lambda row: str(row.get("ligand_id", ""))))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(dock_one, row, index, args, config, vina_version): row["ligand_id"]
            for index, row in tasks
        }
        for future in as_completed(futures):
            ligand_id = futures[future]
            result = future.result()
            output_rows = replace_ligand_rows(output_rows, ligand_id, result)
            save()

    write_csv(args.score_table, sorted(output_rows, key=lambda row: str(row.get("ligand_id", ""))))
    ok_ids = {row["ligand_id"] for row in output_rows if row.get("status") == "ok"}
    failed_ids = {row["ligand_id"] for row in output_rows if row.get("status") == "failed"}
    print(f"selected_ligands={len(selected)}")
    print(f"ok_ligands={len(ok_ids)}")
    print(f"failed_ligands={len(failed_ids)}")
    print(f"workers={args.workers}")
    print(f"cpu_per_worker={config_cpu}")
    print(f"score_table={args.score_table}")
    return 0 if not failed_ids and ok_ids == selected_ids else 1


if __name__ == "__main__":
    raise SystemExit(main())
