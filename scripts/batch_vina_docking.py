"""Run AutoDock Vina for a ligand PDBQT manifest and write a long score table."""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import time
from pathlib import Path


REQUIRED_COLUMNS = {"ligand_id", "label", "target_id", "pdbqt_status", "pdbqt_path"}
VINA_CONFIG_KEYS = {
    "center_x",
    "center_y",
    "center_z",
    "size_x",
    "size_y",
    "size_z",
    "exhaustiveness",
    "num_modes",
}


def validate_columns(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ValueError("input manifest has no header")
    missing = REQUIRED_COLUMNS.difference(fieldnames)
    if missing:
        raise ValueError(f"input manifest is missing required columns: {sorted(missing)}")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_columns(reader.fieldnames)
        return list(reader)


def read_vina_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", maxsplit=1)]
            values[key] = value
    missing = VINA_CONFIG_KEYS.difference(values)
    if missing:
        raise ValueError(f"Vina config is missing required keys: {sorted(missing)}")
    return values


def safe_filename(text: str) -> str:
    keep = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def select_rows(
    rows: list[dict[str, str]],
    max_ligands: int | None,
    sample_per_label: int | None,
    sample_seed: int,
) -> list[dict[str, str]]:
    ok_rows = [row for row in rows if row["pdbqt_status"] == "ok"]
    if sample_per_label is not None:
        rng = random.Random(sample_seed)
        selected: list[dict[str, str]] = []
        for label in sorted({row["label"] for row in ok_rows}):
            label_rows = [row for row in ok_rows if row["label"] == label]
            if sample_per_label > len(label_rows):
                raise ValueError(
                    f"requested {sample_per_label} {label} rows, but only {len(label_rows)} are available"
                )
            selected.extend(rng.sample(label_rows, sample_per_label))
        return sorted(selected, key=lambda row: row["ligand_id"])
    if max_ligands is not None:
        return ok_rows[:max_ligands]
    return ok_rows


def get_vina_version(vina_exe: Path) -> str:
    completed = subprocess.run(
        [str(vina_exe), "--version"],
        text=True,
        capture_output=True,
        check=False,
    )
    version = (completed.stdout or completed.stderr).strip()
    return version.replace("\n", " ")


def parse_vina_modes(stdout: str) -> list[dict[str, object]]:
    modes: list[dict[str, object]] = []
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            pose_rank = int(parts[0])
            docking_score = float(parts[1])
            rmsd_lb = float(parts[2])
            rmsd_ub = float(parts[3])
        except ValueError:
            continue
        modes.append(
            {
                "pose_rank": pose_rank,
                "docking_score": docking_score,
                "vina_rmsd_lb_from_best": rmsd_lb,
                "vina_rmsd_ub_from_best": rmsd_ub,
            }
        )
    return modes


def result_rows_for_modes(
    row: dict[str, str],
    receptor_id: str,
    modes: list[dict[str, object]],
    status: str,
    message: str,
    runtime_seconds: float | str,
    seed: int,
    software_version: str,
    output_pose: Path,
    log_path: Path,
) -> list[dict[str, object]]:
    return [
        {
            "target_id": row["target_id"],
            "receptor_id": receptor_id,
            "ligand_id": row["ligand_id"],
            "label": row["label"],
            **mode,
            "status": status,
            "message": message,
            "runtime_seconds": runtime_seconds,
            "seed": seed,
            "software_version": software_version,
            "pose_path": output_pose.as_posix(),
            "log_path": log_path.as_posix(),
        }
        for mode in modes
    ]


def build_vina_command(
    vina_exe: Path,
    receptor: Path,
    ligand: Path,
    output_pose: Path,
    config: dict[str, str],
    seed: int,
) -> list[str]:
    return [
        str(vina_exe),
        "--receptor",
        str(receptor),
        "--ligand",
        str(ligand),
        "--center_x",
        config["center_x"],
        "--center_y",
        config["center_y"],
        "--center_z",
        config["center_z"],
        "--size_x",
        config["size_x"],
        "--size_y",
        config["size_y"],
        "--size_z",
        config["size_z"],
        "--exhaustiveness",
        config["exhaustiveness"],
        "--num_modes",
        config["num_modes"],
        "--seed",
        str(seed),
        "--out",
        str(output_pose),
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
