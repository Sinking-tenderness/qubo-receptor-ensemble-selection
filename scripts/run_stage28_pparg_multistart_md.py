"""Orchestrate resumable Stage28 PPARG multi-start OpenMM execution."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage21_structure_aware_qubo import read_csv, read_json, rooted


def status_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="ascii")).get("status") == "ok"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def invoke(arguments: list[str], root: Path) -> None:
    print("+ " + " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=root, check=True)


def build_system(root: Path, row: dict[str, str]) -> None:
    python = sys.executable
    protocol = read_json(rooted(root, row["protocol_config"]))
    planned = protocol["planned_outputs"]
    system_manifest = rooted(root, row["system_manifest"])
    if not status_ok(system_manifest):
        arguments = [
            python, "scripts/build_openmm_system.py",
            "--protocol", row["protocol_config"],
            "--manifest-output", planned["system_manifest"],
            "--solvated-pdb-output", planned["solvated_pdb"],
            "--system-xml-output", planned["system_xml"],
        ]
        if any(rooted(root, planned[key]).exists() for key in ("system_manifest", "solvated_pdb", "system_xml")):
            arguments.append("--overwrite")
        invoke(arguments, root)


def run_start(root: Path, row: dict[str, str]) -> None:
    python = sys.executable
    build_system(root, row)
    equilibration_manifest = rooted(root, row["equilibration_manifest"])
    if not status_ok(equilibration_manifest):
        arguments = [python, "scripts/run_openmm_equilibration.py", "--config", row["equilibration_config"]]
        if rooted(root, read_json(rooted(root, row["equilibration_config"]))["outputs"]["progress_json"]).exists():
            arguments.append("--resume")
        invoke(arguments, root)
    production_manifest = rooted(root, row["production_manifest"])
    if not status_ok(production_manifest):
        arguments = [python, "scripts/run_openmm_production.py", "--config", row["production_config"]]
        if rooted(root, read_json(rooted(root, row["production_config"]))["outputs"]["progress_json"]).exists():
            arguments.append("--resume")
        invoke(arguments, root)
    qc_summary = rooted(root, row["trajectory_qc_summary"])
    if not status_ok(qc_summary):
        invoke([python, "scripts/analyze_md_trajectory.py", "--config", row["trajectory_qc_config"], "--overwrite"], root)


def run(config_path: Path, root: Path, start_id: str | None, skip_collect: bool) -> None:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    rows = read_csv(rooted(root, config["runtime"]["start_manifest"]))
    if start_id:
        rows = [row for row in rows if row["conformer_id"] == start_id]
        if not rows:
            raise ValueError(f"unknown Stage28 start ID: {start_id}")
    for position, row in enumerate(rows, start=1):
        print(f"Stage28 system preflight {position}/{len(rows)}: {row['conformer_id']}", flush=True)
        build_system(root, row)
    for position, row in enumerate(rows, start=1):
        print(f"Stage28 start {position}/{len(rows)}: {row['conformer_id']}", flush=True)
        run_start(root, row)
    if not start_id and not skip_collect:
        invoke([sys.executable, "scripts/collect_stage28_pparg_md_ensemble.py", "--config", config_path.relative_to(root).as_posix(), "--overwrite"], root)
        invoke([
            sys.executable,
            "scripts/audit_stage28_pparg_multistart_md_ensemble.py",
            "--config",
            config_path.relative_to(root).as_posix(),
            "--output",
            str(config["outputs"]["audit_json"]),
        ], root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage28_pparg_multistart_md_ensemble.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--start-id")
    parser.add_argument("--skip-collect", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.start_id, args.skip_collect)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
