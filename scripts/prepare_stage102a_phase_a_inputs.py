"""Prepare Stage102A EGFR and FA10 ligand PDBQT inputs with checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.experimental.unidock import prepare_development_ligand_inputs as common
from scripts.batch_prepare_ligand_pdbqt import find_meeko_script


def target_config_hash(config_path: Path, target: str) -> str:
    payload = config_path.read_bytes() + b"\nTARGET=" + target.encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def prepare_target(root: Path, config_path: Path, target: str, resume: bool) -> dict[str, Any]:
    config = common.read_json(config_path)
    target_spec = config["phase_a_development_expansion"]["targets"][target]
    source_path = root / f"data/processed/stage102a_{target.lower()}_phase_a_ligand_manifest.csv"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    rows = common.read_csv(source_path)
    if len(rows) != 600 or Counter(row["label"] for row in rows) != Counter({"active": 120, "decoy": 480}):
        raise ValueError(f"{target} Stage102A allocation differs")
    if {row["target_id"] for row in rows} != {target} or {row["split"] for row in rows} != {"train"}:
        raise ValueError(f"{target} Stage102A allocation boundary differs")
    run_directory = root / f"results/runs/stage102a_{target.lower()}_phase_a_inputs"
    sdf_directory = run_directory / "sdf"
    pdbqt_directory = run_directory / "pdbqt"
    checkpoint_directory = run_directory / "checkpoints"
    for directory in (sdf_directory, pdbqt_directory, checkpoint_directory):
        directory.mkdir(parents=True, exist_ok=True)
    signature = target_config_hash(config_path, target)
    meeko_script = find_meeko_script()
    tasks = [
        {
            "row": row,
            "root": str(root),
            "sdf_directory": str(sdf_directory),
            "pdbqt_directory": str(pdbqt_directory),
            "checkpoint_directory": str(checkpoint_directory),
            "meeko_script": str(meeko_script),
            "index": index,
            "resume": resume,
            "overwrite": False,
            "base_seed": 20260821 if target == "EGFR" else 20260822,
            "seed_offsets": [0, 1000003, 2000003],
            "target_id": target,
            "config_sha256": signature,
        }
        for index, row in enumerate(rows)
    ]
    prepared: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(common.prepare_one, task): int(task["index"]) for task in tasks}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            prepared[futures[future]] = future.result()
            if completed % 25 == 0 or completed == len(tasks):
                print(f"{target} prepared_or_resumed {completed}/{len(tasks)}", flush=True)
    output_rows = [prepared[index] for index in range(len(rows))]
    if any(row["pdbqt_status"] != "ok" for row in output_rows):
        raise ValueError(f"{target} contains a failed PDBQT")
    manifest_path = root / f"data/processed/stage102a_{target.lower()}_phase_a_pdbqt_manifest.csv"
    common.write_csv(manifest_path, output_rows)
    summary = {
        "schema_version": "1.0",
        "status": "stage102a_phase_a_inputs_ok",
        "target_id": target,
        "ligand_count": len(output_rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in output_rows).items())),
        "preparation_variant_counts": dict(sorted(Counter(row["preparation_variant"] for row in output_rows).items())),
        "resumed_ligand_count": sum(row["resume_status"] == "validated_checkpoint" for row in output_rows),
        "prepared_ligand_count_this_invocation": sum(row["resume_status"] != "validated_checkpoint" for row in output_rows),
        "future_receptor_count": int(target_spec["passing_receptor_count"]),
        "future_seed_count": 3,
        "future_pair_count": int(target_spec["passing_receptor_count"]) * 600 * 3,
        "manifest": {"path": manifest_path.relative_to(root).as_posix(), "sha256": common.file_sha256(manifest_path)},
        "data_boundary": {"fresh_validation_rows_read": 0, "locked_test_rows_read": 0, "docking_scores_read": 0},
    }
    summary_path = root / f"data/stage102a_{target.lower()}_phase_a_input_summary.json"
    common.write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage102_prospective_marginal_learning.json"))
    parser.add_argument("--target", action="append", choices=("EGFR", "FA10"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = (root / args.config).resolve()
    targets = args.target or ["EGFR", "FA10"]
    summaries = [prepare_target(root, config_path, target, args.resume) for target in targets]
    return 0 if all(summary["status"] == "stage102a_phase_a_inputs_ok" for summary in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
