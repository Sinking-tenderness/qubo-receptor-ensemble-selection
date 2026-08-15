"""Independently audit the completed Stage43 PPARG MD-96 score matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path)
    outputs = config["outputs"]
    summary_path = root / outputs["summary_json"]
    score_path = root / outputs["scores_csv"]
    batch_path = root / outputs["batch_runs_csv"]
    receptor_path = root / outputs["prepared_receptor_manifest"]
    ligand_path = root / config["inputs"]["stage32_ligand_manifest"]
    summary = read_json(summary_path)
    if summary.get("status") != "stage43_pparg_md96_unidock_matrix_ok":
        raise ValueError("Stage43 matrix summary is incomplete")
    receptors = read_csv(receptor_path)
    ligands = read_csv(ligand_path)
    scores = read_csv(score_path)
    batches = read_csv(batch_path)
    receptor_ids = {row["conformer_id"] for row in receptors}
    ligand_ids = {row["ligand_id"] for row in ligands}
    seed_ids = {row["seed_id"] for row in config["inputs"]["seeds"]}
    expected = {(seed, receptor, ligand) for seed in seed_ids for receptor in receptor_ids for ligand in ligand_ids}
    observed = {(row["seed_id"], row["receptor_id"], row["ligand_id"]) for row in scores}
    if len(receptors) != 96 or len(ligands) != 160 or observed != expected or len(scores) != 46080:
        raise ValueError("Stage43 score key coverage differs")
    if len(batches) != 288 or Counter(row["evidence_role"] for row in batches) != Counter({"historical_stage32_reuse": 48, "new_stage43_docking": 240}):
        raise ValueError("Stage43 batch provenance differs")
    if any(row["status"] != "ok" for row in scores + batches):
        raise ValueError("Stage43 contains a failed score or batch")
    if {row["target_id"] for row in scores} != {"PPARG"}:
        raise ValueError("Stage43 final score target metadata differs")
    if any(not math.isfinite(float(row["gpu_score"])) or abs(float(row["gpu_score"])) > 1000 for row in scores):
        raise ValueError("Stage43 contains an invalid score")
    if any(int(row["unresolved_warning_event_count"]) != 0 or int(row["pose_integrity_failure_count"]) != 0 for row in batches):
        raise ValueError("Stage43 technical integrity gate failed")
    rescue_rows = [row for row in batches if row.get("technical_rescue_applied", "False").lower() == "true"]
    if len(rescue_rows) > 1:
        raise ValueError("Stage43 used more than one technical rescue batch")
    if rescue_rows and (rescue_rows[0]["seed_id"], rescue_rows[0]["receptor_id"]) != ("seed2", "PPARG_MD_00177_8CPI"):
        raise ValueError("Stage43 technical rescue identity differs")
    result = {
        "schema_version": "1.0",
        "status": "stage43_pparg_md96_unidock_matrix_independent_audit_ok",
        "counts": {
            "receptors": 96, "ligands": 160, "seeds": 3,
            "batches": 288, "pairs": 46080,
            "historical_pairs": 7680, "new_pairs": 38400,
        },
        "integrity": {
            "unique_key_coverage_complete": True,
            "unresolved_warning_event_count": 0,
            "pose_integrity_failure_count": 0,
            "historical_stage32_audit_reused": True,
            "technical_rescue_batch_count": len(rescue_rows),
        },
        "data_boundary": {
            "train_rows_read": 160, "fresh_validation_rows_read": 0,
            "test_rows_read": 0, "quantum_hardware_jobs": 0,
        },
        "inputs": {
            "summary_sha256": sha256(summary_path), "scores_sha256": sha256(score_path),
            "batch_runs_sha256": sha256(batch_path), "receptor_manifest_sha256": sha256(receptor_path),
        },
        "next_gate": "apply the unchanged Stage42f QUBO and frozen solver benchmark",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output_path = root / outputs["audit_json"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage43_pparg_md96_rank_sensitive_replication.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    run(config_path, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
