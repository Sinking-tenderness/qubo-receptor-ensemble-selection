"""Audit the frozen inputs and compute budget for Stage102 before docking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def nonempty_lines(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage102_prospective_marginal_learning.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    loaded = {}
    for name, spec in config["parents"].items():
        path = root / spec["path"]
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"parent hash mismatch: {name}")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    if loaded["stage101_audit"]["status"] != "stage101_independent_audit_ok":
        raise ValueError("Stage101 independent audit is not valid")
    if loaded["stage101_result"]["decision"]["hardware_authorized"]:
        raise ValueError("Stage101 unexpectedly authorized hardware")
    egfr_ids = loaded["egfr_redocking_adjudication"]["passing_receptor_ids"]
    fa10_ids = loaded["fa10_redocking_adjudication"]["passing_receptor_ids"]
    if (len(egfr_ids), len(fa10_ids)) != (12, 13):
        raise ValueError("passing receptor counts changed")
    source_counts = {}
    for target, spec in config["phase_a_development_expansion"]["targets"].items():
        active_count = nonempty_lines(root / spec["active_source"])
        decoy_count = nonempty_lines(root / spec["decoy_source"])
        source_counts[target] = {"active": active_count, "decoy": decoy_count}
        panel = config["phase_a_development_expansion"]["ligand_panel_per_target"]
        if active_count < int(panel["active_count"]) or decoy_count < int(panel["decoy_count"]):
            raise ValueError(f"insufficient ligand source for {target}")
    parp1 = loaded["parp1_source_audit"]
    if parp1["status"] != "stage17a_parp1_source_and_active_allocation_ok":
        raise ValueError("PARP1 source gate is not valid")
    expected = config["phase_a_development_expansion"]["docking"]["expected_receptor_ligand_seed_pairs"]
    recomputed = {
        "EGFR": len(egfr_ids) * 600 * 3,
        "FA10": len(fa10_ids) * 600 * 3,
    }
    recomputed["total"] = recomputed["EGFR"] + recomputed["FA10"]
    if recomputed != expected:
        raise ValueError("Stage102 pair budget mismatch")
    readiness = {
        "schema_version": "1.0",
        "status": "stage102_phase_a_readiness_ok",
        "config": {"path": args.config.as_posix(), "sha256": sha256(config_path)},
        "phase_a_targets": {
            "EGFR": {"passing_receptor_count": len(egfr_ids), "passing_receptor_ids": egfr_ids},
            "FA10": {"passing_receptor_count": len(fa10_ids), "passing_receptor_ids": fa10_ids},
        },
        "source_ligand_counts": source_counts,
        "phase_a_expected_receptor_ligand_seed_pairs": recomputed,
        "phase_b_target": "PARP1",
        "phase_b_source_gate": parp1["status"],
        "phase_b_released": False,
        "new_docking_jobs_started": 0,
        "parp1_fresh_validation_rows_read": 0,
        "quantum_hardware_jobs": 0,
        "next_action": "Build the deterministic EGFR and FA10 600-ligand Phase A input bundle; do not prepare PARP1 yet."
    }
    output = root / config["outputs"]["readiness_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(readiness, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(readiness, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
