"""Independently audit the Stage32c PPARG MD-pair failure diagnosis."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.stage32b_common import descriptor, read_csv, read_json, sha256


def audit(config_path: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    result_path = root / config["outputs"]["result_json"]
    result = read_json(result_path)
    if result.get("status") != "stage32c_pparg_md_pair_failure_diagnostic_complete":
        raise ValueError("unexpected Stage32c result status")
    if result["config"]["sha256"] != sha256(config_path):
        raise ValueError("Stage32c config hash differs")
    if read_json(root / config["inputs"]["stage32b_audit"]).get("status") != "stage32b_pparg_md_pair_fresh_validation_audit_ok":
        raise ValueError("Stage32b audit gate differs")
    for record in result["outputs"].values():
        path = root / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"] or path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Stage32c descriptor differs: {record['path']}")

    ligands = read_csv(root / config["outputs"]["ligand_diagnostic_csv"])
    scenarios = read_csv(root / config["outputs"]["scenario_summary_csv"])
    if len(ligands) != 1576 or Counter(row["label"] for row in ligands) != Counter({"active": 75, "decoy": 1501}):
        raise ValueError("Stage32c ligand coverage differs")
    if len(scenarios) != 10 or Counter(row["split"] for row in scenarios) != Counter({"train": 5, "validation": 5}):
        raise ValueError("Stage32c scenario coverage differs")
    primary = next(row for row in scenarios if row["split"] == "validation" and row["scenario"] == "primary")
    active = [row for row in ligands if row["label"] == "active"]
    decoy = [row for row in ligands if row["label"] == "decoy"]

    def truth(value: str) -> bool:
        return value.lower() == "true"

    recomputed = {
        "active_extra_win_count": sum(truth(row["extra_receptor_wins"]) for row in active),
        "decoy_extra_win_count": sum(truth(row["extra_receptor_wins"]) for row in decoy),
        "active_mean_fraction_improvement": statistics.fmean(float(row["fraction_improvement"]) for row in active),
        "decoy_mean_fraction_improvement": statistics.fmean(float(row["fraction_improvement"]) for row in decoy),
        "pair_top1pct_new_decoy_count": sum(truth(row["pair_top1pct"]) and not truth(row["single_top1pct"]) for row in decoy),
        "pair_top1pct_lost_active_count": sum(truth(row["single_top1pct"]) and not truth(row["pair_top1pct"]) for row in active),
        "pair_top5pct_new_decoy_count": sum(truth(row["pair_top5pct"]) and not truth(row["single_top5pct"]) for row in decoy),
        "pair_top5pct_lost_active_count": sum(truth(row["single_top5pct"]) and not truth(row["pair_top5pct"]) for row in active),
    }
    maximum_difference = 0.0
    for key, value in recomputed.items():
        maximum_difference = max(maximum_difference, abs(float(primary[key]) - float(value)), abs(float(result["primary_validation"][key]) - float(value)))
    active_win_rate = recomputed["active_extra_win_count"] / 75
    decoy_win_rate = recomputed["decoy_extra_win_count"] / 1501
    maximum_difference = max(maximum_difference, abs(active_win_rate - float(primary["active_extra_win_rate"])), abs(decoy_win_rate - float(primary["decoy_extra_win_rate"])))
    all_seed_negative = all(float(next(row for row in scenarios if row["split"] == "validation" and row["scenario"] == seed)["pair_minus_single_bedroc20"]) < 0 for seed in ("seed0", "seed1", "seed2"))
    nonselective = float(primary["pair_minus_single_bedroc20"]) < 0 and all_seed_negative and (
        active_win_rate <= decoy_win_rate
        or recomputed["active_mean_fraction_improvement"] <= recomputed["decoy_mean_fraction_improvement"]
    )
    if nonselective != result["mechanism"]["nonselective_decoy_promotion"] or nonselective != result["decision"]["stop_frozen_min_aggregation_md_pair_efficacy_route"]:
        raise ValueError("Stage32c mechanism or route decision differs")
    checks = {
        "config_and_output_descriptors_verified": True,
        "stage32b_audit_gate_verified": True,
        "all_1576_ligand_diagnostics_verified": True,
        "all_10_split_scenario_rows_verified": True,
        "primary_win_rates_and_improvements_recomputed": maximum_difference <= 1e-12,
        "top1_and_top5_entry_exit_counts_recomputed": True,
        "all_three_seed_direction_checks_recomputed": all_seed_negative,
        "nonselective_decoy_promotion_recomputed": nonselective,
        "locked_test_docking_and_hardware_boundary_verified": all(int(result["data_boundary"][key]) == 0 for key in ("locked_test_rows_read", "new_docking_jobs", "quantum_hardware_jobs")),
    }
    if not all(checks.values()):
        raise ValueError(f"Stage32c audit failed: {checks}")
    audit_result = {
        "schema_version": "1.0",
        "status": "stage32c_pparg_md_pair_failure_diagnostic_audit_ok",
        "config": descriptor(root, config_path),
        "result": descriptor(root, result_path),
        "checks": checks,
        "coverage": {"ligand_count": 1576, "scenario_row_count": 10, "maximum_recomputed_abs_difference": maximum_difference},
        "mechanism": result["mechanism"],
        "decision": result["decision"],
        "data_boundary": result["data_boundary"],
        "interpretation_boundary": config["interpretation_boundary"],
    }
    output = root / config["outputs"]["audit_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit_result, indent=2, sort_keys=True))
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/stage32c_pparg_md_pair_failure_diagnostic.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    audit(args.config, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
