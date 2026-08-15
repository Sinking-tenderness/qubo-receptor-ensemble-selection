"""Independently audit the frozen Stage104 quality-plateau replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage104_quality_plateau_egfr_fa10_replay as stage104


TARGETS = ("EGFR", "FA10")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_close(actual: float, expected: float, message: str) -> None:
    require(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), message)


def as_bool(value: str) -> bool:
    require(value in {"True", "False"}, f"expected Boolean CSV value, received {value!r}")
    return value == "True"


def audit(root: Path, config_path: Path) -> dict[str, Any]:
    config = read_json(root / config_path)
    stage104.validate_hashes(root, config)
    outputs = config["outputs"]
    result = read_json(root / outputs["result_json"])
    rows = read_csv(root / outputs["fold_csv"])
    summaries = read_csv(root / outputs["target_csv"])

    require(result["status"] == "stage104_quality_plateau_egfr_fa10_replay_complete", "unexpected result status")
    require(tuple(result["target_ids"]) == TARGETS, "unexpected target IDs")
    require(len(rows) == 40, f"expected 40 fold rows, received {len(rows)}")
    require(len(summaries) == 4, f"expected 4 target summaries, received {len(summaries)}")
    expected_keys = {(target, fold, k, solver) for target in TARGETS for fold in range(1, 6) for k in (2, 3) for solver in ("pair_off_baseline", "continuous_milp_certificate")}
    observed_keys = {(row["target_id"], int(row["outer_fold"]), int(row["subset_size"]), row["solver_id"]) for row in rows}
    require(observed_keys == expected_keys and len(observed_keys) == len(rows), "incomplete or duplicate Stage104 coverage")
    for row in rows:
        require(not as_bool(row["uses_outer_labels_for_selection"]), "outer labels leaked to Stage104 selection")
        require(float(row["train_quality_margin"]) >= -1e-10, "selected subset violates frozen quality floor")
        require(math.isfinite(float(row["holdout_robust_bedroc"])), "non-finite holdout BEDROC")
        if row["solver_id"] == "continuous_milp_certificate":
            require(float(row["milp_gap"]) == 0.0, "MILP certificate has nonzero gap")

    baseline = {(row["target_id"], int(row["outer_fold"]), int(row["subset_size"])): row for row in rows if row["solver_id"] == "pair_off_baseline"}
    recomputed_checks: list[bool] = []
    for summary in summaries:
        target = summary["target_id"]
        k = int(summary["subset_size"])
        selected = [row for row in rows if row["target_id"] == target and int(row["subset_size"]) == k and row["solver_id"] == "continuous_milp_certificate"]
        gains = [float(row["holdout_robust_bedroc"]) - float(baseline[(target, int(row["outer_fold"]), k)]["holdout_robust_bedroc"]) for row in selected]
        reductions = [float(baseline[(target, int(row["outer_fold"]), k)]["stable_redundancy_mean"]) - float(row["stable_redundancy_mean"]) for row in selected]
        require_close(float(summary["mean_gain_over_pair_off"]), sum(gains) / len(gains), "summary gain does not reproduce")
        require_close(float(summary["mean_stable_redundancy_reduction"]), sum(reductions) / len(reductions), "summary redundancy does not reproduce")
        recomputed_checks.append(float(summary["mean_gain_over_pair_off"]) >= -0.01 and float(summary["mean_stable_redundancy_reduction"]) >= 0.0)
    require(result["transfer_checks"]["cell_count"] == len(recomputed_checks), "transfer cell count mismatch")
    require(result["transfer_checks"]["passing_cell_count"] == sum(recomputed_checks), "transfer passing count mismatch")
    require(result["transfer_checks"]["all_target_k_cells_quality_noninferior_and_redundancy_nonnegative"] == all(recomputed_checks), "transfer decision mismatch")
    require(result["decision"]["new_target_protocol_authorized"] is False, "Stage104 incorrectly authorized a new protocol")
    require(result["decision"]["parp1_released"] is False, "Stage104 incorrectly released PARP1")
    require(result["decision"]["quantum_hardware_authorized"] is False, "Stage104 incorrectly authorized hardware")
    require(all(value == 0 for value in result["data_boundary"].values()), "Stage104 data boundary breached")
    return {
        "schema_version": "1.0",
        "status": "stage104_independent_audit_ok",
        "target_count": len(TARGETS),
        "fold_metric_count": len(rows),
        "target_summary_count": len(summaries),
        "transfer_passing_cell_count": sum(recomputed_checks),
        "transfer_cell_count": len(recomputed_checks),
        "outer_labels_used_by_selector": False,
        "parp1_released": False,
        "quantum_hardware_authorized": False,
        "data_boundary": result["data_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage104_quality_plateau_egfr_fa10_replay.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    record = audit(root, args.config)
    config = read_json(root / args.config)
    output = root / config["outputs"]["audit_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
