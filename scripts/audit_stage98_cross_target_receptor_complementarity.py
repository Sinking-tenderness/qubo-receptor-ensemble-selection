from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    result = json.loads((root / "data/stage98_cross_target_receptor_complementarity_result.json").read_text(encoding="utf-8"))
    fold_rows = read_csv(root / "results/runs/stage98_cross_target_receptor_complementarity/fold_metrics.csv")
    target_rows = read_csv(root / "results/runs/stage98_cross_target_receptor_complementarity/target_summary.csv")
    pair_rows = read_csv(root / "results/runs/stage98_cross_target_receptor_complementarity/receptor_pair_complementarity.csv")
    if result["status"] != "stage98_cross_target_receptor_complementarity_complete":
        raise ValueError("unexpected Stage98 status")
    if result["target_ids"] != ["BACE1", "MK14", "PPARA", "PPARD", "PPARG"]:
        raise ValueError("target coverage mismatch")
    if len(fold_rows) != 300 or len(target_rows) != 60 or len(pair_rows) != 5727:
        raise ValueError("output row count mismatch")
    if {row["method"] for row in fold_rows} != {"mean_score", "complementarity", "oracle_train", "random"}:
        raise ValueError("method coverage mismatch")
    if {int(row["ensemble_size"]) for row in fold_rows} != {1, 2, 3}:
        raise ValueError("ensemble-size coverage mismatch")
    if result["audit"] != {"selector_labels_allowed": False, "oracle_train_is_evaluation_only": True, "new_docking_jobs": 0, "quantum_hardware_jobs": 0, "synthetic_scores": 0, "fresh_validation_rows": 0}:
        raise ValueError("data-boundary audit mismatch")
    if result["gate"]["passes"] is not False or result["gate"]["positive_target_count"] != 1:
        raise ValueError("expected conservative NO-GO gate changed")
    audit = {"schema_version": "1.0", "status": "stage98_audit_ok", "targets": result["target_ids"], "fold_rows": len(fold_rows), "target_rows": len(target_rows), "pair_rows": len(pair_rows), "selector_labels_used": False, "new_docking_jobs": 0, "quantum_hardware_jobs": 0, "go_gate_passes": False}
    output = root / "data/stage98_cross_target_receptor_complementarity_audit.json"
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
