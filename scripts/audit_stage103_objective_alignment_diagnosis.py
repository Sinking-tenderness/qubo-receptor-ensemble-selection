"""Independently audit the Stage103 objective-alignment diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_stage103_objective_alignment_diagnosis as stage103


TARGETS = ("BACE1", "EGFR", "FA10", "MK14", "PPARA", "PPARD", "PPARG")


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
    stage103.validate_parent_hashes(root, config)
    outputs = config["outputs"]
    result = read_json(root / outputs["result_json"])
    folds = read_csv(root / outputs["fold_csv"])
    subsets = read_csv(root / outputs["subset_csv"])
    summaries = read_csv(root / outputs["target_csv"])

    require(result["status"] == "stage103_objective_alignment_diagnosis_complete", "unexpected result status")
    require(tuple(result["target_ids"]) == TARGETS, "unexpected target IDs")
    require(len(folds) == 105, f"expected 105 fold rows, received {len(folds)}")
    require(len(subsets) == 105, f"expected 105 selected rows, received {len(subsets)}")
    require(len(summaries) == 21, f"expected 21 target summaries, received {len(summaries)}")
    expected_keys = {(target, fold, k) for target in TARGETS for fold in range(1, 6) for k in (1, 2, 3)}
    fold_keys = {(row["target_id"], int(row["outer_fold"]), int(row["k"])) for row in folds}
    subset_keys = {(row["target_id"], int(row["outer_fold"]), int(row["k"])) for row in subsets}
    require(fold_keys == expected_keys and len(fold_keys) == len(folds), "incomplete or duplicate fold coverage")
    require(subset_keys == expected_keys and len(subset_keys) == len(subsets), "incomplete or duplicate selected-subset coverage")

    numeric_fields = (
        "qubo_vs_train_primary_spearman",
        "qubo_vs_outer_primary_spearman",
        "train_primary_vs_outer_primary_spearman",
        "train_qubo_optimum_objective",
        "train_qubo_optimum_train_primary_bedroc20",
        "train_qubo_optimum_outer_primary_bedroc20",
        "outer_primary_oracle_bedroc20",
        "outer_primary_regret_of_train_qubo_optimum",
    )
    for row in folds:
        require(not as_bool(row["uses_outer_labels_for_selection"]), "outer labels leaked to Stage103 selector")
        require(int(row["subset_count"]) >= 1, "invalid subset count")
        require(all(math.isfinite(float(row[field])) for field in numeric_fields), "non-finite fold diagnostic")
        require(float(row["outer_primary_regret_of_train_qubo_optimum"]) >= -1e-12, "negative oracle regret")
    require(all(not as_bool(row["uses_outer_labels_for_selection"]) for row in subsets), "selected subset used outer labels")

    k2_summaries = [row for row in summaries if int(row["k"]) == 2]
    require(len(k2_summaries) == 7, "missing k=2 target summaries")
    median_train = float(np.median([float(row["mean_qubo_vs_train_primary_spearman"]) for row in k2_summaries]))
    median_outer = float(np.median([float(row["mean_qubo_vs_outer_primary_spearman"]) for row in k2_summaries]))
    thresholds = config["diagnostics"]["correlation_interpretation_thresholds"]
    stored = result["summary"]
    require_close(stored["k2_target_median_qubo_vs_train_primary_spearman"], median_train, "train median does not reproduce")
    require_close(stored["k2_target_median_qubo_vs_outer_primary_spearman"], median_outer, "outer median does not reproduce")
    require(stored["k2_train_alignment_supported"] == (median_train >= float(thresholds["train_alignment_supported_if_median_spearman_at_least"])), "train alignment interpretation mismatch")
    require(stored["k2_outer_alignment_supported"] == (median_outer >= float(thresholds["outer_alignment_supported_if_median_spearman_at_least"])), "outer alignment interpretation mismatch")
    require(stored["k2_outer_alignment_failure_supported"] == (median_outer < float(thresholds["outer_alignment_failure_if_median_spearman_below"])), "outer failure interpretation mismatch")

    require(result["decision"]["replacement_objective_authorized"] is False, "Stage103 incorrectly authorized retuning")
    require(result["decision"]["parp1_released"] is False, "Stage103 incorrectly released PARP1")
    require(result["decision"]["quantum_hardware_authorized"] is False, "Stage103 incorrectly authorized hardware")
    require(all(value == 0 for value in result["data_boundary"].values()), "Stage103 data boundary breached")
    return {
        "schema_version": "1.0",
        "status": "stage103_independent_audit_ok",
        "target_count": len(TARGETS),
        "fold_alignment_count": len(folds),
        "selected_subset_count": len(subsets),
        "target_summary_count": len(summaries),
        "k2_target_median_qubo_vs_train_primary_spearman": median_train,
        "k2_target_median_qubo_vs_outer_primary_spearman": median_outer,
        "outer_labels_used_by_selector": False,
        "parp1_released": False,
        "quantum_hardware_authorized": False,
        "data_boundary": result["data_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage103_objective_alignment_diagnosis.json"))
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
