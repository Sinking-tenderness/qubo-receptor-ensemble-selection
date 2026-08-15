"""Independently audit the frozen Stage45 PPARG generalization diagnosis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import statistics
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance


def subset(value: str) -> frozenset[str]:
    return frozenset(item for item in value.split("+") if item)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / len(left | right)


def select_metric(
    rows: list[dict[str, str]], fold: int, method: str, size: int
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["scope"] == "outer_holdout"
        and int(row["fold"]) == fold
        and row["method"] == method
        and int(row["subset_size"]) == size
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one Stage44 metric row, got {len(matches)}")
    return matches[0]


def audit(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    result = read_json(root / "data/stage45_pparg_md96_generalization_diagnosis_result.json")
    config = read_json(root / result["config"]["path"])
    if result["status"] != "stage45_pparg_md96_generalization_diagnosis_complete":
        raise ValueError("Stage45 result status is not complete")
    if sha256(root / result["config"]["path"]) != result["config"]["sha256"]:
        raise ValueError("Stage45 config identity mismatch")
    if sha256(root / result["implementation"]["path"]) != result["implementation"]["sha256"]:
        raise ValueError("Stage45 implementation identity mismatch")
    for value in result["outputs"].values():
        if sha256(root / value["path"]) != value["sha256"]:
            raise ValueError(f"Stage45 output identity mismatch: {value['path']}")

    paths = {key: root / value["path"] for key, value in result["outputs"].items()}
    coefficient_stability = read_csv(paths["coefficient_stability_csv"])
    coefficient_correlations = read_csv(paths["coefficient_correlations_csv"])
    landscape = read_csv(paths["landscape_correlations_csv"])
    selections = read_csv(paths["selection_stability_csv"])
    expected_counts = {
        "coefficient_stability": 96,
        "coefficient_correlations": 6,
        "landscape_correlations": 12,
        "selection_stability": 2,
    }
    observed_counts = {
        "coefficient_stability": len(coefficient_stability),
        "coefficient_correlations": len(coefficient_correlations),
        "landscape_correlations": len(landscape),
        "selection_stability": len(selections),
    }
    if observed_counts != expected_counts:
        raise ValueError(f"Stage45 output row counts differ: {observed_counts}")

    summary = result["summary"]
    recomputed = {
        "median_objective_holdout_spearman": statistics.median(
            float(row["objective_vs_holdout_bedroc_spearman"]) for row in landscape
        ),
        "median_train_holdout_bedroc_spearman": statistics.median(
            float(row["train_vs_holdout_bedroc_spearman"]) for row in landscape
        ),
        "median_singleton_fold_spearman": statistics.median(
            float(row["singleton_spearman"]) for row in coefficient_correlations
        ),
        "median_pair_fold_spearman": statistics.median(
            float(row["pair_spearman"]) for row in coefficient_correlations
        ),
    }
    for key, value in recomputed.items():
        if not close(value, float(summary[key])):
            raise ValueError(f"Stage45 summary differs for {key}")

    stage44_metrics = read_csv(root / config["inputs"]["stage44_metrics"]["path"])
    selection_checks: dict[int, dict[str, Any]] = {}
    for size, method in ((3, "exact"), (6, "strong_classical")):
        fold_subsets = [subset(select_metric(stage44_metrics, fold, method, size)["selected_subset"]) for fold in range(4)]
        mean_jaccard = statistics.fmean(
            jaccard(left, right) for left, right in itertools.combinations(fold_subsets, 2)
        )
        recorded = next(row for row in selections if int(row["subset_size"]) == size)
        if not close(mean_jaccard, float(recorded["mean_pairwise_fold_jaccard"])):
            raise ValueError(f"Stage45 k={size} selection stability differs")
        selection_checks[size] = {
            "mean_pairwise_fold_jaccard": mean_jaccard,
            "unique_fold_subset_count": len(set(fold_subsets)),
        }

    k6_gains = []
    for fold in range(4):
        best_single = select_metric(stage44_metrics, fold, "exact", 1)
        k6 = select_metric(stage44_metrics, fold, "strong_classical", 6)
        k6_gains.append(float(k6["robust_bedroc_composite"]) - float(best_single["robust_bedroc_composite"]))
    thresholds = config["diagnostic_thresholds"]
    k6_eligible = (
        statistics.fmean(k6_gains) >= float(thresholds["minimum_k6_mean_holdout_gain"])
        and sum(value > 0 for value in k6_gains) >= int(thresholds["minimum_k6_positive_fold_count"])
        and selection_checks[6]["mean_pairwise_fold_jaccard"]
        >= float(thresholds["minimum_mean_pairwise_fold_jaccard"])
    )
    if k6_eligible != result["decision"]["k6_new_target_preregistration_authorized"]:
        raise ValueError("Stage45 k=6 gate differs")
    boundary = result["data_boundary"]
    if any(boundary[key] != 0 for key in ("fresh_validation_rows_read", "test_rows_read", "new_docking_jobs", "quantum_hardware_jobs")):
        raise ValueError("Stage45 crossed its evidence boundary")

    audit_result = {
        "schema_version": "1.0",
        "status": "stage45_pparg_md96_generalization_diagnosis_independent_audit_ok",
        "audited_result": {
            "path": "data/stage45_pparg_md96_generalization_diagnosis_result.json",
            "sha256": sha256(root / "data/stage45_pparg_md96_generalization_diagnosis_result.json"),
        },
        "row_counts": observed_counts,
        "recomputed_summary": recomputed,
        "selection_checks": {str(key): value for key, value in selection_checks.items()},
        "k6_holdout_gains": k6_gains,
        "k6_gate_recomputed": k6_eligible,
        "evidence_boundary_ok": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(audit_result, indent=2, sort_keys=True))
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stage45_pparg_md96_generalization_diagnosis_audit.json"),
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    audit(args.root, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
