"""V5 nested outer-fold evaluation of cardinality decision rules.

Same-caliber comparison defined in docs/adaptive_k_failure_analysis_and_improvement_zh.md (P1/V5):
for each existing balanced `outer_fold`, every decision rule chooses a receptor
subset using train rows only, and all subsets are scored identically on the
held-out fold with mean_score + BEDROC20.

Rules compared per outer fold:
- fixed_k{k}: exact QUBO solved on train with target_size=k (baseline k=1 included);
  post-hoc max over k is the fold oracle.
- adaptive_controller: the production risk_adjusted_oof policy run on train
  inner scaffold folds; final subset is S[selected_k] from the enumeration.
- enumerate_select: argmax over k of the mean inner-CV OOF utility on train,
  ties broken toward smaller k; no bootstrap gates.

Run against one target:
    python scripts/nested_outer_k_evaluation.py \
      --matrix results/runs/mk14_adaptive_remote/matrices/primary_median_matrix.csv \
      --prepared-manifest results/runs/mk14_adaptive_remote/prepared_ligands.csv \
      --target-id MK14 --output-dir results/nested_v5/mk14
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qubo_receptor_ensemble.adaptive_cardinality import estimate_adaptive_cardinality
from qubo_receptor_ensemble.screening import bedroc, roc_auc_pairwise
from qubo_receptor_ensemble.solvers import build_problem, solve_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--inner-fold-count", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--required-probability", type=float, default=0.9)
    parser.add_argument("--minimum-effect", type=float, default=0.0)
    parser.add_argument("--lower-quantile", type=float, default=0.05)
    parser.add_argument("--cost-per-receptor", type=float, default=0.0)
    parser.add_argument("--selection-tie-tolerance", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--redundancy-weight", type=float, default=0.25,
        help="QUBO redundancy weight; keep identical to production configs",
    )
    return parser.parse_args()


def load_rows(matrix_path: str, manifest_path: str) -> tuple[list[dict[str, object]], list[str]]:
    with open(manifest_path, encoding="utf-8") as fh:
        meta = {
            str(row["ligand_id"]): row
            for row in csv.DictReader(fh)
        }
    with open(matrix_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        receptors = [
            column for column in (reader.fieldnames or [])
            if column not in {"target_id", "ligand_id", "label", "selection_role"}
        ]
        rows: list[dict[str, object]] = []
        for record in reader:
            ligand_id = str(record["ligand_id"])
            entry = meta.get(ligand_id)
            if entry is None:
                raise SystemExit(f"ligand {ligand_id} missing from prepared manifest")
            row: dict[str, object] = {
                "ligand_id": ligand_id,
                "label": str(record["label"]),
                "scaffold_smiles": str(entry["scaffold_smiles"]),
                "outer_fold": int(float(entry["outer_fold"])),
            }
            for receptor in receptors:
                row[receptor] = float(record[receptor])
            rows.append(row)
    labels = {str(row["label"]) for row in rows}
    if labels != {"active", "decoy"}:
        raise SystemExit(f"matrix must contain active and decoy rows, found {labels}")
    return rows, receptors


def solve_subset(rows: list[dict[str, object]], receptor_ids: list[str], k: int,
                 redundancy_weight: float) -> list[str]:
    config: dict[str, object] = {
        "type": "receptor_subset",
        "strategy": "qubo",
        "target_size": k,
        "utility_metric": "bedroc",
        "bedroc_alpha": 20.0,
        "utility_normalization": "none",
        "weights": {
            "redundancy": redundancy_weight,
            "count": 0.1,
            "size": 10.0,
        },
        "receptor_ids": list(receptor_ids),
    }
    problem = build_problem(rows, config)
    return list(solve_problem(problem, "exact").subset)


def subset_metrics(rows: list[dict[str, object]], subset: list[str], alpha: float = 20.0) -> tuple[float, float]:
    ranked = sorted(
        (
            (sum(float(row[receptor]) for receptor in subset) / len(subset),
             str(row["ligand_id"]),
             int(str(row["label"]) == "active"))
            for row in rows
        ),
        key=lambda value: (value[0], value[1]),
    )
    labels = [entry[2] for entry in ranked]
    scores = [-entry[0] for entry in ranked]
    return bedroc([{"binary_label": item} for item in labels], alpha), roc_auc_pairwise(labels, scores)


def inner_cv_mean_utility(train_rows: list[dict[str, object]], receptor_ids: list[str],
                          candidate_ks: range, inner_fold_count: int,
                          redundancy_weight: float) -> dict[int, float]:
    """Mean OOF BEDROC per candidate k over deterministic scaffold folds."""

    scaffolds = sorted({str(row["scaffold_smiles"]) for row in train_rows})
    assignment = {
        scaffold: index % inner_fold_count
        for index, scaffold in enumerate(scaffolds)
    }
    totals: dict[int, list[float]] = {k: [] for k in candidate_ks}
    for fold in range(inner_fold_count):
        held = [row for row in train_rows if assignment[str(row["scaffold_smiles"])] == fold]
        part = [row for row in train_rows if assignment[str(row["scaffold_smiles"])] != fold]
        if not held or not part:
            raise SystemExit("inner fold is empty; reduce --inner-fold-count")
        for k in candidate_ks:
            subset = solve_subset(part, receptor_ids, k, redundancy_weight)
            value, _ = subset_metrics(held, subset)
            totals[k].append(value)
    return {k: sum(values) / len(values) for k, values in totals.items()}


def main() -> None:
    args = parse_args()
    rows, receptors = load_rows(args.matrix, args.prepared_manifest)
    candidates = list(range(1, args.candidates + 1))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_config_base: dict[str, object] = {
        "type": "receptor_subset",
        "strategy": "qubo",
        "utility_metric": "bedroc",
        "bedroc_alpha": 20.0,
        "utility_normalization": "none",
        "weights": {
            "redundancy": args.redundancy_weight,
            "count": 0.1,
            "size": 10.0,
        },
    }

    long_rows: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for fold in sorted({int(row["outer_fold"]) for row in rows}):
        train = [row for row in rows if int(row["outer_fold"]) != fold]
        test = [row for row in rows if int(row["outer_fold"]) == fold]
        print(f"[{args.target_id}] outer fold {fold}: train={len(train)} test={len(test)}",
              flush=True)

        subsets: dict[int, list[str]] = {}
        for k in candidates:
            subsets[k] = solve_subset(train, receptors, k, args.redundancy_weight)

        estimator = estimate_adaptive_cardinality(
            train,
            receptors,
            problem_config=dict(problem_config_base),
            solve_subset=lambda part, k: solve_subset(part, receptors, k, args.redundancy_weight),
            solver_backend="exact",
            candidate_ks=candidates,
            scaffold_field="scaffold_smiles",
            inner_fold_count=args.inner_fold_count,
            bootstrap_iterations=args.bootstrap_iterations,
            lower_quantile=args.lower_quantile,
            minimum_effect=args.minimum_effect,
            required_probability=args.required_probability,
            cost_per_receptor=args.cost_per_receptor,
            selection_tie_tolerance=args.selection_tie_tolerance,
            require_rescue_contrast=False,
            rescue_fractions=[0.01, 0.05],
            bedroc_alpha=20.0,
            random_seed=args.random_seed,
            aggregation="mean_score",
        )
        adaptive_k = estimator.selected_k

        inner_means = inner_cv_mean_utility(
            train, receptors, candidates, args.inner_fold_count, args.redundancy_weight
        )
        best_inner = max(inner_means.values())
        enumerate_k = min(
            k for k in candidates if inner_means[k] >= best_inner - 1e-12
        )

        methods: dict[str, tuple[int, list[str]]] = {
            **{f"fixed_k{k}": (k, subsets[k]) for k in candidates},
            "adaptive_controller": (adaptive_k, subsets[adaptive_k]),
            "enumerate_select": (enumerate_k, subsets[enumerate_k]),
        }
        for method, (chosen_k, subset) in sorted(methods.items()):
            score, auc = subset_metrics(test, subset)
            long_rows.append({
                "target_id": args.target_id,
                "outer_fold": fold,
                "method": method,
                "selected_k": chosen_k,
                "test_bedroc_alpha_20": round(score, 6),
                "test_roc_auc": round(auc, 6),
            })
        decisions.append({
            "outer_fold": fold,
            "adaptive_selected_k": adaptive_k,
            "adaptive_stop_reason": estimator.stop_reason,
            "adaptive_evaluated_candidates": list(estimator.evaluated_candidates),
            "enumerate_selected_k": enumerate_k,
            "inner_mean_utilities": {str(k): round(v, 6) for k, v in inner_means.items()},
            "fold_subsets": {str(k): subsets[k] for k in candidates},
        })
        print(f"  adaptive->k={adaptive_k} ({estimator.stop_reason}) | "
              f"enum_select->k={enumerate_k}", flush=True)

    with open(output_dir / "folds_long.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)
    with open(output_dir / "decision_log.json", "w", encoding="utf-8") as fh:
        json.dump({"target_id": args.target_id, "args": vars(args), "folds": decisions},
                  fh, ensure_ascii=False, indent=2)

    oracle_by_fold: dict[int, float] = {}
    single_by_fold: dict[int, float] = {}
    for row in long_rows:
        fold = int(row["outer_fold"])
        if str(row["method"]).startswith("fixed_k"):
            oracle_by_fold[fold] = max(
                oracle_by_fold.get(fold, float("-inf")),
                float(row["test_bedroc_alpha_20"]),
            )
            if str(row["method"]) == "fixed_k1":
                single_by_fold[fold] = float(row["test_bedroc_alpha_20"])

    by_method: dict[str, list[dict[str, object]]] = {}
    for row in long_rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    summary: list[dict[str, object]] = []
    for method in sorted(by_method):
        entries = by_method[method]
        scores = [float(r["test_bedroc_alpha_20"]) for r in entries]
        deltas = [
            score - single_by_fold[int(r["outer_fold"])]
            for r, score in zip(entries, scores)
        ]
        summary.append({
            "method": method,
            "mean_test_bedroc": round(sum(scores) / len(scores), 6),
            "mean_gain_over_single": round(sum(deltas) / len(deltas), 6),
            "worst_fold_gain_over_single": round(min(deltas), 6),
            "selected_ks": "|".join(str(r["selected_k"]) for r in entries),
            "hits_oracle": sum(
                1
                for r, score in zip(entries, scores)
                if score >= oracle_by_fold[int(r["outer_fold"])] - 1e-9
            ),
        })
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\n=== summary ===")
    for entry in summary:
        print(entry)
    print(f"\noutputs written to {output_dir}")


if __name__ == "__main__":
    main()
