"""Compare screening signals from two receptor score tables.

Thin CLI wrapper; the metric core lives in ``qubo_receptor_ensemble.screening``.
"""



from __future__ import annotations

# --- src bootstrap (bare-checkout import path) ---
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
import argparse
import csv
import json
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

from qubo_receptor_ensemble.screening import ranked_metrics_with_ids

__all__ = ["read_rank1", "ranked_metrics_with_ids"]


def read_rank1(path: Path) -> dict[str, dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        selected: dict[str, dict[str, object]] = {}
        for row in rows:
            if row.get("status") != "ok" or row.get("pose_rank") != "1":
                continue
            selected[row["ligand_id"]] = {
                "label": row["label"],
                "score": float(row["docking_score"]),
            }
    if not selected:
        raise ValueError(f"no rank-1 successful rows found in {path}")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-table-a", type=Path, required=True)
    parser.add_argument("--receptor-a", required=True)
    parser.add_argument("--score-table-b", type=Path, required=True)
    parser.add_argument("--receptor-b", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    first = read_rank1(args.score_table_a)
    second = read_rank1(args.score_table_b)
    ligand_ids = sorted(set(first) & set(second))
    if len(ligand_ids) < 2:
        raise ValueError("fewer than two shared ligand IDs are available")
    if any(first[i]["label"] != second[i]["label"] for i in ligand_ids):
        raise ValueError("active/decoy labels are inconsistent between score tables")

    first_scores = [float(first[i]["score"]) for i in ligand_ids]
    second_scores = [float(second[i]["score"]) for i in ligand_ids]
    delta = [second_scores[i] - first_scores[i] for i in range(len(ligand_ids))]

    ensemble_data = {
        ligand_id: {
            "label": first[ligand_id]["label"],
            "min_score": min(float(first[ligand_id]["score"]), float(second[ligand_id]["score"])),
            "mean_score": (float(first[ligand_id]["score"]) + float(second[ligand_id]["score"])) / 2,
        }
        for ligand_id in ligand_ids
    }
    top_a = set(sorted(ligand_ids, key=lambda i: float(first[i]["score"]))[:10])
    top_b = set(sorted(ligand_ids, key=lambda i: float(second[i]["score"]))[:10])
    summary = {
        "receptor_a": args.receptor_a,
        "receptor_b": args.receptor_b,
        "shared_ligand_count": len(ligand_ids),
        "score_correlation": {
            "spearman": float(spearmanr(first_scores, second_scores).statistic),
            "pearson": float(pearsonr(first_scores, second_scores).statistic),
        },
        "score_delta_b_minus_a": {
            "mean_all": sum(delta) / len(delta),
            "mean_active": sum(delta[i] for i, ligand_id in enumerate(ligand_ids) if first[ligand_id]["label"] == "active") / sum(first[i]["label"] == "active" for i in ligand_ids),
            "mean_decoy": sum(delta[i] for i, ligand_id in enumerate(ligand_ids) if first[ligand_id]["label"] == "decoy") / sum(first[i]["label"] == "decoy" for i in ligand_ids),
        },
        "top10": {
            "overlap_count": len(top_a & top_b),
            "receptor_a_active_count": sum(first[i]["label"] == "active" for i in top_a),
            "receptor_b_active_count": sum(second[i]["label"] == "active" for i in top_b),
            "receptor_a_ids": sorted(top_a),
            "receptor_b_ids": sorted(top_b),
        },
        "metrics": {
            args.receptor_a: ranked_metrics_with_ids(first),
            args.receptor_b: ranked_metrics_with_ids(second),
            "ensemble_min_score": ranked_metrics_with_ids(ensemble_data, "min_score"),
            "ensemble_mean_score": ranked_metrics_with_ids(ensemble_data, "mean_score"),
        },
        "score_direction": "lower Vina docking_score is better; ranking_score is -score",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
