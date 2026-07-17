import csv
from pathlib import Path

from scripts.audit_locked_test_release import audit_ranking, compare_metrics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_audit_ranking_reproduces_metrics(tmp_path: Path) -> None:
    path = tmp_path / "ranking.csv"
    write_csv(
        path,
        [
            {
                "rank": 1,
                "ligand_id": "A1",
                "label": "active",
                "docking_score": -9.0,
                "ranking_score": 9.0,
                "matrix_role": "role",
            },
            {
                "rank": 2,
                "ligand_id": "A2",
                "label": "active",
                "docking_score": -8.0,
                "ranking_score": 8.0,
                "matrix_role": "role",
            },
            {
                "rank": 3,
                "ligand_id": "D1",
                "label": "decoy",
                "docking_score": -6.0,
                "ranking_score": 6.0,
                "matrix_role": "role",
            },
            {
                "rank": 4,
                "ligand_id": "D2",
                "label": "decoy",
                "docking_score": -5.0,
                "ranking_score": 5.0,
                "matrix_role": "role",
            },
        ],
    )
    _, metrics = audit_ranking(
        path,
        {"A1": "active", "A2": "active", "D1": "decoy", "D2": "decoy"},
        "role",
    )
    assert metrics["roc_auc"] == 1.0
    assert compare_metrics(metrics, metrics) == 0.0
