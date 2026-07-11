from scripts.select_receptor_baselines import metrics_for_subset


def test_single_receptor_metrics_use_that_column() -> None:
    rows = [
        {"ligand_id": "a", "label": "active", "r1": "-10", "r2": "-1"},
        {"ligand_id": "d", "label": "decoy", "r1": "-5", "r2": "-2"},
    ]

    summary = metrics_for_subset(rows, ("r1",), "min_score")

    assert summary["roc_auc"] == 1.0
