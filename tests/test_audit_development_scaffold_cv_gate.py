import pytest

from scripts.audit_development_scaffold_cv_gate import (
    independent_metrics,
    percentile,
)


def test_independent_metrics_rebuild_rank_based_screening_metrics():
    records = [
        {"label": "active", "score": -4.0},
        {"label": "decoy", "score": -3.0},
        {"label": "active", "score": -2.0},
        {"label": "decoy", "score": -1.0},
    ]
    metrics = independent_metrics(records)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["pr_auc_average_precision"] == pytest.approx(5.0 / 6.0)
    assert metrics["pr_auc_sklearn"] == pytest.approx(5.0 / 6.0)
    assert 0.0 < metrics["bedroc_alpha_20"] < 1.0


def test_percentile_uses_linear_interpolation():
    assert percentile([0.0, 10.0], 0.25) == pytest.approx(2.5)
