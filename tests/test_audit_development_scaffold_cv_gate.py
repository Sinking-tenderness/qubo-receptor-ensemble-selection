import pytest

from scripts.audit_development_scaffold_cv_gate import (
    audit_consensus_constraints,
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


def test_audit_consensus_constraints_rebuilds_frequency_rule():
    row = {
        "method": "consensus_qubo",
        "consensus_required_receptors": '["R1"]',
        "subset": "R1+R2",
        "target_size": "2",
        "inner_subsets": '[["R1", "R2"], ["R1", "R3"], ["R1", "R4"]]',
        "consensus_reference_inner_subsets": (
            '[["R1", "R2"], ["R1", "R3"], ["R1", "R4"]]'
        ),
        "consensus_reference_config": '{"family": "coverage_qubo"}',
        "selected_config": '{"required_receptors": ["R1"]}',
    }
    assert audit_consensus_constraints([row], 2.0 / 3.0)


def test_audit_core_plus_one_rebuilds_fixed_core_and_residual_budget():
    row = {
        "method": "core_plus_one_qubo",
        "consensus_required_receptors": '["R1", "R2"]',
        "subset": "R1+R2+R4",
        "target_size": "3",
        "inner_subsets": '[["R1", "R2"], ["R1", "R2"], ["R1", "R3"]]',
        "consensus_reference_inner_subsets": (
            '[["R1", "R2"], ["R1", "R2"], ["R1", "R3"]]'
        ),
        "consensus_reference_config": '{"family": "coverage_qubo"}',
        "selected_config": (
            '{"family": "core_plus_one_qubo", '
            '"required_receptors": ["R1", "R2"]}'
        ),
    }
    assert audit_consensus_constraints([row], 2.0 / 3.0, 2)
