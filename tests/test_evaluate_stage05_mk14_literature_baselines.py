import numpy as np
import pytest

from scripts.evaluate_stage05_mk14_literature_baselines import (
    consensus_ranking_scores,
    fit_supervised_method,
    paired_group_bootstrap_delta,
    select_hantz_top_k,
    signed_geometric_mean,
)


RECEPTORS = ["R1", "R2", "R3", "R4", "R5"]


def synthetic_matrix():
    rows = {}
    for index in range(20):
        active = index < 10
        row = {
            "ligand_id": f"L{index:02d}",
            "label": "active" if active else "decoy",
        }
        for receptor_index, receptor in enumerate(RECEPTORS):
            signal = 2.0 if receptor_index < 3 and active else 0.0
            row[receptor] = -5.0 - signal - receptor_index * 0.1 - index * 0.001
        rows[row["ligand_id"]] = row
    return rows


def model_config():
    return {
        "model_seed": 17,
        "budget_matched_feature_count": 3,
        "logistic_regression": {
            "parameters": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 2000,
            }
        },
        "gradient_boosting": {
            "parameters": {
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 2,
            }
        },
        "random_forest": {
            "parameters": {
                "n_estimators": 20,
                "max_features": "sqrt",
                "class_weight": "balanced",
            }
        },
    }


def test_signed_geometric_mean_preserves_negative_docking_direction():
    assert signed_geometric_mean([-8.0, -2.0]) == pytest.approx(-4.0)
    assert signed_geometric_mean([8.0, 2.0]) == pytest.approx(4.0)
    assert signed_geometric_mean([0.0, -2.0]) == 0.0
    assert signed_geometric_mean([-1.0, -2.0, 3.0]) == pytest.approx(6 ** (1 / 3))
    with pytest.raises(ValueError):
        signed_geometric_mean([-1.0, 1.0])


def test_hantz_selection_uses_train_only_singleton_auc():
    matrix = synthetic_matrix()
    subset, auc_values = select_hantz_top_k(
        matrix, sorted(matrix), RECEPTORS, 3
    )

    assert subset == ("R1", "R2", "R3")
    assert all(auc_values[receptor] == pytest.approx(1.0) for receptor in subset)


def test_consensus_scores_rank_more_negative_values_higher():
    matrix = {
        "A": {"R1": -10.0, "R2": -8.0},
        "B": {"R1": -7.0, "R2": -6.0},
    }
    for strategy in ("min", "mean", "geometric"):
        scores = consensus_ranking_scores(
            matrix, ["A", "B"], ("R1", "R2"), strategy
        )
        assert scores["A"] > scores["B"]


def test_rfe_method_returns_exact_budget_and_finite_probabilities():
    matrix = synthetic_matrix()
    ids = sorted(matrix)
    model, subset = fit_supervised_method(
        "ricci_gbt_rfe3", matrix, ids, RECEPTORS, model_config()
    )

    assert len(subset) == 3
    assert set(subset).issubset(RECEPTORS)
    probabilities = model.predict_proba(
        np.asarray([[0.5] * 3], dtype=np.float64)
    )
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()


def test_paired_group_bootstrap_reports_explicit_direction():
    left = {
        "A1": {"label": "active", "score": -4.0},
        "A2": {"label": "active", "score": -3.0},
        "D1": {"label": "decoy", "score": -2.0},
        "D2": {"label": "decoy", "score": -1.0},
    }
    right = {
        "A1": {"label": "active", "score": -2.0},
        "A2": {"label": "active", "score": -1.0},
        "D1": {"label": "decoy", "score": -4.0},
        "D2": {"label": "decoy", "score": -3.0},
    }
    groups = {
        "A1": "active_group_1",
        "A2": "active_group_2",
        "D1": "decoy_group_1",
        "D2": "decoy_group_2",
    }

    result = paired_group_bootstrap_delta(left, right, groups, 20, 17)

    assert result["valid_replicates"] == 20
    assert result["direction"] == "left BEDROC20 minus right BEDROC20"
    assert result["mean"] > 0.0
