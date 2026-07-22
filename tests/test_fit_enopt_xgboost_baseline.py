import numpy as np

from scripts.fit_enopt_xgboost_baseline import (
    feature_subsets,
    fit_primary_model,
    minmax_bounds,
    normalized_features,
    parameter_grid,
    predict_matrix,
    select_trial,
)


RECEPTORS = ["R1", "R2", "R3", "R4", "R5"]


def synthetic_matrix():
    rows = {}
    for index in range(12):
        label = "active" if index < 6 else "decoy"
        rows[f"L{index:02d}"] = {
            "ligand_id": f"L{index:02d}",
            "label": label,
            **{
                receptor: float(index + receptor_index / 10)
                for receptor_index, receptor in enumerate(RECEPTORS)
            },
        }
    return rows


def test_feature_subsets_match_five_choose_three_budget():
    assert feature_subsets("xgboost_all5", RECEPTORS, 3) == [tuple(RECEPTORS)]
    budget = feature_subsets("xgboost_budget3", RECEPTORS, 3)
    assert len(budget) == 10
    assert len(set(budget)) == 10
    assert all(len(subset) == 3 for subset in budget)


def test_parameter_grid_is_deterministic_cartesian_product():
    config = {
        "fixed_parameters": {"booster": "gbtree"},
        "hyperparameter_grid": {
            "max_depth": [2, 3],
            "n_estimators": [10, 20],
        },
    }
    first = parameter_grid(config)
    second = parameter_grid(config)
    assert first == second
    assert len(first) == 4
    assert first[0] == {
        "booster": "gbtree",
        "max_depth": 2,
        "n_estimators": 10,
    }


def test_train_bounds_are_applied_without_validation_clipping():
    matrix = synthetic_matrix()
    bounds = minmax_bounds(matrix, ["L02", "L03", "L04"], ["R1"])
    values = normalized_features(matrix, ["L00", "L11"], ("R1",), bounds)

    assert bounds == {"R1": {"minimum": 2.0, "maximum": 4.0}}
    assert values[:, 0].tolist() == [-1.0, 4.5]


def test_xgboost_fit_and_prediction_are_deterministic():
    matrix = synthetic_matrix()
    train_ids = [f"L{index:02d}" for index in range(10)]
    validation_ids = ["L10", "L11"]
    params = {
        "booster": "gbtree",
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "max_depth": 2,
        "min_child_weight": 1.0,
        "learning_rate": 0.1,
        "n_estimators": 10,
    }
    first, _ = fit_primary_model(
        matrix, train_ids, ("R1", "R2", "R3"), params, 17
    )
    second, _ = fit_primary_model(
        matrix, train_ids, ("R1", "R2", "R3"), params, 17
    )
    first_prediction = predict_matrix(
        first,
        matrix,
        train_ids,
        validation_ids,
        ("R1", "R2", "R3"),
    )
    second_prediction = predict_matrix(
        second,
        matrix,
        train_ids,
        validation_ids,
        ("R1", "R2", "R3"),
    )

    assert first_prediction == second_prediction
    assert all(np.isfinite(value) for value in first_prediction.values())


def test_trial_selection_uses_bedroc_before_complexity():
    common = {
        "method": "xgboost_all5",
        "subset": RECEPTORS,
        "parameters": {
            "max_depth": 2,
            "n_estimators": 150,
            "learning_rate": 0.03,
            "min_child_weight": 1.0,
        },
        "bedroc_population_std": 0.01,
        "fold_count": 3,
    }
    low = {
        **common,
        "candidate_id": "LOW",
        "mean_validation_metrics": {
            "bedroc_alpha_20": 0.8,
            "pr_auc_average_precision": 0.9,
            "roc_auc": 0.9,
        },
    }
    high = {
        **common,
        "candidate_id": "HIGH",
        "mean_validation_metrics": {
            "bedroc_alpha_20": 0.81,
            "pr_auc_average_precision": 0.1,
            "roc_auc": 0.1,
        },
    }

    assert select_trial([low, high])["candidate_id"] == "HIGH"
