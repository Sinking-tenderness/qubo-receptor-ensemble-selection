from scripts.solve_qubo_receptor_subset import objective


def test_size_penalty_prefers_target_size() -> None:
    qubo = {
        "target_size": 2,
        "weights": {"redundancy": 0.0, "count": 0.0, "size": 10.0},
        "utilities_train_roc_auc": {"r1": 0.5, "r2": 0.5},
        "redundancy_train_spearman_clipped": {"r1__r2": 0.0},
    }

    assert objective(("r1", "r2"), qubo) < objective(("r1",), qubo)
