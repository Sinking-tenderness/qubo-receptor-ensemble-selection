from scripts.solve_qubo_receptor_subset import build_qubo, objective


def test_size_penalty_prefers_target_size() -> None:
    qubo = {
        "target_size": 2,
        "weights": {"redundancy": 0.0, "count": 0.0, "size": 10.0},
        "utilities_train_roc_auc": {"r1": 0.5, "r2": 0.5},
        "redundancy_train_spearman_clipped": {"r1__r2": 0.0},
    }

    assert objective(("r1", "r2"), qubo) < objective(("r1",), qubo)


def test_minmax_utility_is_scaled_to_zero_one() -> None:
    rows = [
        {"ligand_id": "a", "label": "active", "r1": "-10", "r2": "-5"},
        {"ligand_id": "d", "label": "decoy", "r1": "-5", "r2": "-10"},
    ]

    qubo = build_qubo(rows, ["r1", "r2"], 1, 0.0, 0.0, 1.0, "roc_auc", "minmax")

    assert set(qubo["utilities_train_roc_auc"].values()) == {0.0, 1.0}
