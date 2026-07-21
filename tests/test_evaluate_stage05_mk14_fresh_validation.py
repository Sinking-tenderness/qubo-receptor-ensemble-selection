from scripts.evaluate_stage05_mk14_fresh_validation import (
    normalize_matrices,
    paired_group_bootstrap,
    quantile,
)


def test_normalization_uses_frozen_bounds_without_clipping():
    matrices = {
        matrix: {
            "L1": {"ligand_id": "L1", "label": "active", "R1": -12.0},
            "L2": {"ligand_id": "L2", "label": "decoy", "R1": -4.0},
        }
        for matrix in ("primary", "sensitivity", "seed0", "seed1", "seed2")
    }
    bounds = {
        matrix: {"R1": {"minimum": -10.0, "maximum": -5.0}}
        for matrix in matrices
    }

    normalized = normalize_matrices(matrices, bounds, ["R1"])

    assert normalized["primary"]["L1"]["R1"] == -0.4
    assert normalized["primary"]["L2"]["R1"] == 1.2


def test_group_bootstrap_is_deterministic_and_paired():
    q = {
        "A1": {"label": "active", "score": -3.0},
        "A2": {"label": "active", "score": -2.0},
        "D1": {"label": "decoy", "score": -1.0},
        "D2": {"label": "decoy", "score": 0.0},
    }
    linear = {
        "A1": {"label": "active", "score": -2.0},
        "A2": {"label": "active", "score": 0.0},
        "D1": {"label": "decoy", "score": -3.0},
        "D2": {"label": "decoy", "score": -1.0},
    }
    groups = {ligand_id: ligand_id for ligand_id in q}
    records = {
        "pair_synergy_qubo": q,
        "matched_linear_top_k": linear,
        "nested_exhaustive_final": linear,
    }

    first = paired_group_bootstrap(records, groups, 50, 91)
    second = paired_group_bootstrap(records, groups, 50, 91)

    assert first == second
    assert first["valid_replicates"] == 50
    assert first["deltas"]["matched_linear_top_k"] == first["deltas"][
        "nested_exhaustive_final"
    ]


def test_quantile_uses_linear_interpolation():
    assert quantile([0.0, 10.0], 0.25) == 2.5
