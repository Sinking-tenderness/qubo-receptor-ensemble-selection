from scripts.compare_receptor_screening import ranked_metrics_with_ids


def test_vina_scores_are_ranked_from_most_negative_to_least_negative() -> None:
    data = {
        "active_best": {"label": "active", "score": -12.0},
        "active_second": {"label": "active", "score": -10.0},
        "decoy": {"label": "decoy", "score": -11.0},
        "decoy_worse": {"label": "decoy", "score": -5.0},
    }

    summary = ranked_metrics_with_ids(data)

    assert summary["top10_ligand_ids"][:3] == [
        "active_best",
        "decoy",
        "active_second",
    ]


def test_equal_scores_use_ligand_id_as_a_deterministic_tie_breaker() -> None:
    data = {
        "Z_ligand": {"label": "active", "score": -5.0},
        "A_ligand": {"label": "decoy", "score": -5.0},
    }
    assert ranked_metrics_with_ids(data)["top10_ligand_ids"] == [
        "A_ligand",
        "Z_ligand",
    ]
