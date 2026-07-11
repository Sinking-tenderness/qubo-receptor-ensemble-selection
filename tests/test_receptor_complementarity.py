from scripts.analyze_receptor_complementarity import top_ids


def test_top_ids_use_vina_ascending_score_and_stable_tie_break() -> None:
    data = {
        "b": {"label": "active", "score": -10.0},
        "a": {"label": "decoy", "score": -10.0},
        "c": {"label": "active", "score": -9.0},
    }

    assert top_ids(data, n=3) == ["a", "b", "c"]
