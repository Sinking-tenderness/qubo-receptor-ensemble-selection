from scripts.evaluate_virtual_screening import (
    average_precision,
    bedroc,
    enrichment_factor,
    roc_auc_pairwise,
)


def toy_ranked() -> list[dict[str, object]]:
    labels = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    scores = [11.0, 10.5, 10.0, 9.8, 9.4, 9.1, 8.7, 8.3, 8.0, 7.5]
    return [
        {
            "ligand_id": f"L{index}",
            "binary_label": label,
            "ranking_score": score,
            "rank": index,
        }
        for index, (label, score) in enumerate(zip(labels, scores), start=1)
    ]


def test_toy_roc_auc_and_enrichment() -> None:
    ranked = toy_ranked()
    labels = [int(row["binary_label"]) for row in ranked]
    scores = [float(row["ranking_score"]) for row in ranked]

    assert roc_auc_pairwise(labels, scores) == 0.625
    assert enrichment_factor(ranked, 0.3)["ef"] == 1.6666666666666665


def test_toy_average_precision_and_bedroc_bounds() -> None:
    ranked = toy_ranked()

    assert round(average_precision(ranked), 6) == 0.652778
    assert 0.0 <= bedroc(ranked, 20.0) <= 1.0
