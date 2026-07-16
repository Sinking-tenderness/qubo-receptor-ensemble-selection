import pytest

from scripts.analyze_receptor_selection_stability import (
    inclusion_rows,
    jaccard,
    pairwise_jaccard_summary,
)


def test_jaccard_and_inclusion_frequency_are_deterministic():
    subsets = [["R1", "R2"], ["R1", "R3"]]

    assert jaccard(*subsets) == pytest.approx(1.0 / 3.0)
    assert pairwise_jaccard_summary(subsets)["mean"] == pytest.approx(
        1.0 / 3.0
    )
    rows = inclusion_rows(subsets, ["R1", "R2", "R3"])
    assert rows[0] == {
        "receptor_id": "R1",
        "selection_count": 2,
        "selection_frequency": 1.0,
    }
