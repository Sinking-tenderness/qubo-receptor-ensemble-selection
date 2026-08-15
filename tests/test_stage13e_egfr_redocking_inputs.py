import numpy as np

from scripts.prepare_stage13e_egfr_redocking_inputs import (
    citation_intent_matches,
    derive_common_box,
)


def test_common_box_rounding_preserves_required_margin() -> None:
    first = np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
    second = np.array([[-2.0, 4.0, 3.0], [8.0, 13.0, 15.0]])

    box = derive_common_box([first, second], 4.0, 2.0, 22.0, 2)

    assert box["center"] == {"x": 4.0, "y": 7.0, "z": 8.5}
    assert box["size"] == {"x": 22.0, "y": 22.0, "z": 22.0}
    assert box["minimum_observed_margin_angstrom"] >= 4.0


def test_citation_intent_diagnostic_is_case_insensitive() -> None:
    matches = citation_intent_matches(
        "A Covalent and irreversible EGFR inhibitor",
        [r"\bcovalent\b", r"\birreversible\b", r"mutant"],
    )

    assert matches == [r"\bcovalent\b", r"\birreversible\b"]
