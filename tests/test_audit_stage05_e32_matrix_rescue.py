import pytest

from scripts.audit_stage05_e32_matrix_rescue import evaluate_rescue


def test_rescue_passes_when_e32_converges_and_robust_aggregates_agree() -> None:
    result = evaluate_rescue(
        [-9.327, -9.393, -7.096],
        [-9.196, -9.394, -9.294],
        threshold=0.5,
    )

    assert result["e32_seed_range_kcal_per_mol"] == pytest.approx(0.198)
    assert result["absolute_e16_e32_median_delta_kcal_per_mol"] == pytest.approx(0.033)
    assert result["absolute_e16_e32_minimum_delta_kcal_per_mol"] == pytest.approx(0.001)
    assert result["rescue_passed"] is True


def test_rescue_fails_when_e32_remains_seed_unstable() -> None:
    result = evaluate_rescue(
        [-10.0, -12.0, -12.1],
        [-10.1, -12.1, -12.2],
        threshold=0.5,
    )

    assert result["checks"]["e32_seed_range_within_threshold"] is False
    assert result["rescue_passed"] is False


def test_rescue_requires_paired_multi_seed_scores() -> None:
    with pytest.raises(ValueError, match="equal multi-seed lengths"):
        evaluate_rescue([-9.0], [-9.1], threshold=0.5)
