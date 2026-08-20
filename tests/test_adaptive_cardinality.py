import pytest

from qubo_receptor_ensemble.adaptive_cardinality import (
    AdaptiveCardinalityError,
    MarginalObservation,
    TransitionEvidence,
    estimate_adaptive_cardinality,
    select_adaptive_k,
)


def _observation(
    scaffold: str,
    gain: float,
    active_top1: float,
    active_top5: float,
    decoy_top1: float = 0.0,
    decoy_top5: float = 0.0,
) -> MarginalObservation:
    return MarginalObservation(
        scaffold_id=scaffold,
        paired_bedroc_gain=gain,
        active_rescue_top1=active_top1,
        active_rescue_top5=active_top5,
        decoy_rescue_top1=decoy_top1,
        decoy_rescue_top5=decoy_top5,
    )


def test_positive_transition_enables_two_conformations() -> None:
    transition = TransitionEvidence(
        from_k=1,
        to_k=2,
        observations=tuple(
            _observation("S" + str(index), 0.08, 0.04, 0.02)
            for index in range(4)
        ),
    )

    decision = select_adaptive_k(
        [transition], bootstrap_iterations=200, random_seed=17
    )

    assert decision.selected_k == 2
    assert decision.need_multi_conformation is True
    assert decision.transitions[0]["passed"] is True
    assert decision.transitions[0]["bootstrap_lcb"] > 0
    assert decision.transitions[0]["mean_rescue_contrast"] > 0
    assert decision.uses_outer_labels is False


def test_failed_transition_stops_before_evaluating_three() -> None:
    failed = TransitionEvidence(
        from_k=1,
        to_k=2,
        observations=tuple(
            _observation("S" + str(index), -0.05, 0.01, 0.01)
            for index in range(4)
        ),
    )
    later = TransitionEvidence(
        from_k=2,
        to_k=3,
        observations=tuple(
            _observation("T" + str(index), 0.20, 0.05, 0.05)
            for index in range(4)
        ),
    )

    decision = select_adaptive_k(
        [failed, later], bootstrap_iterations=200, random_seed=17
    )

    assert decision.selected_k == 1
    assert decision.need_multi_conformation is False
    assert [item["transition"] for item in decision.transitions] == ["1->2"]


def test_positive_gain_with_negative_rescue_stops() -> None:
    transition = TransitionEvidence(
        from_k=1,
        to_k=2,
        observations=tuple(
            _observation("S" + str(index), 0.08, 0.01, 0.01, 0.03, 0.03)
            for index in range(4)
        ),
    )

    decision = select_adaptive_k(
        [transition], bootstrap_iterations=200, random_seed=17
    )

    assert decision.selected_k == 1
    assert decision.transitions[0]["bootstrap_lcb"] > 0
    assert decision.transitions[0]["mean_rescue_contrast"] < 0
    assert decision.transitions[0]["passed"] is False


def test_bootstrap_is_deterministic_and_requires_scaffold_groups() -> None:
    transition = TransitionEvidence(
        from_k=1,
        to_k=2,
        observations=tuple(
            _observation("S" + str(index), 0.08, 0.04, 0.02)
            for index in range(4)
        ),
    )

    first = select_adaptive_k(
        [transition], bootstrap_iterations=200, random_seed=99
    ).as_dict()
    second = select_adaptive_k(
        [transition], bootstrap_iterations=200, random_seed=99
    ).as_dict()

    assert first == second

    invalid = TransitionEvidence(
        from_k=1,
        to_k=2,
        observations=(
            _observation("", 0.1, 0.1, 0.1),
            _observation("S2", 0.1, 0.1, 0.1),
        ),
    )
    with pytest.raises(AdaptiveCardinalityError, match="scaffold_id"):
        select_adaptive_k([invalid])


def test_estimator_solves_sequential_candidates_from_inner_folds() -> None:
    rows = []
    for outer_fold, suffix in ((0, ("1", "2")), (1, ("3", "4"))):
        for number in suffix:
            rows.append(
                {
                    "ligand_id": "A" + number,
                    "label": "active",
                    "scaffold_smiles": "A" + number,
                    "outer_fold": outer_fold,
                    "R1": -float(7 - int(number)),
                    "R2": -10.0 - int(number),
                }
            )
            rows.append(
                {
                    "ligand_id": "D" + number,
                    "label": "decoy",
                    "scaffold_smiles": "D" + number,
                    "outer_fold": outer_fold,
                    "R1": -10.0 + int(number),
                    "R2": -1.0,
                }
            )

    calls: list[int] = []

    def solve_subset(train_rows: list[dict[str, object]], k: int) -> tuple[str, ...]:
        del train_rows
        calls.append(k)
        return ("R1",) if k == 1 else ("R1", "R2")

    decision = estimate_adaptive_cardinality(
        rows,
        ["R1", "R2"],
        solve_subset=solve_subset,
        candidate_ks=(1, 2),
        inner_fold_count=2,
        bootstrap_iterations=100,
        random_seed=13,
    )

    assert decision.selected_k == 2
    assert decision.need_multi_conformation is True
    assert calls.count(1) == calls.count(2)
    assert calls.count(1) > 0
