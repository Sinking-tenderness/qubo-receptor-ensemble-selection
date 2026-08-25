import pytest

from qubo_receptor_ensemble.adaptive_cardinality import (
    AdaptiveCardinalityError,
    MarginalObservation,
    TransitionEvidence,
    _score_records,
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
        [transition],
        bootstrap_iterations=200,
        random_seed=17,
    )

    assert decision.selected_k == 2
    assert decision.need_multi_conformation is True
    assert decision.transitions[0]["passed"] is True
    assert decision.transitions[0]["bootstrap_lcb"] > 0
    assert decision.transitions[0]["mean_rescue_contrast"] > 0
    assert decision.transitions[0]["lower_quantile"] == 0.05
    assert decision.uses_outer_labels is False


def test_failed_one_to_two_does_not_block_three() -> None:
    failed = TransitionEvidence(
        from_k=1,
        to_k=2,
        observations=tuple(
            _observation("S" + str(index), -0.05, 0.01, 0.01)
            for index in range(4)
        ),
    )
    direct_to_three = TransitionEvidence(
        from_k=2,
        to_k=3,
        observations=tuple(
            _observation("T" + str(index), 0.20, 0.05, 0.05)
            for index in range(4)
        ),
    )

    decision = select_adaptive_k(
        [failed, direct_to_three], bootstrap_iterations=200, random_seed=17
    )

    assert decision.selected_k == 3
    assert decision.need_multi_conformation is True
    assert [item["transition"] for item in decision.transitions] == ["1->2", "2->3"]
    assert decision.transitions[0]["marginal_state"] == "harmful"
    assert decision.transitions[1]["marginal_state"] == "supported"
    assert decision.transitions[1]["cumulative_risk_adjusted_gain"] > 0


def test_positive_oof_mean_can_select_candidate_with_negative_lcb() -> None:
    transition = TransitionEvidence(
        from_k=1,
        to_k=5,
        observations=tuple(_observation("S" + str(index), 0.0, 0.02, 0.02) for index in range(4)),
        bootstrap_samples=(-0.02, -0.01, 0.01, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09),
    )

    decision = select_adaptive_k(
        [transition],
        bootstrap_iterations=100,
        required_probability=0.75,
        random_seed=17,
    )

    assert decision.selected_k == 5
    assert decision.transitions[0]["bootstrap_lcb"] < 0
    assert decision.transitions[0]["mean_paired_bedroc_gain"] > 0
    assert decision.transitions[0]["risk_adjusted_gain"] > 0


def test_estimator_retains_all_adjacent_candidate_transitions() -> None:
    rows = []
    for number in range(1, 5):
        rows.extend(
            (
                {
                    "ligand_id": f"A{number}",
                    "label": "active",
                    "scaffold_smiles": f"A{number}",
                    "R1": -float(10 - number),
                    "R2": -float(11 - number),
                    "R3": -float(12 - number),
                },
                {
                    "ligand_id": f"D{number}",
                    "label": "decoy",
                    "scaffold_smiles": f"D{number}",
                    "R1": -float(number),
                    "R2": -float(number + 1),
                    "R3": -float(number + 2),
                },
            )
        )

    def solve_subset(train_rows: list[dict[str, object]], k: int) -> tuple[str, ...]:
        del train_rows
        return tuple(("R1", "R2", "R3")[:k])

    decision = estimate_adaptive_cardinality(
        rows,
        ["R1", "R2", "R3"],
        solve_subset=solve_subset,
        candidate_ks=(1, 2, 3),
        inner_fold_count=2,
        bootstrap_iterations=100,
        random_seed=17,
    )

    assert [item["transition"] for item in decision.transitions] == [
        "1->2",
        "2->3",
    ]


def test_estimator_accepts_candidate_k_above_three() -> None:
    rows = []
    receptor_ids = [f"R{number}" for number in range(1, 6)]
    for number in range(1, 5):
        rows.extend(
            (
                {
                    "ligand_id": f"A{number}",
                    "label": "active",
                    "scaffold_smiles": f"A{number}",
                    **{receptor_id: -float(number + index) for index, receptor_id in enumerate(receptor_ids)},
                },
                {
                    "ligand_id": f"D{number}",
                    "label": "decoy",
                    "scaffold_smiles": f"D{number}",
                    **{receptor_id: -float(10 + number + index) for index, receptor_id in enumerate(receptor_ids)},
                },
            )
        )

    calls: list[int] = []

    def solve_subset(train_rows: list[dict[str, object]], k: int) -> tuple[str, ...]:
        del train_rows
        calls.append(k)
        return tuple(receptor_ids[:k])

    decision = estimate_adaptive_cardinality(
        rows,
        receptor_ids,
        solve_subset=solve_subset,
        candidate_ks=None,
        inner_fold_count=2,
        bootstrap_iterations=100,
        random_seed=17,
    )

    assert set(calls) == {1, 2, 3, 4, 5}
    assert [item["transition"] for item in decision.transitions] == [
        "1->2",
        "2->3",
        "3->4",
        "4->5",
    ]

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
        [transition],
        bootstrap_iterations=200,
        random_seed=17,
        require_rescue_contrast=True,
    )

    assert decision.selected_k == 1
    assert decision.transitions[0]["bootstrap_lcb"] > 0
    assert decision.transitions[0]["mean_rescue_contrast"] < 0
    assert decision.transitions[0]["passed"] is False


def test_adjacent_gains_accumulate_for_candidate_selection() -> None:
    first = TransitionEvidence(
        from_k=1,
        to_k=2,
        observations=tuple(
            _observation(f"S{index}", 0.08, 0.04, 0.02) for index in range(4)
        ),
        bootstrap_samples=(0.08,) * 10,
    )
    second = TransitionEvidence(
        from_k=2,
        to_k=3,
        observations=tuple(
            _observation(f"S{index}", 0.06, 0.04, 0.02) for index in range(4)
        ),
        bootstrap_samples=(0.06,) * 10,
    )

    decision = select_adaptive_k(
        [first, second], bootstrap_iterations=10, required_probability=0.8
    )

    assert decision.selected_k == 3
    assert decision.transitions[0]["risk_adjusted_gain"] == pytest.approx(0.08)
    assert decision.transitions[1]["risk_adjusted_gain"] == pytest.approx(0.06)
    assert decision.transitions[1]["cumulative_risk_adjusted_gain"] == pytest.approx(0.14)


def test_uncertain_marginal_does_not_block_later_candidate() -> None:
    first = TransitionEvidence(
        from_k=1,
        to_k=2,
        observations=tuple(
            _observation(f"S{index}", 0.0, 0.04, 0.02) for index in range(4)
        ),
        bootstrap_samples=(-0.02, 0.02) * 5,
    )
    second = TransitionEvidence(
        from_k=2,
        to_k=3,
        observations=tuple(
            _observation(f"S{index}", 0.08, 0.04, 0.02) for index in range(4)
        ),
        bootstrap_samples=(0.08,) * 10,
    )

    decision = select_adaptive_k(
        [first, second], bootstrap_iterations=10, required_probability=0.8
    )

    assert decision.selected_k == 3
    assert decision.transitions[0]["marginal_state"] == "uncertain"
    assert decision.transitions[0]["passed"] is False
    assert decision.transitions[1]["candidate_passed"] is True


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


def test_score_records_uses_configured_aggregation() -> None:
    rows = [
        {
            "ligand_id": "L1",
            "label": "active",
            "R1": -10.0,
            "R2": -2.0,
        }
    ]

    assert _score_records(rows, ("R1", "R2"), "min_score")["L1"]["score"] == -10.0
    assert _score_records(rows, ("R1", "R2"), "mean_score")["L1"]["score"] == -6.0


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


@pytest.mark.parametrize(
    ("metric", "gain_key"),
    (
        ("roc_auc", "mean_paired_roc_auc_gain"),
        ("bedroc", "mean_paired_bedroc_gain"),
        ("ef5", "mean_paired_ef5_gain"),
    ),
)
def test_estimator_uses_configured_utility_metric_and_reports_progress(
    metric: str, gain_key: str
) -> None:
    rows = []
    for number in range(1, 5):
        rows.extend(
            (
                {
                    "ligand_id": f"A{number}",
                    "label": "active",
                    "scaffold_smiles": f"A{number}",
                    "R1": -float(7 - number),
                    "R2": -10.0 - number,
                },
                {
                    "ligand_id": f"D{number}",
                    "label": "decoy",
                    "scaffold_smiles": f"D{number}",
                    "R1": -10.0 + number,
                    "R2": -1.0,
                },
            )
        )

    events: list[tuple[str, dict[str, object]]] = []

    def solve_subset(train_rows: list[dict[str, object]], k: int) -> tuple[str, ...]:
        del train_rows
        return ("R1",) if k == 1 else ("R1", "R2")

    decision = estimate_adaptive_cardinality(
        rows,
        ["R1", "R2"],
        problem_config={"utility_metric": metric, "bedroc_alpha": 20.0},
        solve_subset=solve_subset,
        candidate_ks=(1, 2),
        inner_fold_count=2,
        bootstrap_iterations=100,
        random_seed=13,
        progress=lambda event, payload: events.append((event, dict(payload))),
        aggregation="mean_score",
    )

    transition = decision.transitions[0]
    expected_mean_gain = {
        "roc_auc": 0.96875,
        "bedroc": 0.9995339465195752,
        "ef5": 2.0,
    }[metric]
    assert decision.metric == metric
    assert decision.aggregation == "mean_score"
    assert transition["metric"] == metric
    assert gain_key in transition
    assert transition[gain_key] == pytest.approx(expected_mean_gain)
    assert not (
        metric != "bedroc" and "mean_paired_bedroc_gain" in transition
    )
    assert events[0] == (
        "adaptive_started",
        {
            "metric": metric,
            "aggregation": "mean_score",
            "candidates": [1, 2],
        },
    )
    assert any(event == "inner_fold_started" for event, _ in events)
    assert any(event == "candidate_completed" for event, _ in events)
    assert events[-1][0] == "adaptive_completed"
