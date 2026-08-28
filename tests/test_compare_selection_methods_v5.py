from __future__ import annotations

from scripts.compare_selection_methods_v5 import (
    select_greedy,
    select_linear,
    select_methods,
    select_single,
    resolve_fixed_k_by_fold,
)


def sample_rows() -> list[dict[str, object]]:
    return [
        {"ligand_id": "a1", "label": "active", "A": -10.0, "B": -9.0, "C": -8.0},
        {"ligand_id": "a2", "label": "active", "A": -9.0, "B": -8.0, "C": -7.0},
        {"ligand_id": "d1", "label": "decoy", "A": -7.0, "B": -8.5, "C": -9.5},
        {"ligand_id": "d2", "label": "decoy", "A": -6.0, "B": -7.5, "C": -9.0},
    ]


def test_linear_uses_singleton_train_bedroc_order() -> None:
    rows = sample_rows()

    selected = select_linear(rows, ["A", "B", "C"], 2)

    assert selected == ("A", "B")


def test_greedy_returns_unique_subset_of_requested_size() -> None:
    rows = sample_rows()

    selected = select_greedy(rows, ["A", "B", "C"], 2)

    assert len(selected) == 2
    assert len(set(selected)) == 2


def test_matched_k_methods_share_k_but_single_stays_single() -> None:
    rows = sample_rows()

    selected = select_methods(rows, ["A", "B", "C"], 2)

    assert len(selected["qubo"]) == 2
    assert len(selected["linear"]) == 2
    assert len(selected["greedy"]) == 2
    assert len(selected["single"]) == 1
    assert selected["single"] == select_single(rows, ["A", "B", "C"])


def test_fixed_k_is_applied_to_every_outer_fold() -> None:
    assert resolve_fixed_k_by_fold([1, 2, 5], 4) == {1: 4, 2: 4, 5: 4}
