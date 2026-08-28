from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts.compare_selection_methods_v5 import (
    resolve_fixed_k_by_fold,
    evaluate_target,
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


def test_fixed_k_comparison_accepts_a_full_matrix_without_outer_folds() -> None:
    rows = [
        {"ligand_id": "a1", "label": "active", "target_id": "PPARA", "A": -10.0, "B": -9.0, "C": -8.0},
        {"ligand_id": "a2", "label": "active", "target_id": "PPARA", "A": -9.0, "B": -8.0, "C": -7.0},
        {"ligand_id": "d1", "label": "decoy", "target_id": "PPARA", "A": -5.0, "B": -4.0, "C": -3.0},
        {"ligand_id": "d2", "label": "decoy", "target_id": "PPARA", "A": -4.0, "B": -3.0, "C": -2.0},
    ]
    with tempfile.TemporaryDirectory() as directory:
        problem = Path(directory) / "problem.json"
        problem.write_text(
            json.dumps({"rows": rows, "problem": {"receptor_ids": ["A", "B", "C"]}}),
            encoding="utf-8",
        )

        records, metadata = evaluate_target(problem, fixed_k=2)

    assert metadata["comparison_scope"] == "full_data"
    assert {record["method"] for record in records} == {"qubo", "linear", "greedy", "single"}
    assert {record["selected_k"] for record in records if record["method"] != "single"} == {2}
