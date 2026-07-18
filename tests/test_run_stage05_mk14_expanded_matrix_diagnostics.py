import pytest

from scripts.run_stage05_mk14_expanded_matrix_diagnostics import (
    select_flagged_rows,
    summarize_case,
)


def aggregate_row(scores):
    return {
        "ligand_id": "L1",
        "receptor_id": "R1",
        "seed0_representative_score": str(scores[0]),
        "seed1_representative_score": str(scores[1]),
        "seed2_representative_score": str(scores[2]),
        "seed_score_range": str(max(scores) - min(scores)),
    }


def test_selector_includes_nonnegative_and_excessive_range() -> None:
    rows = [
        aggregate_row([-9.0, -9.1, -9.2]),
        {**aggregate_row([0.5, -9.0, -9.1]), "ligand_id": "L2"},
        {**aggregate_row([-9.0, -11.2, -9.1]), "ligand_id": "L3"},
    ]
    selected = select_flagged_rows(rows, ["seed0", "seed1", "seed2"], 0, 2.0)
    assert [row["ligand_id"] for row in selected] == ["L2", "L3"]


def test_case_summary_rescues_isolated_positive_source() -> None:
    case = {
        "case_id": "C1",
        "ligand_id": "L1",
        "receptor_id": "R1",
        "selection_role": "development_train",
        "actual_seeds": {"seed0": 101, "seed1": 102, "seed2": 103},
        "source_e16_scores": {"seed0": 0.5, "seed1": -9.0, "seed2": -9.1},
    }
    rows = [
        {"case_id": "C1", "seed": seed, "status": "ok", "docking_score": score}
        for seed, score in zip([101, 102, 103], [-9.05, -9.10, -9.00], strict=True)
    ]
    summary = summarize_case(case, rows, threshold=0.5)
    assert summary["rescue_passed"] is True
    assert summary["diagnostic_classification"] == (
        "isolated_e16_nonnegative_search_failure_rescued"
    )


def test_case_summary_fails_when_e32_remains_unstable() -> None:
    case = {
        "case_id": "C1",
        "ligand_id": "L1",
        "receptor_id": "R1",
        "selection_role": "development_train",
        "actual_seeds": {"seed0": 101, "seed1": 102, "seed2": 103},
        "source_e16_scores": {"seed0": -9.0, "seed1": -11.2, "seed2": -9.1},
    }
    rows = [
        {"case_id": "C1", "seed": seed, "status": "ok", "docking_score": score}
        for seed, score in zip([101, 102, 103], [-9.0, -11.1, -9.1], strict=True)
    ]
    summary = summarize_case(case, rows, threshold=0.5)
    assert summary["rescue_passed"] is False
    assert summary["diagnostic_classification"] == (
        "persistent_or_inconclusive_search_instability"
    )
