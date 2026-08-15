from scripts.adjudicate_stage08_mk14_expanded16_redocking_failure import (
    summarize_gate,
)


def test_failure_gate_keeps_two_of_three_rule() -> None:
    rows = [
        {"receptor_id": "R1", "rmsd_angstrom": value}
        for value in (1.5, 2.2, 2.3)
    ]
    result = summarize_gate(rows, ["R1"], 2.0, 2)
    assert result[0]["successful_seed_count"] == 1
    assert result[0]["gate_pass"] is False


def test_failure_gate_admits_one_outlier_when_two_seeds_pass() -> None:
    rows = [
        {"receptor_id": "R1", "rmsd_angstrom": value}
        for value in (1.0, 1.1, 13.0)
    ]
    result = summarize_gate(rows, ["R1"], 2.0, 2)
    assert result[0]["successful_seed_count"] == 2
    assert result[0]["median_rmsd_angstrom"] == 1.1
    assert result[0]["gate_pass"] is True
