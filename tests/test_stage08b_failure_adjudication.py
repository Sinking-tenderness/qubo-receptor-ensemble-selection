from pathlib import Path

from scripts.adjudicate_stage08b_mk14_replacement_failure import read_json


def test_stage08b_failure_result_has_one_admitted_and_one_failed() -> None:
    path = Path("data/stage08b_mk14_replacement_redocking_failure_adjudication.json")
    if not path.exists():
        return
    result = read_json(path)
    assert result["status"] == "stage08b_replacement_gate_failed_one_receptor"
    assert result["admitted_receptor_ids"] == ["MK14_3ITZ_aligned"]
    assert result["failed_receptor_ids"] == ["MK14_2BAK_aligned"]
    assert result["unresolved_warning_event_count"] == 0
    assert result["pose_integrity_failure_count"] == 0
