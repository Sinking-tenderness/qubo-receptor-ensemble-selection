from pathlib import Path

import pytest

from scripts.run_openmm_equilibration import (
    add_time_fields,
    initialize_progress,
    output_paths,
    validate_progress,
)


def test_initialize_progress_starts_at_nvt():
    progress = initialize_progress("experiment", {"phase": "minimized"})
    assert progress["phase"] == "NVT"
    assert progress["nvt_completed_steps"] == 0
    assert progress["records"] == [{"phase": "minimized"}]


def test_validate_progress_rejects_unknown_phase():
    with pytest.raises(ValueError, match="phase"):
        validate_progress({
            "phase": "production", "nvt_completed_steps": 0,
            "npt_completed_steps": 0, "records": [],
        })


def test_add_time_fields_offsets_npt_time():
    record = add_time_fields({"elapsed_ps": 10.0, "phase": "NPT"}, 100.0)
    assert record["total_elapsed_ps"] == 110.0


def test_output_paths_converts_strings():
    assert output_paths({"manifest": "results/test.json"}) == {
        "manifest": Path("results/test.json")
    }
