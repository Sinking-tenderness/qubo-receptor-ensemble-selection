import pytest

from scripts.run_openmm_production import (
    chunk_filename,
    initialize_progress,
    production_steps,
    validate_progress,
    validate_schedule,
)


def test_production_steps_converts_ns_to_two_fs_steps():
    assert production_steps(2.0, 2.0) == 1_000_000


def test_validate_schedule_accepts_documented_intervals():
    validate_schedule(1_000_000, 5_000, 10_000, 50_000)


def test_validate_schedule_rejects_misaligned_frame_interval():
    with pytest.raises(ValueError, match="frame interval"):
        validate_schedule(1_000_000, 6_000, 10_000, 50_000)


def test_chunk_filename_is_stable():
    assert chunk_filename("trajectory", 0.0, 100.0) == "trajectory_000000.000_000100.000ps.dcd"


def test_progress_starts_empty_and_validates():
    progress = initialize_progress("experiment")
    validate_progress(progress, 1_000_000)
    assert progress["completed_steps"] == 0
    assert progress["completed_trajectory_chunks"] == []
