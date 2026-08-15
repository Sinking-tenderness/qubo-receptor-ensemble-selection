from pathlib import Path

from scripts.audit_stage18a_pparg_source import allocate_active_panels


def test_active_allocation_has_the_expected_roles(tmp_path: Path) -> None:
    path = tmp_path / "actives.ism"
    path.write_text(
        "CC a1\nCCC a2\nCCCC a3\nCCCCC a4\n", encoding="ascii"
    )
    rows, summary = allocate_active_panels(
        path,
        {
            "hash_seed": "unit-test",
            "source_active_count": 4,
            "selected_active_count": 4,
            "unallocated_source_surplus_count": 0,
            "train_active_count": 2,
            "fresh_validation_active_count": 1,
            "locked_test_active_count": 1,
            "maximum_scaffold_group_size": 1,
        },
    )

    assert len(rows) == 4
    assert summary["panel_counts"] == {
        "development_train": 2,
        "fresh_validation": 1,
        "locked_test": 1,
    }
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
