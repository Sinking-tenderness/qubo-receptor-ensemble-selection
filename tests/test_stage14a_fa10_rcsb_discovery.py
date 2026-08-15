from scripts.discover_stage14_fa10_rcsb_candidates import write_report


def test_fa10_report_names_target_and_counts(tmp_path) -> None:
    output = tmp_path / "report.md"
    write_report(
        output,
        {
            "query": {"search_result_count": 20},
            "counts": {
                "metadata_eligible_count": 18,
                "new_metadata_eligible_count": 17,
                "reference_eligible": True,
            },
        },
    )
    text = output.read_text(encoding="ascii")
    assert "FA10" in text
    assert "Metadata-eligible entries: 18" in text
