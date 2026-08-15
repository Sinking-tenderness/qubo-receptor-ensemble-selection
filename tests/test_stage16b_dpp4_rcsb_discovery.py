from scripts.discover_stage16b_dpp4_rcsb_candidates import write_report


def test_dpp4_report_names_target_and_counts(tmp_path) -> None:
    output = tmp_path / "report.md"
    write_report(
        output,
        {
            "query": {"search_result_count": 109},
            "counts": {
                "metadata_eligible_count": 30,
                "new_metadata_eligible_count": 29,
                "reference_eligible": True,
            },
        },
    )
    text = output.read_text(encoding="ascii")
    assert "DPP4" in text
    assert "Metadata-eligible entries: 30" in text
