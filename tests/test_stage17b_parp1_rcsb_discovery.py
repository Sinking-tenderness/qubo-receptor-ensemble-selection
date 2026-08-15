from scripts.discover_stage17b_parp1_rcsb_candidates import write_report


def test_parp1_report_names_target(tmp_path) -> None:
    output = tmp_path / "report.md"
    write_report(
        output,
        {
            "query": {"search_result_count": 94},
            "counts": {"metadata_eligible_count": 30, "new_metadata_eligible_count": 29, "reference_eligible": True},
        },
    )
    assert "PARP1" in output.read_text(encoding="ascii")
