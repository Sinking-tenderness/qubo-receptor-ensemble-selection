from scripts.discover_stage13_external_target_rcsb_candidates import (
    apply_title_exclusions,
)


def test_title_exclusions_are_case_insensitive_and_preserve_prior_reasons() -> None:
    row = {
        "title": "Wild-Type EGFR in Covalent Complex",
        "status": "metadata_excluded",
        "exclusion_reasons": "resolution_above_limit",
    }
    result = apply_title_exclusions(row, ["covalent", "mutant"])

    assert result["status"] == "metadata_excluded"
    assert result["excluded_title_patterns"] == "covalent"
    assert result["exclusion_reasons"] == (
        "excluded_title_pattern;resolution_above_limit"
    )


def test_title_exclusions_leave_reversible_wild_type_entry_eligible() -> None:
    row = {
        "title": "Wild-type EGFR bound with reversible inhibitor",
        "status": "metadata_eligible",
        "exclusion_reasons": "",
    }
    result = apply_title_exclusions(row, ["covalent", "mutant", "\\bapo\\b"])

    assert result["status"] == "metadata_eligible"
    assert result["excluded_title_patterns"] == ""
