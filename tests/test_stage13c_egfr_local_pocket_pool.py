from scripts.select_stage13c_egfr_local_pocket_pool import (
    citation_is_excluded,
    clean_cif_text,
)


def test_citation_gate_matches_targeted_covalent_study() -> None:
    assert citation_is_excluded(
        "Profiling and Optimizing Targeted Covalent Inhibitors through EGFR-Guided Studies.",
        "10.1021/acs.jmedchem.5c01661",
        r"targeted\s+covalent\s+inhibitors",
        "10.1021/acs.jmedchem.5c01661",
    )


def test_citation_gate_does_not_exclude_reversible_control_by_generic_context() -> None:
    assert not citation_is_excluded(
        "Structural and functional studies of covalent EGFR inhibitor",
        "?",
        r"targeted\s+covalent\s+inhibitors",
        "10.1021/acs.jmedchem.5c01661",
    )


def test_clean_cif_text_normalizes_multiline_values() -> None:
    assert clean_cif_text(";A title\nwith spacing\n;") == "A title with spacing"
