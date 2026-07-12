from scripts.download_alphafold_structure import audit_pdb_text, select_entry


def test_select_entry_uses_canonical_accession_only():
    entries = [
        {"isUniProt": True, "uniprotAccession": "P24941-2", "entryId": "AF-P24941-2-F1"},
        {"isUniProt": True, "uniprotAccession": "P24941", "entryId": "AF-P24941-F1"},
    ]
    assert select_entry(entries, "P24941")["entryId"] == "AF-P24941-F1"


def test_audit_pdb_reads_plddt_from_b_factor():
    line = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 88.44           C"
    audit = audit_pdb_text(line + "\n")
    assert audit["atom_count"] == 1
    assert audit["plddt_mean"] == 88.44
