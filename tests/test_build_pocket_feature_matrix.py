from scripts.build_pocket_feature_matrix import residue_label


def test_residue_label_is_stable_for_blank_insertion_code():
    row = {"residue_chain": "A", "residue_number": "10", "insertion_code": "", "residue_name": "ILE"}
    assert residue_label(row) == "10___ILE"
