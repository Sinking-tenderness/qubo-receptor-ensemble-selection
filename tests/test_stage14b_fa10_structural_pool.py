from scripts.select_stage14b_fa10_structural_pool import fa10_conformer_id


def test_fa10_conformer_id_preserves_reference_role() -> None:
    assert fa10_conformer_id("3KL6") == "FA10_3KL6_reference"
    assert fa10_conformer_id("1FJS") == "FA10_1FJS_aligned"
