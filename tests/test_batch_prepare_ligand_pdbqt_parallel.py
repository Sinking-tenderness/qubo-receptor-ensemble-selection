from scripts.batch_prepare_ligand_pdbqt_parallel import read_existing


def test_read_existing_empty_path(tmp_path):
    assert read_existing(tmp_path / "missing.csv") == {}
