from pathlib import Path

from scripts.audit_stage15_hivpr_source import ca_coordinates


def test_dude_chain_b_numbering_is_normalized(tmp_path: Path) -> None:
    path = tmp_path / "receptor.pdb"
    path.write_text(
        "ATOM      1  CA  PRO A   1       1.000   2.000   3.000\n"
        "ATOM      2  CA  PRO B 100       4.000   5.000   6.000\n",
        encoding="ascii",
    )

    result = ca_coordinates(path, dude_numbering=True)

    assert result[("A", 1)] == (1.0, 2.0, 3.0)
    assert result[("B", 1)] == (4.0, 5.0, 6.0)
