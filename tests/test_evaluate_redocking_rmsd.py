from __future__ import annotations

from scripts.evaluate_redocking_rmsd import parse_vina_affinities


def test_parse_vina_affinities_across_models() -> None:
    text = """MODEL 1
REMARK VINA RESULT:    -10.123      0.000      0.000
ENDMDL
MODEL 2
REMARK VINA RESULT:     -9.500      1.000      2.000
ENDMDL
"""

    assert parse_vina_affinities(text) == [-10.123, -9.5]
