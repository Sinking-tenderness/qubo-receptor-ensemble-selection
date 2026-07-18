from pathlib import Path

import pytest

from scripts.audit_stage05_expanded_e32_diagnostics import fixed_order_rmsd


def write_pose(path: Path, shift: float) -> None:
    def atom(serial: int, name: str, y: float, atom_type: str) -> str:
        return (
            f"ATOM  {serial:5d}  {name:<3s} UNL     1    "
            f"{shift:8.3f}{y:8.3f}{0.0:8.3f}  1.00  0.00     0.000 {atom_type}\n"
        )

    path.write_text(
        atom(1, "C", 0.0, "C") + atom(2, "H", 1.0, "HD"),
        encoding="ascii",
    )


def test_fixed_order_rmsd_excludes_polar_hydrogen(tmp_path: Path) -> None:
    first = tmp_path / "first.pdbqt"
    second = tmp_path / "second.pdbqt"
    write_pose(first, 0.0)
    write_pose(second, 2.0)

    assert fixed_order_rmsd(first, second) == pytest.approx(2.0)
