import subprocess
from pathlib import Path

import pytest

from qubo_receptor_ensemble.full_workflow import (
    ConfigError,
    select_ism_ligands,
    select_preselected_ligands,
    select_manual_receptors,
    select_receptor_manifest,
)


def test_prepare_one_ligand_generates_sdf_and_pdbqt_with_installed_chemistry_stack(
    tmp_path: Path,
) -> None:
    pytest.importorskip("rdkit")
    from qubo_receptor_ensemble.experiment import _prepare_one_ligand
    from qubo_receptor_ensemble.preparation import find_meeko_script

    row = {
        "target_id": "TEST",
        "ligand_id": "L1",
        "smiles": "CCO",
        "label": "active",
        "selection_role": "development",
        "split": "train",
    }

    prepared = _prepare_one_ligand(
        row,
        index=0,
        root=tmp_path,
        sdf_directory=tmp_path / "sdf",
        pdbqt_directory=tmp_path / "pdbqt",
        meeko_script=find_meeko_script(),
        seed=101,
    )

    assert prepared["pdbqt_status"] == "ok"
    assert (tmp_path / "sdf" / "L1.sdf").is_file()
    assert (tmp_path / "pdbqt" / "L1.pdbqt").is_file()


def _fake_pdbqt(atom_type: str) -> str:
    return (
        "ROOT\n"
        f"ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00     0.000 {atom_type}\n"
        "ENDROOT\n"
        "TORSDOF 0\n"
    )


@pytest.mark.parametrize("atom_type", ["CG0", "G0", "G12"])
def test_macrocycle_closure_atom_types_detects_meeko_pseudoatoms(
    atom_type: str, tmp_path: Path
) -> None:
    pytest.importorskip("rdkit")
    from qubo_receptor_ensemble.preparation import macrocycle_closure_atom_types

    pdbqt = tmp_path / "macrocycle.pdbqt"
    pdbqt.write_text(_fake_pdbqt(atom_type), encoding="ascii")

    assert macrocycle_closure_atom_types(pdbqt) == [atom_type]


def test_prepare_one_ligand_retries_macrocycle_with_rigid_meeko(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("rdkit")
    from qubo_receptor_ensemble import preparation
    from qubo_receptor_ensemble.experiment import _prepare_one_ligand

    calls: list[bool] = []

    def fake_run_meeko(
        meeko_script: Path,
        sdf_path: Path,
        pdbqt_path: Path,
        rigid_macrocycles: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        del meeko_script, sdf_path
        calls.append(rigid_macrocycles)
        pdbqt_path.write_text(
            _fake_pdbqt("C" if rigid_macrocycles else "G0"), encoding="ascii"
        )
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(preparation, "run_meeko", fake_run_meeko)
    prepared = _prepare_one_ligand(
        {
            "target_id": "TEST",
            "ligand_id": "L_MACRO",
            "smiles": "CCO",
            "label": "active",
            "selection_role": "development",
            "split": "train",
        },
        index=0,
        root=tmp_path,
        sdf_directory=tmp_path / "sdf",
        pdbqt_directory=tmp_path / "pdbqt",
        meeko_script=tmp_path / "mk_prepare_ligand.py",
        seed=101,
    )

    assert calls == [False, True]
    assert prepared["preparation_variant"] == "meeko_rigid_macrocycles"
    assert prepared["pdbqt_message"] == "meeko_rigid_after_closure_pseudoatom_detection"
    assert prepared["pdbqt_atom_types"] == "C"


def test_prepare_one_ligand_retries_rigid_meeko_after_flexible_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("rdkit")
    from qubo_receptor_ensemble import preparation
    from qubo_receptor_ensemble.experiment import _prepare_one_ligand

    calls: list[bool] = []

    def fake_run_meeko(
        meeko_script: Path,
        sdf_path: Path,
        pdbqt_path: Path,
        rigid_macrocycles: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        del meeko_script, sdf_path
        calls.append(rigid_macrocycles)
        if rigid_macrocycles:
            pdbqt_path.write_text(_fake_pdbqt("C"), encoding="ascii")
            return subprocess.CompletedProcess([], 0, "", "")
        return subprocess.CompletedProcess([], 1, "", "flexible failed")

    monkeypatch.setattr(preparation, "run_meeko", fake_run_meeko)
    prepared = _prepare_one_ligand(
        {
            "target_id": "TEST",
            "ligand_id": "L_RETRY",
            "smiles": "CCO",
            "label": "decoy",
            "selection_role": "development",
            "split": "train",
        },
        index=0,
        root=tmp_path,
        sdf_directory=tmp_path / "sdf",
        pdbqt_directory=tmp_path / "pdbqt",
        meeko_script=tmp_path / "mk_prepare_ligand.py",
        seed=101,
    )

    assert calls == [False, True]
    assert prepared["preparation_variant"] == "meeko_rigid_macrocycles"
    assert prepared["pdbqt_message"] == "meeko_rigid_after_flexible_failure"


def _write_ism(path: Path, rows: list[tuple[str, str]]) -> Path:
    path.write_text(
        "".join(f"{smiles} source-{ligand_id} {ligand_id}\n" for smiles, ligand_id in rows),
        encoding="utf-8",
    )
    return path


def test_select_ism_ligands_preserves_configured_label_quotas(tmp_path: Path) -> None:
    active = _write_ism(
        tmp_path / "active.ism",
        [("CC", "A1"), ("CCC", "A2"), ("CCCC", "A3")],
    )
    decoy = _write_ism(
        tmp_path / "decoy.ism",
        [("CO", "D1"), ("COC", "D2"), ("COCC", "D3")],
    )

    rows = select_ism_ligands(
        active,
        decoy,
        target_id="TEST",
        label_counts={"active": 2, "decoy": 1},
        ordering="manifest_order",
    )

    assert [(row["ligand_id"], row["label"]) for row in rows] == [
        ("A1", "active"),
        ("A2", "active"),
        ("D1", "decoy"),
    ]
    assert all(row["target_id"] == "TEST" for row in rows)


def test_select_ism_ligands_rejects_insufficient_label_source(tmp_path: Path) -> None:
    active = _write_ism(tmp_path / "active.ism", [("CC", "A1")])
    decoy = _write_ism(tmp_path / "decoy.ism", [("CO", "D1")])

    with pytest.raises(ConfigError, match="active"):
        select_ism_ligands(
            active,
            decoy,
            target_id="TEST",
            label_counts={"active": 2, "decoy": 0},
            ordering="manifest_order",
        )


def test_select_ism_ligands_keeps_duplicate_raw_ids_line_addressable(tmp_path: Path) -> None:
    active = _write_ism(tmp_path / "active.ism", [("CC", "A1")])
    decoy = tmp_path / "decoy.ism"
    decoy.write_text(
        "CO source-D1 D1\nCOC source-D1 D1\n",
        encoding="utf-8",
    )

    rows = select_ism_ligands(
        active,
        decoy,
        target_id="TEST",
        label_counts={"active": 1, "decoy": 2},
        ordering="manifest_order",
    )

    assert len({row["ligand_id"] for row in rows}) == 3
    assert all("source_line_number" in row for row in rows)


def test_select_preselected_ligands_preserves_frozen_manifest_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "ligands.csv"
    manifest.write_text(
        "target_id,ligand_id,smiles,label,selection_role,split\n"
        "TEST,A2,CCC,active,development_train,train\n"
        "TEST,D1,CO,decoy,development_train,train\n",
        encoding="utf-8",
    )

    rows = select_preselected_ligands(
        manifest,
        target_id="TEST",
        label_counts={"active": 1, "decoy": 1},
        ligand_count=2,
    )

    assert [row["ligand_id"] for row in rows] == ["A2", "D1"]
    assert [row["selection_role"] for row in rows] == [
        "development_train",
        "development_train",
    ]
    assert all(row["split"] == "train" for row in rows)


def test_select_preselected_ligands_can_use_a_configured_subset(tmp_path: Path) -> None:
    manifest = tmp_path / "ligands.csv"
    manifest.write_text(
        "target_id,ligand_id,smiles,label,selection_role,split\n"
        "TEST,A1,CC,active,development_train,train\n"
        "TEST,D1,CO,decoy,development_train,train\n"
        "TEST,A2,CCC,active,development_train,train\n",
        encoding="utf-8",
    )

    rows = select_preselected_ligands(
        manifest,
        target_id="TEST",
        label_counts={"active": 1, "decoy": 1},
        ligand_count=2,
    )

    assert [row["ligand_id"] for row in rows] == ["A1", "D1"]


@pytest.mark.parametrize(
    ("manifest_text", "error"),
    [
        (
            "target_id,ligand_id,smiles,label,selection_role,split\n"
            "TEST,A1,CC,active,development_train,test\n"
            "TEST,D1,CO,decoy,development_train,train\n",
            "split=train",
        ),
        (
            "target_id,ligand_id,smiles,label,selection_role,split\n"
            "TEST,A1,CC,active,development_train,train\n"
            "TEST,A2,CCC,active,development_train,train\n",
            "label_counts",
        ),
    ],
)
def test_select_preselected_ligands_rejects_invalid_frozen_manifest(
    tmp_path: Path, manifest_text: str, error: str
) -> None:
    manifest = tmp_path / "ligands.csv"
    manifest.write_text(manifest_text, encoding="utf-8")

    with pytest.raises(ConfigError, match=error):
        select_preselected_ligands(
            manifest,
            target_id="TEST",
            label_counts={"active": 1, "decoy": 1},
            ligand_count=2,
        )


def test_select_receptor_manifest_is_deterministic_and_checks_status(tmp_path: Path) -> None:
    manifest = tmp_path / "receptors.csv"
    manifest.write_text(
        "conformer_id,receptor_pdbqt,status,stage102a_gate_pass\n"
        "R1,r1.pdbqt,ok,True\n"
        "R2,r2.pdbqt,ok,True\n"
        "R3,r3.pdbqt,failed,False\n",
        encoding="utf-8",
    )

    rows = select_receptor_manifest(manifest, receptor_count=2)

    assert [row["conformer_id"] for row in rows] == ["R1", "R2"]

    with pytest.raises(ConfigError, match="receptor_count"):
        select_receptor_manifest(manifest, receptor_count=3)


def test_select_manual_receptors_preserves_configured_order() -> None:
    rows = select_manual_receptors(
        [
            {"conformer_id": "R2", "receptor_pdbqt": "r2.pdbqt"},
            {"conformer_id": "R1", "receptor_pdbqt": "r1.pdbqt"},
        ],
        receptor_count=2,
    )

    assert [row["conformer_id"] for row in rows] == ["R2", "R1"]
    assert rows[0]["receptor_pdbqt"] == "r2.pdbqt"


def test_select_manual_receptors_rejects_duplicate_ids() -> None:
    with pytest.raises(ConfigError, match="duplicate"):
        select_manual_receptors(
            [
                {"conformer_id": "R1", "receptor_pdbqt": "r1.pdbqt"},
                {"conformer_id": "R1", "receptor_pdbqt": "r1-copy.pdbqt"},
            ],
            receptor_count=2,
        )
