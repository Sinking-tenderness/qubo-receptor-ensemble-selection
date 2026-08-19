from pathlib import Path

import pytest

from qubo_receptor_ensemble.docking_adapters import (
    UniDockAdapter,
    VinaCpuAdapter,
    get_docking_adapter,
    parse_vina_score,
)


def _docking_config(engine: str) -> dict[str, object]:
    return {
        "target_id": "TEST",
        "docking": {
            "engine": engine,
            "executable": "unidock" if engine == "unidock" else "vina",
            "box": {
                "center_x": 1,
                "center_y": 2,
                "center_z": 3,
                "size_x": 20,
                "size_y": 20,
                "size_z": 20,
            },
            "parameters": {
                "scoring": "vina",
                "exhaustiveness": 8,
                "max_step": 0,
                "refine_step": 5,
                "num_modes": 1,
                "energy_range": 3,
                "verbosity": 1,
                "cpu": 2,
            },
        },
    }


def test_default_adapter_is_local_unidock() -> None:
    adapter = get_docking_adapter(_docking_config("unidock"))

    assert isinstance(adapter, UniDockAdapter)
    assert adapter.name == "unidock"


def test_vina_cpu_is_an_explicit_alternative() -> None:
    adapter = get_docking_adapter(_docking_config("vina_cpu"))

    assert isinstance(adapter, VinaCpuAdapter)
    assert adapter.name == "vina_cpu"


def test_unidock_batch_command_uses_local_executable_and_ligand_index(tmp_path: Path) -> None:
    adapter = UniDockAdapter()
    command = adapter.build_batch_command(
        executable="unidock",
        receptor=tmp_path / "receptor.pdbqt",
        ligand_index=tmp_path / "ligands.index",
        pose_directory=tmp_path / "poses",
        seed=101,
        config=_docking_config("unidock")["docking"],
    )

    assert command[0] == "unidock"
    assert "--ligand_index" in command
    assert "--dir" in command
    assert "--seed" in command
    assert command[command.index("--seed") + 1] == "101"


def test_vina_score_parser_requires_a_real_result() -> None:
    assert parse_vina_score("REMARK VINA RESULT:   -7.4      0.0      0.0") == -7.4

    with pytest.raises(ValueError, match="no Vina result"):
        parse_vina_score("vina failed")
