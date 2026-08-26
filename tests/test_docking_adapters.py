from pathlib import Path
import subprocess

import pytest

from qubo_receptor_ensemble.docking_adapters import (
    UniDockAdapter,
    VinaCpuAdapter,
    get_docking_adapter,
    parse_vina_score,
)
import qubo_receptor_ensemble.docking_adapters as docking_adapters


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


def test_unidock_retries_ligands_with_invalid_batch_poses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "unidock"
    executable.write_text("", encoding="ascii")
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("receptor", encoding="ascii")
    ligand_paths = []
    for ligand_id in ("L1", "L2"):
        path = tmp_path / f"{ligand_id}.pdbqt"
        path.write_text("ligand", encoding="ascii")
        ligand_paths.append(path)
    ligands = [
        {"ligand_id": "L1", "label": "active", "pdbqt_path": "L1.pdbqt"},
        {"ligand_id": "L2", "label": "decoy", "pdbqt_path": "L2.pdbqt"},
    ]
    calls: list[tuple[Path, list[str]]] = []
    observed_seed_values: list[str] = []
    observed_omp_values: list[str] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        observed_omp_values.append(str(environment["OMP_NUM_THREADS"]))
        observed_seed_values.append(command[command.index("--seed") + 1])
        index_path = Path(command[command.index("--ligand_index") + 1])
        pose_directory = Path(command[command.index("--dir") + 1])
        indexed_paths = [Path(line) for line in index_path.read_text().splitlines()]
        calls.append((pose_directory, [path.stem for path in indexed_paths]))
        pose_directory.mkdir(parents=True, exist_ok=True)
        for ligand_path in indexed_paths:
            if len(indexed_paths) > 1 and ligand_path.stem == "L2":
                (pose_directory / f"{ligand_path.stem}_out.pdbqt").write_text(
                    "", encoding="ascii"
                )
                continue
            (pose_directory / f"{ligand_path.stem}_out.pdbqt").write_text(
                "REMARK VINA RESULT:   -7.4      0.0      0.0\n",
                encoding="ascii",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(docking_adapters.subprocess, "run", fake_run)
    monkeypatch.setenv("OMP_NUM_THREADS", "0")

    rows = UniDockAdapter().run_batch(
        target_id="TEST",
        receptor_id="R1",
        receptor_path=receptor,
        ligands=ligands,
        seed=101,
        output_dir=tmp_path / "batch",
        score_table=tmp_path / "scores.csv",
        config={
            **_docking_config("unidock"),
            "docking": {
                **_docking_config("unidock")["docking"],
                "executable": str(executable),
            },
        },
        root=tmp_path,
    )

    assert [row["ligand_id"] for row in rows] == ["L1", "L2"]
    assert calls[0][1] == ["L1", "L2"]
    assert calls[1][1] == ["L2"]
    assert observed_seed_values == ["101", "1000101"]
    assert observed_omp_values == ["1", "1"]


def test_unidock_retry_seed_skips_configured_seed() -> None:
    assert docking_adapters._retry_seed(
        101, {"docking": {"seeds": [101, 1_000_101]}}
    ) == 1_000_102


def test_unidock_does_not_reuse_stale_pose_after_batch_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "unidock"
    executable.write_text("", encoding="ascii")
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("receptor", encoding="ascii")
    ligand = tmp_path / "L1.pdbqt"
    ligand.write_text("ligand", encoding="ascii")
    output_dir = tmp_path / "batch"
    stale_pose_directory = output_dir / "poses"
    stale_pose_directory.mkdir(parents=True)
    (stale_pose_directory / "L1_out.pdbqt").write_text(
        "REMARK VINA RESULT:   -7.4      0.0      0.0\n", encoding="ascii"
    )

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(docking_adapters.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="no valid pose"):
        UniDockAdapter().run_batch(
            target_id="TEST",
            receptor_id="R1",
            receptor_path=receptor,
            ligands=[
                {"ligand_id": "L1", "label": "active", "pdbqt_path": "L1.pdbqt"}
            ],
            seed=101,
            output_dir=output_dir,
            score_table=tmp_path / "scores.csv",
            config={
                **_docking_config("unidock"),
                "docking": {
                    **_docking_config("unidock")["docking"],
                    "executable": str(executable),
                },
            },
            root=tmp_path,
        )
