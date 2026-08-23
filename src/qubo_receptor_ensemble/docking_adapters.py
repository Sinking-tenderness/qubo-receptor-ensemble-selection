"""Local docking adapters with one normalized score-table contract."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Protocol

from .io import safe_filename, write_csv


_VINA_RESULT = re.compile(
    r"^REMARK\s+VINA\s+RESULT:\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
    re.MULTILINE,
)


def parse_vina_score(text: str) -> float:
    match = _VINA_RESULT.search(text)
    if match is None:
        raise ValueError("no Vina result found in docking output")
    return float(match.group(1))


def _resolve_executable(value: object, label: str) -> str:
    requested = str(value or label)
    candidate = Path(requested)
    if candidate.is_file():
        return str(candidate.resolve())
    resolved = shutil.which(requested)
    if resolved is None:
        raise FileNotFoundError(
            f"{label} executable was not found locally: {requested}"
        )
    return resolved


def _docking_values(config: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    docking = config.get("docking", config)
    if not isinstance(docking, dict):
        raise ValueError("docking configuration must be an object")
    box = docking.get("box", {})
    parameters = docking.get("parameters", {})
    if not isinstance(box, dict) or not isinstance(parameters, dict):
        raise ValueError("docking.box and docking.parameters must be objects")
    return box, parameters


def _required_box(box: dict[str, object]) -> dict[str, object]:
    required = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
    missing = [key for key in required if key not in box]
    if missing:
        raise ValueError(f"docking.box is missing: {missing}")
    return box


def _relative_or_absolute(path: object, root: Path | None) -> Path:
    value = Path(str(path))
    if value.is_absolute() or root is None:
        return value
    return (root / value).resolve()


def _subprocess_environment() -> dict[str, str]:
    """Avoid invalid OpenMP settings inherited from remote shells."""
    environment = os.environ.copy()
    raw_threads = environment.get("OMP_NUM_THREADS")
    if raw_threads is not None:
        try:
            valid_threads = int(raw_threads) > 0
        except ValueError:
            valid_threads = False
        if not valid_threads:
            environment["OMP_NUM_THREADS"] = "1"
    return environment


def _score_row(
    *,
    target_id: str,
    receptor_id: str,
    ligand: dict[str, str],
    score: float,
    seed: int,
    engine: str,
    pose_path: Path,
    log_path: Path,
) -> dict[str, object]:
    return {
        "target_id": target_id,
        "receptor_id": receptor_id,
        "ligand_id": ligand["ligand_id"],
        "label": ligand["label"],
        "pose_rank": 1,
        "docking_score": score,
        "status": "ok",
        "seed": seed,
        "engine": engine,
        "pose_path": pose_path.as_posix(),
        "log_path": log_path.as_posix(),
    }


class DockingAdapter(Protocol):
    name: str

    def run_batch(
        self,
        *,
        target_id: str,
        receptor_id: str,
        receptor_path: Path,
        ligands: list[dict[str, str]],
        seed: int,
        output_dir: Path,
        score_table: Path,
        config: dict[str, object],
        root: Path | None = None,
        resume: bool = False,
    ) -> list[dict[str, object]]:
        ...


class UniDockAdapter:
    name = "unidock"

    def build_batch_command(
        self,
        *,
        executable: str,
        receptor: Path,
        ligand_index: Path,
        pose_directory: Path,
        seed: int,
        config: dict[str, object],
    ) -> list[str]:
        box, parameters = _docking_values(config)
        _required_box(box)
        values = {
            "scoring": parameters.get("scoring", "vina"),
            "exhaustiveness": parameters.get("exhaustiveness", 8),
            "max_step": parameters.get("max_step", 0),
            "refine_step": parameters.get("refine_step", 5),
            "num_modes": parameters.get("num_modes", 1),
            "energy_range": parameters.get("energy_range", 3),
            "verbosity": parameters.get("verbosity", 1),
        }
        return [
            executable,
            "--receptor",
            str(receptor),
            "--ligand_index",
            str(ligand_index),
            "--scoring",
            str(values["scoring"]),
            "--center_x",
            str(box["center_x"]),
            "--center_y",
            str(box["center_y"]),
            "--center_z",
            str(box["center_z"]),
            "--size_x",
            str(box["size_x"]),
            "--size_y",
            str(box["size_y"]),
            "--size_z",
            str(box["size_z"]),
            "--exhaustiveness",
            str(values["exhaustiveness"]),
            "--max_step",
            str(values["max_step"]),
            "--refine_step",
            str(values["refine_step"]),
            "--num_modes",
            str(values["num_modes"]),
            "--energy_range",
            str(values["energy_range"]),
            "--verbosity",
            str(values["verbosity"]),
            "--seed",
            str(seed),
            "--dir",
            str(pose_directory),
        ]

    def run_batch(
        self,
        *,
        target_id: str,
        receptor_id: str,
        receptor_path: Path,
        ligands: list[dict[str, str]],
        seed: int,
        output_dir: Path,
        score_table: Path,
        config: dict[str, object],
        root: Path | None = None,
        resume: bool = False,
    ) -> list[dict[str, object]]:
        if resume and score_table.is_file():
            rows = _read_score_table(score_table)
            if _complete_score_rows(rows, ligands, receptor_id, seed):
                return rows
        executable = _resolve_executable(
            config.get("docking", {}).get("executable", "unidock")
            if isinstance(config.get("docking"), dict)
            else "unidock",
            "Uni-Dock",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        ligand_index = output_dir / "ligands.index"
        pose_directory = output_dir / "poses"
        log_path = output_dir / "unidock.log"
        _reset_pose_directory(pose_directory)
        ligand_paths = [
            _relative_or_absolute(row["pdbqt_path"], root) for row in ligands
        ]
        if any(not path.is_file() for path in ligand_paths):
            missing = [str(path) for path in ligand_paths if not path.is_file()]
            raise FileNotFoundError(f"missing ligand PDBQT files: {missing[:5]}")
        if not receptor_path.is_file():
            raise FileNotFoundError(receptor_path)
        ligand_index.write_text(
            "\n".join(str(path) for path in ligand_paths) + "\n", encoding="utf-8"
        )
        command = self.build_batch_command(
            executable=executable,
            receptor=receptor_path,
            ligand_index=ligand_index,
            pose_directory=pose_directory,
            seed=seed,
            config=config,
        )
        completed = _run_unidock_command(command, log_path=log_path, root=root)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Uni-Dock failed for {seed}/{receptor_id}; see {log_path}"
            )
        rows: list[dict[str, object]] = []
        for ligand in ligands:
            try:
                pose_path = _find_pose(pose_directory, ligand)
                score = parse_vina_score(
                    pose_path.read_text(encoding="utf-8", errors="replace")
                )
                ligand_log_path = log_path
            except (FileNotFoundError, ValueError) as exc:
                pose_path, score, ligand_log_path = _retry_unidock_ligand(
                    executable=executable,
                    receptor=receptor_path,
                    ligand=ligand,
                    seed=seed,
                    output_dir=output_dir,
                    config=config,
                    root=root,
                    original_error=exc,
                )
            rows.append(
                _score_row(
                    target_id=target_id,
                    receptor_id=receptor_id,
                    ligand=ligand,
                    score=score,
                    seed=seed,
                    engine=self.name,
                    pose_path=pose_path,
                    log_path=ligand_log_path,
                )
            )
        write_csv(score_table, rows)
        return rows


class VinaCpuAdapter:
    name = "vina_cpu"

    def build_ligand_command(
        self,
        *,
        executable: str,
        receptor: Path,
        ligand: Path,
        output_pose: Path,
        seed: int,
        config: dict[str, object],
    ) -> list[str]:
        box, parameters = _docking_values(config)
        _required_box(box)
        command = [
            executable,
            "--receptor",
            str(receptor),
            "--ligand",
            str(ligand),
            "--center_x",
            str(box["center_x"]),
            "--center_y",
            str(box["center_y"]),
            "--center_z",
            str(box["center_z"]),
            "--size_x",
            str(box["size_x"]),
            "--size_y",
            str(box["size_y"]),
            "--size_z",
            str(box["size_z"]),
            "--exhaustiveness",
            str(parameters.get("exhaustiveness", 8)),
            "--num_modes",
            str(parameters.get("num_modes", 1)),
            "--seed",
            str(seed),
            "--out",
            str(output_pose),
        ]
        cpu = parameters.get("cpu")
        if cpu is not None:
            command.extend(["--cpu", str(cpu)])
        return command

    def run_batch(
        self,
        *,
        target_id: str,
        receptor_id: str,
        receptor_path: Path,
        ligands: list[dict[str, str]],
        seed: int,
        output_dir: Path,
        score_table: Path,
        config: dict[str, object],
        root: Path | None = None,
        resume: bool = False,
    ) -> list[dict[str, object]]:
        if resume and score_table.is_file():
            rows = _read_score_table(score_table)
            if _complete_score_rows(rows, ligands, receptor_id, seed):
                return rows
        docking = config.get("docking", {})
        executable_value = docking.get("executable", "vina") if isinstance(docking, dict) else "vina"
        executable = _resolve_executable(executable_value, "VinaCPU")
        if not receptor_path.is_file():
            raise FileNotFoundError(receptor_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        for ligand in ligands:
            ligand_path = _relative_or_absolute(ligand["pdbqt_path"], root)
            if not ligand_path.is_file():
                raise FileNotFoundError(ligand_path)
            safe_id = safe_filename(ligand["ligand_id"])
            output_pose = output_dir / f"{safe_id}_out.pdbqt"
            log_path = output_dir / f"{safe_id}.log"
            command = self.build_ligand_command(
                executable=executable,
                receptor=receptor_path,
                ligand=ligand_path,
                output_pose=output_pose,
                seed=seed,
                config=config,
            )
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            log_path.write_text(
                "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip()),
                encoding="utf-8",
            )
            if completed.returncode != 0 or not output_pose.is_file():
                raise RuntimeError(f"VinaCPU failed for {ligand['ligand_id']}; see {log_path}")
            score = parse_vina_score(output_pose.read_text(encoding="utf-8", errors="replace"))
            rows.append(
                _score_row(
                    target_id=target_id,
                    receptor_id=receptor_id,
                    ligand=ligand,
                    score=score,
                    seed=seed,
                    engine=self.name,
                    pose_path=output_pose,
                    log_path=log_path,
                )
            )
        write_csv(score_table, rows)
        return rows


def _find_pose(directory: Path, ligand: dict[str, str]) -> Path:
    stem = Path(ligand["pdbqt_path"]).stem
    candidates = [
        directory / f"{stem}_out.pdbqt",
        directory / f"{safe_filename(ligand['ligand_id'])}_out.pdbqt",
        directory / f"{stem}.pdbqt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = [
        path
        for path in directory.glob("*.pdbqt")
        if path.stem.startswith(stem) or path.stem.startswith(safe_filename(ligand["ligand_id"]))
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"missing Uni-Dock pose for {ligand['ligand_id']}")


def _run_unidock_command(
    command: list[str], *, log_path: Path, root: Path | None
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        return subprocess.run(
            command,
            cwd=root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=_subprocess_environment(),
        )


def _retry_unidock_ligand(
    *,
    executable: str,
    receptor: Path,
    ligand: dict[str, str],
    seed: int,
    output_dir: Path,
    config: dict[str, object],
    root: Path | None,
    original_error: Exception,
) -> tuple[Path, float, Path]:
    """Retry one missing/invalid batch pose without dropping the ligand."""
    retry_directory = output_dir / "retries" / safe_filename(ligand["ligand_id"])
    retry_directory.mkdir(parents=True, exist_ok=True)
    retry_index = retry_directory / "ligands.index"
    ligand_path = _relative_or_absolute(ligand["pdbqt_path"], root)
    retry_index.write_text(f"{ligand_path}\n", encoding="utf-8")
    pose_directory = retry_directory / "poses"
    _reset_pose_directory(pose_directory)
    log_path = retry_directory / "unidock.log"
    command = UniDockAdapter().build_batch_command(
        executable=executable,
        receptor=receptor,
        ligand_index=retry_index,
        pose_directory=pose_directory,
        seed=seed,
        config=config,
    )
    completed = _run_unidock_command(command, log_path=log_path, root=root)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Uni-Dock retry failed for ligand {ligand['ligand_id']}; "
            f"original_error={original_error}; see {log_path}"
        )
    try:
        pose_path = _find_pose(pose_directory, ligand)
        score = parse_vina_score(
            pose_path.read_text(encoding="utf-8", errors="replace")
        )
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(
            f"Uni-Dock retry produced no valid pose for ligand {ligand['ligand_id']}; "
            f"original_error={original_error}; retry_error={exc}; see {log_path}"
        ) from exc
    return pose_path, score, log_path


def _reset_pose_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def _read_score_table(path: Path) -> list[dict[str, object]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _complete_score_rows(
    rows: list[dict[str, object]], ligands: list[dict[str, str]], receptor_id: str, seed: int
) -> bool:
    expected = {row["ligand_id"] for row in ligands}
    observed = {str(row.get("ligand_id", "")) for row in rows}
    return (
        len(rows) == len(ligands)
        and observed == expected
        and all(str(row.get("receptor_id")) == receptor_id for row in rows)
        and all(int(float(str(row.get("seed", seed)))) == seed for row in rows)
        and all(str(row.get("status")) == "ok" for row in rows)
    )


def get_docking_adapter(config: dict[str, object]) -> DockingAdapter:
    docking = config.get("docking", {})
    if not isinstance(docking, dict):
        raise ValueError("docking configuration must be an object")
    engine = str(docking.get("engine", "unidock"))
    if engine == "unidock":
        return UniDockAdapter()
    if engine == "vina_cpu":
        return VinaCpuAdapter()
    raise ValueError(f"unsupported docking engine: {engine}")
