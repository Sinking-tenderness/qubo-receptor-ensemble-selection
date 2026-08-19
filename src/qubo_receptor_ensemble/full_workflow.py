"""Configuration and data contracts for a source-to-result experiment run."""

from __future__ import annotations

import json
import csv
import random
from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

FULL_WORKFLOW_STAGES = (
    "prepare",
    "dock",
    "aggregate",
    "build_problem",
    "solve",
    "evaluate",
    "persist",
)
SUPPORTED_WORKFLOW_MODES = {"full", "reference_replay"}
SUPPORTED_DOCKING_ENGINES = {"unidock", "vina_cpu"}


class ConfigError(ValueError):
    """Raised when a full experiment config violates its contract."""


@dataclass(frozen=True)
class FullExperimentConfig:
    """Resolved full-workflow configuration.

    ``data`` keeps the JSON-shaped configuration for stage implementations;
    ``paths`` contains only resolved filesystem paths for the known path keys.
    """

    path: Path
    data_root: Path
    data: dict[str, object]
    paths: dict[str, Path]
    stages: tuple[str, ...]
    start_stage: str
    end_stage: str

    @property
    def workflow_mode(self) -> str:
        return str(self.data["workflow_mode"])

    @property
    def selected_stages(self) -> tuple[str, ...]:
        start = self.stages.index(self.start_stage)
        end = self.stages.index(self.end_stage)
        return self.stages[start : end + 1]


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_ism(path: Path, label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            parts = raw_line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ConfigError(f"{label} ISM line {line_number} has no ligand ID")
            rows.append(
                {
                    "smiles": parts[0],
                    "ligand_id": parts[-1],
                    "source_molecule_id": parts[-1],
                    "source_line_number": str(line_number),
                    "label": label,
                }
            )
    if not rows:
        raise ConfigError(f"{label} ISM contains no ligands: {path}")
    ligand_ids = [row["ligand_id"] for row in rows]
    duplicate_ids = {
        ligand_id for ligand_id, count in Counter(ligand_ids).items() if count > 1
    }
    if duplicate_ids:
        for row in rows:
            if row["ligand_id"] in duplicate_ids:
                row["ligand_id"] = (
                    f"{label}_{row['source_molecule_id']}_L{row['source_line_number']}"
                )
    return rows


def select_ism_ligands(
    active_path: str | Path,
    decoy_path: str | Path,
    *,
    target_id: str,
    label_counts: Mapping[str, int],
    ordering: str = "manifest_order",
    sample_seed: int = 0,
) -> list[dict[str, str]]:
    """Select labeled source ligands without allowing class drift."""
    sources = {
        "active": _read_ism(Path(active_path), "active"),
        "decoy": _read_ism(Path(decoy_path), "decoy"),
    }
    if ordering not in {"manifest_order", "seeded_sample"}:
        raise ConfigError(f"unsupported ligand ordering: {ordering}")
    selected: list[dict[str, str]] = []
    for label in ("active", "decoy"):
        requested = int(label_counts.get(label, 0))
        available = sources[label]
        if requested < 0:
            raise ConfigError(f"negative ligand count for {label}")
        if requested > len(available):
            raise ConfigError(
                f"requested {requested} {label} ligands but only {len(available)} are available"
            )
        if ordering == "seeded_sample":
            rng = random.Random(sample_seed + (0 if label == "active" else 1))
            chosen = sorted(rng.sample(available, requested), key=lambda row: row["ligand_id"])
        else:
            chosen = available[:requested]
        selected.extend(
            {
                **row,
                "target_id": target_id,
                "split": "train",
                "selection_role": "development",
            }
            for row in chosen
        )
    if len({row["ligand_id"] for row in selected}) != len(selected):
        raise ConfigError("active and decoy source manifests overlap in ligand IDs")
    return selected


def select_preselected_ligands(
    manifest_path: str | Path,
    *,
    target_id: str,
    label_counts: Mapping[str, int],
    ligand_count: int,
) -> list[dict[str, str]]:
    """Read a frozen ligand allocation without consulting docking artifacts."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if ligand_count <= 0:
        raise ConfigError("selection.ligand_count must be positive")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {"target_id", "ligand_id", "smiles", "label", "selection_role", "split"}
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ConfigError(f"preselected ligand manifest requires: {missing}")
        forbidden = {"docking_score", "representative_score", "pose_rank", "receptor_id"}
        present_forbidden = sorted(forbidden.intersection(fieldnames))
        if present_forbidden:
            raise ConfigError(
                "preselected ligand manifest must not contain score/matrix fields: "
                f"{present_forbidden}"
            )
        rows = [{str(key): str(value or "") for key, value in row.items()} for row in reader]

    if len(rows) < ligand_count:
        raise ConfigError(
            f"preselected ligand manifest has {len(rows)} rows; "
            f"needs at least selection.ligand_count={ligand_count}"
        )
    ligand_ids = [row["ligand_id"] for row in rows]
    if any(not ligand_id for ligand_id in ligand_ids):
        raise ConfigError("preselected ligand manifest contains an empty ligand_id")
    if len(set(ligand_ids)) != len(ligand_ids):
        raise ConfigError("preselected ligand manifest contains duplicate ligand IDs")

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["target_id"] != target_id:
            raise ConfigError(
                f"preselected ligand manifest target mismatch: {row['target_id']} != {target_id}"
            )
        if row["split"] != "train":
            raise ConfigError("preselected ligand manifest requires split=train")
        if row["label"] not in {"active", "decoy"}:
            raise ConfigError(f"preselected ligand manifest has unsupported label: {row['label']}")
        if not row["smiles"] or not row["selection_role"]:
            raise ConfigError("preselected ligand manifest contains an empty required value")
        counts[row["label"]] += 1

    expected_counts = {str(key): int(value) for key, value in label_counts.items()}
    if sum(expected_counts.values()) != ligand_count:
        raise ConfigError("label_counts must sum to selection.ligand_count")
    if any(counts.get(label, 0) < requested for label, requested in expected_counts.items()):
        raise ConfigError(
            f"preselected ligand manifest label_counts={dict(counts)}; "
            f"needs at least {expected_counts}"
        )
    selected: list[dict[str, str]] = []
    selected_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        label = row["label"]
        if selected_counts[label] < expected_counts.get(label, 0):
            selected.append(row)
            selected_counts[label] += 1
    if len(selected) != ligand_count:
        raise ConfigError("preselected ligand manifest could not satisfy configured label_counts")
    return selected


def front_input_keys(config: Mapping[str, object], stage: str) -> tuple[str, ...]:
    """Return the configured front-input path keys required by a stage."""
    selection = _mapping(config.get("selection"), "selection")
    ordering = str(selection.get("ordering", "manifest_order"))
    sources = _mapping(config.get("sources", {}), "sources")
    requirements: dict[str, tuple[str, ...]] = {
        "prepare": ("receptor_manifest",),
        "dock": ("prepared_ligand_manifest", "selected_receptor_manifest"),
        "aggregate": (
            "prepared_ligand_manifest",
            "selected_receptor_manifest",
            "score_tables",
        ),
        "build_problem": ("primary_matrix", "selected_receptor_manifest"),
        "solve": ("problem",),
        "evaluate": ("selection",),
        "persist": ("evaluation",),
    }
    if stage not in requirements:
        raise ConfigError(f"unknown workflow stage: {stage}")
    if stage == "prepare":
        raw_receptor_sources = (
            "reference_receptor_pdb",
            "crystal_ligand",
            "rcsb_directory",
        )
        if all(key in sources for key in raw_receptor_sources):
            ligand_sources = (
                ("ligand_manifest",)
                if ordering == "preselected_manifest"
                else ("active_ism", "decoy_ism")
            )
            return (*ligand_sources, *raw_receptor_sources)
        ligand_sources = (
            ("ligand_manifest",)
            if ordering == "preselected_manifest"
            else ("active_ism", "decoy_ism")
        )
        return (*ligand_sources, "receptor_manifest")
    if stage == "dock" and all(
        key in sources for key in ("reference_receptor_pdb", "crystal_ligand", "rcsb_directory")
    ):
        return (*requirements[stage], "docking_box")
    return requirements[stage]


def select_receptor_manifest(
    manifest_path: str | Path, *, receptor_count: int
) -> list[dict[str, str]]:
    """Select passing receptors in manifest order."""
    if receptor_count <= 0:
        raise ConfigError("selection.receptor_count must be positive")
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"conformer_id", "receptor_pdbqt"}.issubset(
            reader.fieldnames
        ):
            raise ConfigError("receptor manifest requires conformer_id and receptor_pdbqt")
        rows = list(reader)
    passing = [
        row
        for row in rows
        if str(row.get("status", "ok")).lower() == "ok"
        and str(row.get("stage102a_gate_pass", "true")).lower() != "false"
    ]
    if len(passing) < receptor_count:
        raise ConfigError(
            f"selection.receptor_count={receptor_count} exceeds passing receptors={len(passing)}"
        )
    selected = passing[:receptor_count]
    ids = [row["conformer_id"] for row in selected]
    if len(set(ids)) != len(ids):
        raise ConfigError("receptor manifest contains duplicate conformer_id values")
    return selected


def _validate_stages(config: Mapping[str, object]) -> tuple[tuple[str, ...], str, str]:
    configured = config.get("pipeline", list(FULL_WORKFLOW_STAGES))
    if not isinstance(configured, list) or not configured:
        raise ConfigError("pipeline must be a non-empty list")
    stages = tuple(str(stage) for stage in configured)
    if stages != FULL_WORKFLOW_STAGES:
        raise ConfigError(
            "schema 3.0 pipeline must contain the complete canonical stage order"
        )
    start = str(config.get("start_stage", "prepare"))
    end = str(config.get("end_stage", "persist"))
    if start not in stages:
        raise ConfigError(f"unknown start_stage: {start}")
    if end not in stages:
        raise ConfigError(f"unknown end_stage: {end}")
    if stages.index(start) > stages.index(end):
        raise ConfigError("start_stage must not follow end_stage")
    return stages, start, end


def _validate_selection(config: Mapping[str, object]) -> None:
    selection = _mapping(config.get("selection"), "selection")
    receptor_count = _positive_int(
        selection.get("receptor_count"), "selection.receptor_count"
    )
    ligand_count = _positive_int(
        selection.get("ligand_count"), "selection.ligand_count"
    )
    label_counts = _mapping(selection.get("label_counts"), "selection.label_counts")
    if not label_counts:
        raise ConfigError("selection.label_counts must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in label_counts.values()
    ):
        raise ConfigError("selection.label_counts values must be non-negative integers")
    if sum(label_counts.values()) != ligand_count:
        raise ConfigError(
            "selection.label_counts must sum to selection.ligand_count"
        )
    if "active" not in label_counts or "decoy" not in label_counts:
        raise ConfigError("selection.label_counts must include active and decoy")
    ordering = str(selection.get("ordering", "manifest_order"))
    if ordering not in {
        "manifest_order",
        "seeded_sample",
        "scaffold_hash_allocation",
        "preselected_manifest",
    }:
        raise ConfigError(
            "selection.ordering must be manifest_order, seeded_sample, "
            "scaffold_hash_allocation, or preselected_manifest"
        )
    if ordering == "scaffold_hash_allocation":
        policy = _mapping(selection.get("allocation", {}), "selection.allocation")
        fold_count = _positive_int(
            policy.get("outer_fold_count", 5),
            "selection.allocation.outer_fold_count",
        )
        minimums = _mapping(
            policy.get(
                "minimum_label_counts_per_outer_fold",
                {"active": 20, "decoy": 80},
            ),
            "selection.allocation.minimum_label_counts_per_outer_fold",
        )
        for label in ("active", "decoy"):
            minimum = minimums.get(label, 0)
            if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
                raise ConfigError(
                    "selection.allocation minimum fold counts must be "
                    "non-negative integers"
                )
            if minimum * fold_count > int(label_counts[label]):
                raise ConfigError(
                    f"selection.allocation requires more {label} rows than configured"
                )
    if receptor_count < 1 or ligand_count < 1:
        raise ConfigError("selection counts must be positive")


def _validate_docking(config: Mapping[str, object]) -> None:
    docking = _mapping(config.get("docking"), "docking")
    redock = docking.get("redock", True)
    if not isinstance(redock, bool):
        raise ConfigError("docking.redock must be boolean")
    engine = str(docking.get("engine", "unidock"))
    if engine not in SUPPORTED_DOCKING_ENGINES:
        raise ConfigError(
            f"docking.engine must be one of {sorted(SUPPORTED_DOCKING_ENGINES)}"
        )
    seeds = docking.get("seeds", [])
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ConfigError("docking.seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise ConfigError("docking.seeds must be unique")
    _mapping(docking.get("box", {}), "docking.box")
    _mapping(docking.get("parameters", {}), "docking.parameters")


def _validate_paths(
    config: Mapping[str, object], start_stage: str, data_root: Path
) -> dict[str, Path]:
    sources = _mapping(config.get("sources"), "sources")
    paths_config = _mapping(config.get("paths"), "paths")
    required_outputs = {
        "run_directory",
        "prepared_ligand_manifest",
        "selected_receptor_manifest",
        "score_tables",
        "matrices",
        "problem",
        "selection",
        "evaluation",
    }
    missing_outputs = sorted(required_outputs.difference(paths_config))
    if missing_outputs:
        raise ConfigError(f"paths is missing: {missing_outputs}")

    paths: dict[str, Path] = {}
    for key, value in sources.items():
        paths[str(key)] = _resolve(_string(value, f"sources.{key}"), data_root)
    for key, value in paths_config.items():
        paths[str(key)] = _resolve(_string(value, f"paths.{key}"), data_root)

    required = set(front_input_keys(config, start_stage))
    missing = sorted(key for key in required if key not in paths)
    if missing:
        raise ConfigError(
            f"start_stage={start_stage} requires configured front inputs: {missing}"
        )
    return paths


def validate_full_experiment_config(
    config: dict[str, object], *, data_root: Path
) -> tuple[tuple[str, ...], str, str, dict[str, Path]]:
    if config.get("schema_version") != "3.0":
        raise ConfigError("schema_version must be 3.0")
    for key in ("experiment_id", "target_id", "workflow_mode"):
        _string(config.get(key), key)
    mode = str(config["workflow_mode"])
    if mode not in SUPPORTED_WORKFLOW_MODES:
        raise ConfigError(
            f"workflow_mode must be one of {sorted(SUPPORTED_WORKFLOW_MODES)}"
        )
    stages, start, end = _validate_stages(config)
    _validate_selection(config)
    _validate_docking(config)
    docking = _mapping(config["docking"], "docking")
    if mode == "full" and docking.get("redock", True) is not True:
        raise ConfigError(
            "workflow_mode=full requires docking.redock=true; "
            "use reference_replay for existing scores"
        )
    sources = _mapping(config.get("sources"), "sources")
    raw_receptor_sources = {
        "reference_receptor_pdb",
        "crystal_ligand",
        "rcsb_directory",
    }
    if mode == "full" and raw_receptor_sources.issubset(sources):
        box = _mapping(docking.get("box"), "docking.box")
        if box.get("method") != "ligand_bounds":
            raise ConfigError(
                "raw full workflow requires docking.box.method=ligand_bounds"
            )
        try:
            padding = float(box["padding"])
            minimum_size = tuple(float(value) for value in box["minimum_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(
                "raw full workflow requires numeric docking.box.padding and "
                "a three-value docking.box.minimum_size"
            ) from exc
        if padding < 0 or len(minimum_size) != 3 or any(value <= 0 for value in minimum_size):
            raise ConfigError(
                "raw full workflow requires non-negative padding and three positive "
                "minimum_size values"
            )
    problem = _mapping(config.get("problem"), "problem")
    _string(problem.get("type"), "problem.type")
    _string(problem.get("strategy"), "problem.strategy")
    solve = _mapping(config.get("solve"), "solve")
    _string(solve.get("backend"), "solve.backend")
    evaluate = _mapping(config.get("evaluate"), "evaluate")
    metrics = evaluate.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ConfigError("evaluate.metrics must be a non-empty list")
    paths = _validate_paths(config, start, data_root)
    return stages, start, end, paths


def load_full_experiment_config(
    config_path: str | Path, *, data_root: str | Path | None = None
) -> FullExperimentConfig:
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ConfigError("config JSON root must be an object")
    root = _resolve(data_root or path.parent, Path.cwd())
    stages, start, end, paths = validate_full_experiment_config(
        value, data_root=root
    )
    normalized = json.loads(json.dumps(value))
    docking = normalized.setdefault("docking", {})
    docking.setdefault("redock", True)
    docking.setdefault("engine", "unidock")
    normalized.setdefault("start_stage", "prepare")
    normalized.setdefault("end_stage", "persist")
    return FullExperimentConfig(
        path=path,
        data_root=root,
        data=normalized,
        paths=paths,
        stages=stages,
        start_stage=start,
        end_stage=end,
    )
