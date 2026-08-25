"""Configuration and data contracts for a source-to-result experiment run."""

from __future__ import annotations

import json
import csv
import math
import random
from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from .ligand_selection import scaffold_smiles
from .methods import MethodRegistryError, get_method_spec

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
            try:
                scaffold = scaffold_smiles(parts[0])
            except ValueError as exc:
                raise ConfigError(
                    f"{label} ISM line {line_number} has invalid SMILES"
                ) from exc
            rows.append(
                {
                    "smiles": parts[0],
                    "ligand_id": parts[-1],
                    "source_molecule_id": parts[-1],
                    "source_line_number": str(line_number),
                    "label": label,
                    "scaffold_smiles": scaffold,
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


def select_manual_ligands(
    active_path: str | Path,
    decoy_path: str | Path,
    ligand_ids: object,
    *,
    target_id: str,
    label_counts: Mapping[str, int],
    ligand_count: int,
) -> list[dict[str, str]]:
    """Resolve an explicit ligand ID list against the raw ISM files."""
    if ligand_count <= 0:
        raise ConfigError("selection.ligand_count must be positive")
    if not isinstance(ligand_ids, list) or any(
        not isinstance(value, str) or not value.strip() for value in ligand_ids
    ):
        raise ConfigError("selection.ligand_ids must be a list of non-empty strings")
    ids = [value.strip() for value in ligand_ids]
    if len(ids) != ligand_count:
        raise ConfigError("selection.ligand_ids length must equal selection.ligand_count")
    if len(set(ids)) != len(ids):
        raise ConfigError("selection.ligand_ids contains duplicate ligand IDs")

    expected_counts = {str(key): int(value) for key, value in label_counts.items()}
    if sum(expected_counts.values()) != ligand_count:
        raise ConfigError("label_counts must sum to selection.ligand_count")
    raw_rows = [
        *_read_ism(Path(active_path), "active"),
        *_read_ism(Path(decoy_path), "decoy"),
    ]
    rows_by_id: dict[str, dict[str, str]] = {}
    for row in raw_rows:
        label = row["label"]
        source_line = int(row["source_line_number"])
        manual_id = f"{target_id}_{label}_L{source_line:06d}"
        if manual_id in rows_by_id:
            raise ConfigError(f"raw ISM rows produce duplicate manual ligand ID: {manual_id}")
        rows_by_id[manual_id] = {
            **row,
            "target_id": target_id,
            "ligand_id": manual_id,
            "split": "train",
            "selection_role": "development_train",
        }

    missing = [ligand_id for ligand_id in ids if ligand_id not in rows_by_id]
    if missing:
        raise ConfigError(f"manual ligand selection contains missing IDs: {missing[:5]}")
    selected = [rows_by_id[ligand_id] for ligand_id in ids]
    selected_counts = Counter(row["label"] for row in selected)
    if any(selected_counts.get(label, 0) != count for label, count in expected_counts.items()):
        raise ConfigError(
            f"manual ligand selection label_counts={dict(selected_counts)}; "
            f"expected {expected_counts}"
        )
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


def select_manual_receptors(
    receptors: object, *, receptor_count: int
) -> list[dict[str, str]]:
    """Use an explicit ordered receptor list without requiring a manifest file."""
    if receptor_count <= 0:
        raise ConfigError("selection.receptor_count must be positive")
    if not isinstance(receptors, list):
        raise ConfigError("selection.receptor_selection.receptors must be a list")
    if len(receptors) != receptor_count:
        raise ConfigError(
            "selection.receptor_selection.receptors length must equal "
            "selection.receptor_count"
        )

    selected: list[dict[str, str]] = []
    for index, value in enumerate(receptors):
        if not isinstance(value, dict):
            raise ConfigError(
                f"selection.receptor_selection.receptors[{index}] must be an object"
            )
        if not isinstance(value.get("conformer_id"), str) or not value["conformer_id"].strip():
            raise ConfigError(
                f"manual receptor entry {index} requires non-empty: ['conformer_id']"
            )
        receptor_pdbqt = value.get("receptor_pdbqt")
        rcsb_id = value.get("rcsb_id")
        has_pdbqt = isinstance(receptor_pdbqt, str) and bool(receptor_pdbqt.strip())
        has_rcsb = isinstance(rcsb_id, str) and bool(rcsb_id.strip())
        if not has_pdbqt and not has_rcsb:
            raise ConfigError(
                f"manual receptor entry {index} requires non-empty: "
                "['receptor_pdbqt'] or ['rcsb_id']"
            )
        row = {str(key): str(item) for key, item in value.items()}
        row["conformer_id"] = row["conformer_id"].strip()
        if "receptor_pdbqt" in row:
            row["receptor_pdbqt"] = row["receptor_pdbqt"].strip()
        if "rcsb_id" in row:
            row["rcsb_id"] = row["rcsb_id"].strip().upper()
        selected.append(row)

    ids = [row["conformer_id"] for row in selected]
    if len(set(ids)) != len(ids):
        raise ConfigError("manual receptor selection contains duplicate conformer_id values")
    return selected


def front_input_keys(config: Mapping[str, object], stage: str) -> tuple[str, ...]:
    """Return the configured front-input path keys required by a stage."""
    selection = _mapping(config.get("selection"), "selection")
    ordering = str(selection.get("ordering", "manifest_order"))
    receptor_selection = selection.get("receptor_selection")
    manual_receptors = (
        isinstance(receptor_selection, dict)
        and str(receptor_selection.get("mode", "")) == "manual"
    )
    sources = _mapping(config.get("sources", {}), "sources")
    receptor_outputs = () if manual_receptors else ("selected_receptor_manifest",)
    requirements: dict[str, tuple[str, ...]] = {
        "prepare": ("receptor_manifest",),
        "dock": ("prepared_ligand_manifest", *receptor_outputs),
        "aggregate": (
            "prepared_ligand_manifest",
            *receptor_outputs,
            "score_tables",
        ),
        "build_problem": ("primary_matrix", *receptor_outputs),
        "solve": ("problem",),
        "evaluate": ("selection",),
        "persist": ("evaluation",),
    }
    if stage not in requirements:
        raise ConfigError(f"unknown workflow stage: {stage}")
    if stage == "build_problem":
        problem = _mapping(config.get("problem"), "problem")
        k_policy = problem.get("k_policy")
        adaptive = isinstance(k_policy, dict) and k_policy.get("mode") == "adaptive"
        if adaptive:
            requirements[stage] = (
                "primary_matrix",
                *receptor_outputs,
                "prepared_ligand_manifest",
            )
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
        if manual_receptors:
            return ligand_sources
        return (*ligand_sources, "receptor_manifest")
    docking = _mapping(config.get("docking", {}), "docking")
    if stage == "dock" and all(
        key in sources for key in ("reference_receptor_pdb", "crystal_ligand", "rcsb_directory")
    ) and str(_mapping(docking.get("box", {}), "docking.box").get("method", "")) == "ligand_bounds":
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
        "manual_ids",
    }:
        raise ConfigError(
            "selection.ordering must be manifest_order, seeded_sample, "
            "scaffold_hash_allocation, preselected_manifest, or manual_ids"
        )
    if ordering == "manual_ids":
        ligand_ids = selection.get("ligand_ids")
        if not isinstance(ligand_ids, list) or any(
            not isinstance(value, str) or not value.strip() for value in ligand_ids
        ):
            raise ConfigError("selection.ligand_ids must be a list of non-empty strings")
        if len(ligand_ids) != ligand_count:
            raise ConfigError("selection.ligand_ids length must equal selection.ligand_count")
        if len(set(ligand_ids)) != len(ligand_ids):
            raise ConfigError("selection.ligand_ids contains duplicate ligand IDs")
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
    receptor_selection = selection.get("receptor_selection")
    if receptor_selection is not None:
        policy = _mapping(receptor_selection, "selection.receptor_selection")
        mode = str(policy.get("mode", ""))
        if mode != "manual":
            raise ConfigError("selection.receptor_selection.mode must be manual")
        select_manual_receptors(
            policy.get("receptors"),
            receptor_count=receptor_count,
        )


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


def _validate_problem(config: Mapping[str, object]) -> None:
    problem = _mapping(config.get("problem"), "problem")
    _string(problem.get("type"), "problem.type")
    mode = str(problem.get("mode", "single"))
    if mode not in {"single", "compare"}:
        raise ConfigError("problem.mode must be single or compare")
    _string(
        problem.get("strategy", "method_registry" if mode == "compare" else "qubo"),
        "problem.strategy",
    )
    utility_metric = str(problem.get("utility_metric", "bedroc"))
    if utility_metric not in {"roc_auc", "bedroc", "ef5"}:
        raise ConfigError(
            "problem.utility_metric must be roc_auc, bedroc, or ef5"
        )
    alpha = problem.get("bedroc_alpha", 20.0)
    if isinstance(alpha, bool):
        raise ConfigError("problem.bedroc_alpha must be a positive number")
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ConfigError("problem.bedroc_alpha must be a positive number") from exc
    if alpha_value <= 0 or not math.isfinite(alpha_value):
        raise ConfigError("problem.bedroc_alpha must be a positive finite number")
    k_policy = problem.get("k_policy")
    if k_policy is not None:
        policy = _mapping(k_policy, "problem.k_policy")
        policy_mode = str(policy.get("mode", ""))
        if policy_mode != "adaptive":
            raise ConfigError("problem.k_policy.mode must be adaptive")
        if mode == "compare":
            raise ConfigError(
                "problem.k_policy adaptive mode is not supported with problem.mode=compare"
            )
        if str(problem.get("strategy", "qubo")) not in {"qubo", "basic_qubo"}:
            raise ConfigError(
                "problem.k_policy adaptive mode requires problem.strategy=qubo or basic_qubo"
            )
        selector = str(policy.get("selector", ""))
        if selector not in {"mechanistic_bootstrap_lcb", "risk_adjusted_oof"}:
            raise ConfigError(
                "problem.k_policy.selector must be mechanistic_bootstrap_lcb or risk_adjusted_oof"
            )
        selection = _mapping(config.get("selection"), "selection")
        receptor_count = _positive_int(
            selection.get("receptor_count"), "selection.receptor_count"
        )
        candidates = policy.get(
            "candidates", list(range(1, receptor_count + 1))
        )
        if not isinstance(candidates, list) or not candidates:
            raise ConfigError("problem.k_policy.candidates must be a non-empty list")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in candidates
        ):
            raise ConfigError(
                "problem.k_policy.candidates must contain positive integers"
            )
        if (
            candidates[0] != 1
            or candidates != sorted(set(candidates))
            or candidates != list(range(1, candidates[-1] + 1))
        ):
            raise ConfigError(
                "problem.k_policy.candidates must be consecutive, unique, ascending, "
                "and start at 1"
            )
        if any(value > receptor_count for value in candidates):
            raise ConfigError(
                "problem.k_policy.candidates cannot exceed selection.receptor_count"
            )
        _string(policy.get("scaffold_field", "scaffold_smiles"), "problem.k_policy.scaffold_field")
        inner_fold_count = policy.get("inner_fold_count", 3)
        if (
            isinstance(inner_fold_count, bool)
            or not isinstance(inner_fold_count, int)
            or inner_fold_count < 2
        ):
            raise ConfigError(
                "problem.k_policy.inner_fold_count must be an integer >= 2"
            )
        bootstrap_iterations = policy.get("bootstrap_iterations", 1000)
        if (
            isinstance(bootstrap_iterations, bool)
            or not isinstance(bootstrap_iterations, int)
            or bootstrap_iterations <= 0
        ):
            raise ConfigError(
                "problem.k_policy.bootstrap_iterations must be a positive integer"
            )
        lower_quantile = policy.get("lower_quantile", 0.025)
        if isinstance(lower_quantile, bool):
            raise ConfigError("problem.k_policy.lower_quantile must be between 0 and 1")
        try:
            lower_quantile_value = float(lower_quantile)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "problem.k_policy.lower_quantile must be between 0 and 1"
            ) from exc
        if not 0.0 <= lower_quantile_value <= 1.0 or not math.isfinite(
            lower_quantile_value
        ):
            raise ConfigError("problem.k_policy.lower_quantile must be between 0 and 1")
        minimum_effect = policy.get("minimum_effect", 0.0)
        required_probability = policy.get("required_probability", 0.5)
        cost_per_receptor = policy.get("cost_per_receptor", 0.0)
        selection_tie_tolerance = policy.get("selection_tie_tolerance", 0.0)
        for value, name in (
            (minimum_effect, "minimum_effect"),
            (required_probability, "required_probability"),
            (cost_per_receptor, "cost_per_receptor"),
            (selection_tie_tolerance, "selection_tie_tolerance"),
        ):
            if isinstance(value, bool):
                raise ConfigError(f"problem.k_policy.{name} must be numeric")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"problem.k_policy.{name} must be numeric") from exc
            if not math.isfinite(numeric_value):
                raise ConfigError(f"problem.k_policy.{name} must be finite")
        if not 0.0 <= float(required_probability) <= 1.0:
            raise ConfigError(
                "problem.k_policy.required_probability must be between 0 and 1"
            )
        if float(cost_per_receptor) < 0.0:
            raise ConfigError(
                "problem.k_policy.cost_per_receptor must be non-negative"
            )
        if float(selection_tie_tolerance) < 0.0:
            raise ConfigError(
                "problem.k_policy.selection_tie_tolerance must be non-negative"
            )
        require_rescue_contrast = policy.get("require_rescue_contrast", False)
        if not isinstance(require_rescue_contrast, bool):
            raise ConfigError(
                "problem.k_policy.require_rescue_contrast must be boolean"
            )
        rescue_fractions = policy.get("rescue_fractions", [0.01, 0.05])
        if not isinstance(rescue_fractions, list) or not rescue_fractions:
            raise ConfigError(
                "problem.k_policy.rescue_fractions must be a non-empty list"
            )
        for fraction in rescue_fractions:
            if isinstance(fraction, bool):
                raise ConfigError(
                    "problem.k_policy.rescue_fractions must be in (0, 1]"
                )
            try:
                fraction_value = float(fraction)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "problem.k_policy.rescue_fractions must be in (0, 1]"
                ) from exc
            if not 0.0 < fraction_value <= 1.0 or not math.isfinite(fraction_value):
                raise ConfigError(
                    "problem.k_policy.rescue_fractions must be in (0, 1]"
                )
        random_seed = policy.get("random_seed", 0)
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise ConfigError("problem.k_policy.random_seed must be an integer")
    if mode == "compare":
        methods = problem.get("methods")
        if not isinstance(methods, list) or not methods:
            raise ConfigError("problem.methods must be a non-empty list")
        for index, item in enumerate(methods):
            if isinstance(item, str):
                method_id = item
            elif isinstance(item, dict):
                method_id = item.get("id", item.get("method_id", ""))
            else:
                raise ConfigError(f"problem.methods[{index}] must be an object or string")
            try:
                get_method_spec(str(method_id))
            except MethodRegistryError as exc:
                raise ConfigError(str(exc)) from exc


def _validate_paths(
    config: Mapping[str, object], start_stage: str, data_root: Path
) -> dict[str, Path]:
    sources = _mapping(config.get("sources"), "sources")
    paths_config = _mapping(config.get("paths"), "paths")
    required_outputs = {
        "run_directory",
        "prepared_ligand_manifest",
        "score_tables",
        "matrices",
        "problem",
        "selection",
        "evaluation",
    }
    selection = _mapping(config.get("selection"), "selection")
    receptor_selection = selection.get("receptor_selection")
    manual_receptors = (
        isinstance(receptor_selection, dict)
        and str(receptor_selection.get("mode", "")) == "manual"
    )
    if not manual_receptors:
        required_outputs.add("selected_receptor_manifest")
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
        method = str(box.get("method", "fixed_snapshot"))
        if method == "ligand_bounds":
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
        elif method == "fixed_snapshot":
            required_box = (
                "center_x", "center_y", "center_z", "size_x", "size_y", "size_z"
            )
            try:
                values = [float(box[key]) for key in required_box]
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigError(
                    "fixed_snapshot docking.box requires six numeric center/size values"
                ) from exc
            if any(value <= 0 for value in values[3:]):
                raise ConfigError("fixed_snapshot docking.box sizes must be positive")
        else:
            raise ConfigError(
                "raw full workflow requires docking.box.method=ligand_bounds or fixed_snapshot"
            )
    selection = _mapping(config.get("selection"), "selection")
    receptor_selection = selection.get("receptor_selection")
    if (
        isinstance(receptor_selection, dict)
        and str(receptor_selection.get("mode", "")) == "manual"
        and raw_receptor_sources.intersection(sources)
        and any(
            not isinstance(item, dict) or not item.get("rcsb_id")
            for item in receptor_selection.get("receptors", [])
        )
    ):
        raise ConfigError(
            "raw manual receptor selection requires rcsb_id for every receptor"
        )
    _validate_problem(config)
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
    problem = normalized.setdefault("problem", {})
    problem.setdefault("utility_metric", "bedroc")
    problem.setdefault("bedroc_alpha", 20.0)
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
