"""Configuration contracts for the canonical experiment pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .io import file_sha256, read_json


CANONICAL_SCHEMA_VERSION = "2.0"
PIPELINE_STAGES = (
    "prepare",
    "build_problem",
    "solve",
    "evaluate",
    "persist",
)
_SHA256_RE = re.compile(r"^[0-9A-Fa-f]{64}$")


class ConfigError(ValueError):
    """Raised when a canonical pipeline config violates its contract."""


@dataclass(frozen=True)
class InputArtifact:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class ResolvedPipelineConfig:
    path: Path
    root: Path
    data: dict[str, object]
    inputs: dict[str, InputArtifact]
    run_directory: Path


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    return value


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def validate_pipeline_config(config: dict[str, object]) -> None:
    """Validate the stable, target-independent pipeline contract."""
    if config.get("schema_version") != CANONICAL_SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version must be {CANONICAL_SCHEMA_VERSION}"
        )
    for key in ("experiment_id", "target_id", "purpose"):
        _require_nonempty_string(config.get(key), key)

    stages = config.get("pipeline")
    if not isinstance(stages, list) or not stages:
        raise ConfigError("pipeline must be a non-empty list")
    normalized_stages = [str(stage) for stage in stages]
    if normalized_stages != [stage for stage in PIPELINE_STAGES if stage in normalized_stages]:
        raise ConfigError("pipeline stages must preserve canonical order")
    unknown = set(normalized_stages).difference(PIPELINE_STAGES)
    if unknown:
        raise ConfigError(f"unknown pipeline stages: {sorted(unknown)}")

    inputs = _require_mapping(config.get("inputs"), "inputs")
    if not inputs:
        raise ConfigError("inputs must not be empty")
    for name, record in inputs.items():
        item = _require_mapping(record, f"inputs.{name}")
        _require_nonempty_string(item.get("path"), f"inputs.{name}.path")
        sha256 = _require_nonempty_string(
            item.get("sha256"), f"inputs.{name}.sha256"
        )
        if not _SHA256_RE.fullmatch(sha256):
            raise ConfigError(f"inputs.{name}.sha256 must be a SHA-256 digest")

    policy = _require_mapping(config.get("data_policy"), "data_policy")
    allowed_splits = policy.get("allowed_splits")
    locked_splits = policy.get("locked_splits")
    if not isinstance(allowed_splits, list) or not all(
        isinstance(value, str) and value for value in allowed_splits
    ):
        raise ConfigError("data_policy.allowed_splits must be a non-empty list")
    if "train" not in allowed_splits:
        raise ConfigError("data_policy.allowed_splits must include train")
    if not isinstance(locked_splits, list) or not all(
        isinstance(value, str) and value for value in locked_splits
    ):
        raise ConfigError("data_policy.locked_splits must be a non-empty list")
    if policy.get("evaluate_locked_test") is not False:
        raise ConfigError("data_policy.evaluate_locked_test must be false")

    prepare = _require_mapping(config.get("prepare"), "prepare")
    _require_nonempty_string(
        prepare.get("adapter", "existing_matrix"), "prepare.adapter"
    )

    problem = _require_mapping(config.get("problem"), "problem")
    _require_nonempty_string(problem.get("type"), "problem.type")
    _require_nonempty_string(problem.get("strategy"), "problem.strategy")
    receptor_ids = problem.get("receptor_ids")
    if not isinstance(receptor_ids, list) or not receptor_ids:
        raise ConfigError("problem.receptor_ids must be a non-empty list")
    receptor_names = [str(value) for value in receptor_ids]
    if len(receptor_names) != len(set(receptor_names)):
        raise ConfigError("problem.receptor_ids must be unique")
    if "target_size" in problem:
        target_size = problem["target_size"]
        if isinstance(target_size, bool) or not isinstance(target_size, int):
            raise ConfigError("problem.target_size must be an integer")
        if not 0 <= target_size <= len(receptor_names):
            raise ConfigError("problem.target_size must be within the receptor pool")
    k_policy = problem.get("k_policy")
    if k_policy is not None:
        policy = _require_mapping(k_policy, "problem.k_policy")
        mode = str(policy.get("mode", "adaptive"))
        if mode not in {"fixed", "adaptive"}:
            raise ConfigError("problem.k_policy.mode must be fixed or adaptive")
        if mode == "adaptive":
            candidates = policy.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ConfigError(
                    "problem.k_policy.candidates must be a non-empty list"
                )
            normalized_candidates: list[int] = []
            for candidate in candidates:
                if isinstance(candidate, bool) or not isinstance(candidate, int):
                    raise ConfigError(
                        "problem.k_policy.candidates must contain integers"
                    )
                if not 0 <= candidate <= len(receptor_names):
                    raise ConfigError(
                        "problem.k_policy.candidates must fit the receptor pool"
                    )
                normalized_candidates.append(candidate)
            if len(normalized_candidates) != len(set(normalized_candidates)):
                raise ConfigError("problem.k_policy.candidates must be unique")
        elif "value" in policy:
            value = policy["value"]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError("problem.k_policy.value must be an integer")
            if not 0 <= value <= len(receptor_names):
                raise ConfigError("problem.k_policy.value must fit the receptor pool")
        elif "target_size" not in problem:
            raise ConfigError(
                "fixed problem.k_policy requires value or problem.target_size"
            )
        if "selector" in policy:
            _require_nonempty_string(policy["selector"], "problem.k_policy.selector")
        if "selection_split" in policy:
            _require_nonempty_string(
                policy["selection_split"], "problem.k_policy.selection_split"
            )
        if "selection_metric" in policy:
            _require_nonempty_string(
                policy["selection_metric"], "problem.k_policy.selection_metric"
            )

    solve = _require_mapping(config.get("solve"), "solve")
    backend = _require_nonempty_string(solve.get("backend"), "solve.backend")
    if backend not in {"dry_run", "exact", "greedy"}:
        raise ConfigError("solve.backend must be dry_run, exact, or greedy")

    evaluate = _require_mapping(config.get("evaluate"), "evaluate")
    metrics = evaluate.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ConfigError("evaluate.metrics must be a non-empty list")

    outputs = _require_mapping(config.get("outputs"), "outputs")
    _require_nonempty_string(outputs.get("run_directory"), "outputs.run_directory")


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_pipeline_config(
    config_path: str | Path, root: str | Path | None = None
) -> ResolvedPipelineConfig:
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    config = read_json(path)
    validate_pipeline_config(config)
    root_path = resolve_path(root or path.parent, Path.cwd())

    inputs: dict[str, InputArtifact] = {}
    for name, record in config["inputs"].items():
        assert isinstance(record, dict)
        inputs[str(name)] = InputArtifact(
            name=str(name),
            path=resolve_path(str(record["path"]), root_path),
            sha256=str(record["sha256"]).upper(),
        )
    outputs = config["outputs"]
    assert isinstance(outputs, dict)
    return ResolvedPipelineConfig(
        path=path,
        root=root_path,
        data=config,
        inputs=inputs,
        run_directory=resolve_path(str(outputs["run_directory"]), root_path),
    )


def verify_input_artifacts(config: ResolvedPipelineConfig) -> dict[str, dict[str, object]]:
    """Verify all declared inputs before any pipeline stage runs."""
    records: dict[str, dict[str, object]] = {}
    for name, artifact in config.inputs.items():
        if not artifact.path.is_file():
            raise FileNotFoundError(artifact.path)
        actual = file_sha256(artifact.path)
        if actual != artifact.sha256:
            raise ConfigError(
                f"input SHA-256 differs for {name}: {actual} != {artifact.sha256}"
            )
        records[name] = {
            "path": artifact.path.as_posix(),
            "sha256": actual,
            "size_bytes": artifact.path.stat().st_size,
        }
    return records
