import json
from pathlib import Path

import pytest

from qubo_receptor_ensemble.config import (
    ConfigError,
    load_pipeline_config,
)
from qubo_receptor_ensemble.cli import main as pipeline_cli
from qubo_receptor_ensemble.io import file_sha256
from qubo_receptor_ensemble.matrix import load_config as load_aggregation_config
from qubo_receptor_ensemble.pipeline import PipelineRunner, PipelineStateError
from qubo_receptor_ensemble.qubo import build_qubo, objective


def test_matrix_aggregation_config_loader_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "aggregation.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment_id": "aggregation-test",
                "purpose": "test",
                "ligand_manifest": {},
                "seed_runs": [],
                "expected": {},
                "aggregation": {},
                "outputs": {},
                "interpretation_boundary": "test only",
            }
        ),
        encoding="ascii",
    )

    assert load_aggregation_config(path)["experiment_id"] == "aggregation-test"


def test_build_qubo_exposes_metric_specific_utility_values() -> None:
    rows = [
        {"ligand_id": "A1", "label": "active", "R1": "-10", "R2": "-9"},
        {"ligand_id": "A2", "label": "active", "R1": "-9", "R2": "-8"},
        {"ligand_id": "D1", "label": "decoy", "R1": "-5", "R2": "-6"},
        {"ligand_id": "D2", "label": "decoy", "R1": "-4", "R2": "-5"},
    ]

    qubo = build_qubo(
        rows,
        ["R1", "R2"],
        target_size=1,
        redundancy_weight=0.0,
        count_weight=0.0,
        size_weight=1.0,
        utility_metric="bedroc",
    )

    assert qubo["utility_metric"] == "bedroc"
    assert qubo["utilities_train"] == qubo["utilities_train_bedroc"]


def test_build_qubo_defaults_to_bedroc20() -> None:
    rows = [
        {"ligand_id": "A1", "label": "active", "R1": "-10", "R2": "-9"},
        {"ligand_id": "A2", "label": "active", "R1": "-9", "R2": "-8"},
        {"ligand_id": "D1", "label": "decoy", "R1": "-5", "R2": "-6"},
        {"ligand_id": "D2", "label": "decoy", "R1": "-4", "R2": "-5"},
    ]

    qubo = build_qubo(
        rows,
        ["R1", "R2"],
        target_size=1,
        redundancy_weight=0.0,
        count_weight=0.0,
        size_weight=1.0,
    )

    assert qubo["utility_metric"] == "bedroc"
    assert qubo["bedroc_alpha"] == 20.0
    assert "utilities_train_bedroc_alpha_20" in qubo


def test_build_qubo_propagates_custom_bedroc_alpha() -> None:
    rows = [
        {"ligand_id": "A", "label": "active", "R1": "-10", "R2": "-9"},
        {"ligand_id": "D", "label": "decoy", "R1": "-5", "R2": "-4"},
    ]

    qubo = build_qubo(
        rows,
        ["R1", "R2"],
        target_size=1,
        redundancy_weight=0.0,
        count_weight=0.0,
        size_weight=1.0,
        utility_metric="bedroc",
        bedroc_alpha=12.0,
    )

    assert qubo["bedroc_alpha"] == 12.0
    assert "utilities_train_bedroc_alpha_12" in qubo


def test_qubo_objective_accepts_new_utility_field_without_legacy_alias() -> None:
    rows = [
        {"ligand_id": "A", "label": "active", "R1": "-10", "R2": "-9"},
        {"ligand_id": "D", "label": "decoy", "R1": "-5", "R2": "-4"},
    ]
    qubo = build_qubo(
        rows,
        ["R1", "R2"],
        target_size=1,
        redundancy_weight=0.0,
        count_weight=0.0,
        size_weight=1.0,
        utility_metric="roc_auc",
    )
    del qubo["utilities_train_roc_auc"]

    assert objective(("R1",), qubo) == pytest.approx(
        -float(qubo["utilities_train"]["R1"])
    )


def _write_pipeline_config(tmp_path: Path) -> Path:
    source = tmp_path / "data" / "scores.csv"
    source.parent.mkdir(parents=True)
    source.write_text("ligand_id,label,R1\nA,active,-8\n", encoding="utf-8")
    config_path = tmp_path / "configs" / "pipeline.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "experiment_id": "pipeline-test",
                "target_id": "TEST",
                "purpose": "pipeline foundation test",
                "pipeline": [
                    "prepare",
                    "build_problem",
                    "solve",
                    "evaluate",
                    "persist",
                ],
                "inputs": {
                    "score_source": {
                        "path": "data/scores.csv",
                        "sha256": file_sha256(source),
                    }
                },
                "data_policy": {
                    "allowed_splits": ["train"],
                    "locked_splits": ["test"],
                    "evaluate_locked_test": False,
                },
                "prepare": {"adapter": "existing_matrix"},
                "problem": {
                    "type": "receptor_subset",
                    "strategy": "normalized_qubo",
                    "receptor_ids": ["R1"],
                },
                "solve": {"backend": "dry_run"},
                "evaluate": {"metrics": ["roc_auc"]},
                "outputs": {"run_directory": "results/pipeline-test"},
            }
        ),
        encoding="ascii",
    )
    return config_path


def test_pipeline_config_resolves_paths_from_explicit_root(tmp_path: Path) -> None:
    config_path = _write_pipeline_config(tmp_path)

    config = load_pipeline_config(config_path, root=tmp_path)

    assert config.root == tmp_path.resolve()
    assert config.inputs["score_source"].path == (
        tmp_path / "data" / "scores.csv"
    ).resolve()
    assert config.run_directory == (tmp_path / "results" / "pipeline-test").resolve()


def test_pipeline_config_rejects_locked_test_evaluation(tmp_path: Path) -> None:
    config_path = _write_pipeline_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="ascii"))
    payload["data_policy"]["evaluate_locked_test"] = True
    config_path.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(ConfigError, match="evaluate_locked_test"):
        load_pipeline_config(config_path, root=tmp_path)


def test_pipeline_config_requires_prepare_section(tmp_path: Path) -> None:
    config_path = _write_pipeline_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="ascii"))
    del payload["prepare"]
    config_path.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(ConfigError, match="prepare"):
        load_pipeline_config(config_path, root=tmp_path)


def test_pipeline_dry_run_writes_stage_manifests(tmp_path: Path) -> None:
    config_path = _write_pipeline_config(tmp_path)
    config = load_pipeline_config(config_path, root=tmp_path)

    summary = PipelineRunner(config).run(dry_run=True)

    assert summary["status"] == "planned"
    manifest_path = config.run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    assert manifest["experiment_id"] == "pipeline-test"
    assert manifest["stages"]["prepare"]["status"] == "planned"
    assert (config.run_directory / "stages" / "05_persist" / "summary.json").is_file()


def test_pipeline_requires_target_size_for_real_normalized_runs(
    tmp_path: Path,
) -> None:
    config_path = _write_pipeline_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="ascii"))
    payload["solve"]["backend"] = "exact"
    config_path.write_text(json.dumps(payload), encoding="ascii")

    config = load_pipeline_config(config_path, root=tmp_path)
    with pytest.raises(PipelineStateError, match="candidate k values"):
        PipelineRunner(config).run()


def test_pipeline_partial_run_does_not_create_persist_stage(
    tmp_path: Path,
) -> None:
    config_path = _write_pipeline_config(tmp_path)
    config = load_pipeline_config(config_path, root=tmp_path)

    PipelineRunner(config).run(dry_run=True, end_stage="solve")

    assert not (config.run_directory / "stages" / "05_persist").exists()


def test_pipeline_cli_plan_returns_json_summary(tmp_path: Path, capsys) -> None:
    config_path = _write_pipeline_config(tmp_path)

    assert pipeline_cli(
        ["plan", "--config", str(config_path), "--root", str(tmp_path)]
    ) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "planned"
    assert output["experiment_id"] == "pipeline-test"
