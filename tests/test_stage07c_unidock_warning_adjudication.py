import json
from pathlib import Path

from scripts.experimental.unidock.build_stage07c_warning_adjudication_bundle import (
    CONFIG,
    FIXED_PATHS,
)
from scripts.experimental.unidock.evaluate_stage07c_unidock_warning_adjudication import (
    combined_groups,
    group_metric_rows,
    replay_comparison_rows,
    seed_stability_rows,
)
from scripts.experimental.unidock.run_stage07c_unidock_warning_adjudication import (
    classify_warning_log,
    validate_config,
    validate_inputs,
)
from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
    file_sha256,
    read_csv,
    read_json,
)


CONFIG_PATH = Path(
    "configs/stage07c_mk14_unidock113_warning_adjudication.json"
)


def test_stage07c_config_and_actual_inputs_are_train_only():
    config = read_json(CONFIG_PATH)

    validate_config(config)
    receptors, ligands, prior, replay, audit = validate_inputs(
        Path.cwd(), config
    )

    assert len(receptors) == 4
    assert len(ligands) == 160
    assert len(prior) == 1920
    assert len(replay) == 160
    assert audit["new_seed_pair_count"] == 640
    assert audit["warning_replay_pair_count"] == 160
    assert audit["expected_pair_count"] == 800
    assert audit["validation_rows"] == audit["test_rows"] == 0


def test_stage07c_frozen_evidence_provenance_matches_files():
    provenance = read_json(
        Path("data/stage07c_mk14_unidock113_existing_evidence_provenance.json")
    )

    for descriptor in provenance["outputs"].values():
        path = Path(descriptor["path"])
        assert file_sha256(path) == descriptor["sha256"]
        assert len(read_csv(path)) == int(descriptor["rows"])
    assert provenance["fresh_validation_rows_read"] == 0
    assert provenance["test_rows_read"] == 0


def test_known_coordinate_warning_is_resolved_only_with_safe_poses(tmp_path):
    log = tmp_path / "unidock.log"
    log.write_text(
        "WARNING: in add_to_output_container, adding the 1th ligand\n"
        "t.coords.size()=26, out[0].coords.size()=27\n",
        encoding="ascii",
    )

    resolved = classify_warning_log(log, {"failure_count": 0})
    failed_pose = classify_warning_log(log, {"failure_count": 1})

    assert resolved["known_warning_event_count"] == 1
    assert resolved["unresolved_warning_event_count"] == 0
    assert resolved["status"] == "resolved"
    assert failed_pose["unresolved_warning_event_count"] == 1
    assert failed_pose["status"] == "unresolved"


def test_unknown_warning_remains_unresolved(tmp_path):
    log = tmp_path / "unidock.log"
    log.write_text("WARNING: unexpected engine condition\n", encoding="ascii")

    result = classify_warning_log(log, {"failure_count": 0})

    assert result["known_warning_event_count"] == 0
    assert result["unresolved_warning_event_count"] == 1


def test_replay_reference_requires_exact_scores_and_pose_hashes():
    reference = read_csv(
        Path(
            "data/processed/"
            "stage07c_mk14_unidock113_enhanced_warning_replay_reference.csv"
        )
    )
    current = [
        {
            **row,
            "run_role": "warning_replay",
            "run_id": "seed2_replay",
        }
        for row in reference
    ]

    comparisons = replay_comparison_rows(current, reference)

    assert len(comparisons) == 160
    assert all(row["score_exact_match"] for row in comparisons)
    assert all(row["pose_sha256_exact_match"] for row in comparisons)


def test_four_seed_stability_includes_new_seed_pairs():
    prior = read_csv(
        Path(
            "data/processed/"
            "stage07c_mk14_unidock113_enhanced_seed012_scores.csv"
        )
    )
    new_rows = [
        {
            **row,
            "run_role": "new_seed",
            "run_id": "seed3",
            "seed_id": "seed3",
            "base_seed": "20260804",
        }
        for row in prior
        if row["seed_id"] == "seed2"
    ]

    groups = combined_groups(prior, new_rows)
    _, metrics = group_metric_rows(groups)
    stability = seed_stability_rows(groups, metrics)

    assert len(groups) == 16
    assert len(stability) == 24
    assert sum(row["includes_new_seed"] for row in stability) == 12


def test_stage07c_config_hashes_match_implementations():
    config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))

    for descriptor in config["implementation"].values():
        assert file_sha256(Path(descriptor["path"])) == descriptor["sha256"]


def test_stage07c_bundle_contains_every_runtime_dependency():
    assert CONFIG in FIXED_PATHS
    assert any("run_stage07b_unidock" in path for path in FIXED_PATHS)
    assert any("prepare_stage07c_existing_evidence.py" in path for path in FIXED_PATHS)
    assert any("run_stage07c_unidock" in path for path in FIXED_PATHS)
    assert any("evaluate_stage07c_unidock" in path for path in FIXED_PATHS)
    assert any("build_stage07c_pose_diagnostics.py" in path for path in FIXED_PATHS)
    assert any("warning_adjudication_execution.md" in path for path in FIXED_PATHS)
