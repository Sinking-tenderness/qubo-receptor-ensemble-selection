import hashlib
import json

from scripts.aggregate_vina_seed_replicates import (
    aggregate_replicates,
    build_wide_matrix,
    load_source_run,
)


def source(replicate_id, base_seed, scores):
    rows = []
    for ligand_id, label, score in scores:
        rows.append(
            {
                "target_id": "CDK2",
                "ligand_id": ligand_id,
                "label": label,
                "receptor_id": "R1",
                "representative_score": str(score),
                "status": "ok",
            }
        )
    return {
        "replicate_id": replicate_id,
        "base_seed": base_seed,
        "rows": rows,
    }


def aggregation_settings():
    return {
        "primary_method": "minimum_score",
        "sensitivity_method": "median_score",
        "maximum_seed_range_kcal_per_mol": 1.0,
        "maximum_minimum_median_delta_kcal_per_mol": 1.0,
        "minimum_favorable_replicates": 2,
        "flag_nonnegative_minimum_score": True,
    }


def test_aggregate_replicates_keeps_minimum_median_and_stability_warning():
    sources = [
        source("seed0", 10, [("A", "active", 5.0)]),
        source("seed1", 20, [("A", "active", -8.0)]),
        source("seed2", 30, [("A", "active", -7.9)]),
    ]
    rows = aggregate_replicates(sources, [("A", "R1")], aggregation_settings())
    assert len(rows) == 1
    row = rows[0]
    assert row["minimum_score"] == -8.0
    assert row["median_score"] == -7.9
    assert row["favorable_replicate_count"] == 2
    assert row["best_replicate_id"] == "seed1"
    assert row["seed_stability_warning"] is True
    assert row["seed_stability_warning_reasons"] == "seed_range_exceeded"


def test_aggregate_replicates_flags_single_favorable_outlier():
    sources = [
        source("seed0", 10, [("A", "active", 1.0)]),
        source("seed1", 20, [("A", "active", -8.0)]),
        source("seed2", 30, [("A", "active", 2.0)]),
    ]
    row = aggregate_replicates(
        sources, [("A", "R1")], aggregation_settings()
    )[0]
    assert row["minimum_score"] == -8.0
    assert row["median_score"] == 1.0
    assert row["favorable_replicate_count"] == 1
    assert row["seed_stability_warning_reasons"] == (
        "seed_range_exceeded;minimum_median_delta_exceeded;"
        "insufficient_favorable_replicates"
    )


def test_aggregate_replicates_can_designate_median_as_primary():
    settings = aggregation_settings()
    settings["primary_method"] = "median_score"
    settings["sensitivity_method"] = "minimum_score"
    sources = [
        source("seed0", 10, [("A", "active", 5.0)]),
        source("seed1", 20, [("A", "active", -8.0)]),
        source("seed2", 30, [("A", "active", -7.9)]),
    ]
    row = aggregate_replicates(sources, [("A", "R1")], settings)[0]
    assert row["minimum_score"] == -8.0
    assert row["median_score"] == -7.9
    assert row["primary_score"] == -7.9
    assert row["primary_method"] == "median_score"


def test_build_wide_matrix_uses_requested_aggregate_score():
    rows = [
        {
            "target_id": "CDK2",
            "ligand_id": "A",
            "label": "active",
            "receptor_id": "R1",
            "minimum_score": -8.0,
            "median_score": -7.9,
        },
        {
            "target_id": "CDK2",
            "ligand_id": "A",
            "label": "active",
            "receptor_id": "R2",
            "minimum_score": -7.0,
            "median_score": -6.9,
        },
    ]
    assert build_wide_matrix(rows, "median_score") == [
        {
            "target_id": "CDK2",
            "ligand_id": "A",
            "label": "active",
            "R1": -7.9,
            "R2": -6.9,
        }
    ]


def test_load_source_run_verifies_summary_and_returns_serializable_paths(tmp_path):
    representative = tmp_path / "representative.csv"
    representative.write_text(
        "target_id,ligand_id,label,receptor_id,representative_score,status\n"
        "CDK2,A,active,R1,-8.0,ok\n",
        encoding="utf-8",
    )
    representative_hash = hashlib.sha256(representative.read_bytes()).hexdigest().upper()
    summary_path = tmp_path / "summary.json"
    run_config_path = tmp_path / "run.json"
    run_config = {
        "experiment_id": "run-one",
        "docking": {"base_seed": 10, "representative_method": "pose_rank_1"},
        "outputs": {
            "summary_json": str(summary_path),
            "representative_long_csv": str(representative),
        },
    }
    run_config_path.write_text(json.dumps(run_config), encoding="ascii")
    config_hash = hashlib.sha256(run_config_path.read_bytes()).hexdigest().upper()
    summary = {
        "experiment_id": "run-one",
        "status": "ok",
        "successful_receptor_ligand_pairs": 1,
        "config": {"sha256": config_hash},
        "inputs": {"ligand_manifest": {"sha256": "ABC"}},
        "outputs": {
            "representative_long_csv": {"sha256": representative_hash}
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="ascii")
    loaded = load_source_run(
        {
            "replicate_id": "seed0",
            "config_path": str(run_config_path),
            "config_sha256": config_hash,
            "expected_experiment_id": "run-one",
            "expected_base_seed": 10,
        },
        expected_pairs=1,
    )
    assert isinstance(loaded["config_path"], str)
    json.dumps({key: value for key, value in loaded.items() if key != "rows"})
