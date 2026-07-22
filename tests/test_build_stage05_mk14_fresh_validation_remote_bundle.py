import json
from pathlib import Path

from scripts.build_stage05_mk14_fresh_validation_remote_bundle import (
    EXECUTION_PROFILES,
    FIXED_PATHS,
)
from scripts.prepare_receptor import file_sha256


SEED1_CONFIG = Path(
    "configs/stage05_mk14_fresh_validation_e32_seed1_64vcpu_linux.json"
)
AGGREGATE_CONFIG = Path(
    "configs/stage05_mk14_fresh_validation_e32_distributed_seed_aggregation.json"
)
REMOTE_SCRIPT = Path(
    "scripts/run_stage05_mk14_fresh_validation_seed1_64vcpu_remote.sh"
)


def test_distributed_seed1_profile_uses_all_64_vcpus():
    profile = EXECUTION_PROFILES["distributed-seed1-64vcpu"]
    config = json.loads(SEED1_CONFIG.read_text(encoding="ascii"))

    assert profile["workers"] == 32
    assert profile["max_total_cpu"] == 64
    assert config["docking"] == {
        "workers": 32,
        "max_total_cpu": 64,
        "base_seed": 20260802,
        "representative_method": "pose_rank_1",
    }
    assert config["input_sha256"]["parallel_runner"] == file_sha256(
        Path(config["inputs"]["parallel_runner"])
    )


def test_distributed_aggregation_pins_mixed_execution_configs():
    config = json.loads(AGGREGATE_CONFIG.read_text(encoding="ascii"))
    runs = {row["seed_id"]: row for row in config["seed_runs"]}

    assert runs["seed1"]["config_path"] == SEED1_CONFIG.as_posix()
    for run in runs.values():
        assert run["config_sha256"] == file_sha256(Path(run["config_path"]))
    assert config["expected"]["receptor_ligand_pairs_per_seed"] == 7880


def test_seed1_remote_script_and_configs_are_bundled():
    source = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert "stage05_mk14_fresh_validation_e32_seed1_64vcpu_linux.json" in source
    assert "32_workers_x_2_vina_cpu" in source
    assert "aggregate_seed_replicates.py" not in source
    assert "evaluate_stage05_mk14_fresh_validation.py" not in source
    for path in (SEED1_CONFIG, AGGREGATE_CONFIG, REMOTE_SCRIPT):
        assert path.as_posix() in FIXED_PATHS
