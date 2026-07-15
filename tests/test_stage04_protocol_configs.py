from pathlib import Path

from scripts.aggregate_vina_seed_replicates import load_config as load_aggregate_config
from scripts.align_receptor_structure import file_sha256
from scripts.run_md_receptor_ligand_benchmark import load_config as load_benchmark_config


SEED_CONFIGS = sorted(
    Path("configs").glob("stage04_cdk2_non_md_80a80d_e32_seed*_benchmark_linux.json")
)
AGGREGATE_CONFIG = Path(
    "configs/stage04_cdk2_non_md_80a80d_e32_multiseed_aggregate_linux.json"
)


def test_stage04_seed_configs_are_paired_development_only_protocols():
    assert len(SEED_CONFIGS) == 3
    configs = [load_benchmark_config(path) for path in SEED_CONFIGS]

    assert [config["docking"]["base_seed"] for config in configs] == [
        20260901,
        20360901,
        20460901,
    ]
    assert all(config["expected_receptor_count"] == 8 for config in configs)
    assert all(config["expected_ligand_count"] == 160 for config in configs)
    assert all(config["expected_label_counts"] == {"active": 80, "decoy": 80} for config in configs)
    assert len({config["input_sha256"]["receptor_manifest"] for config in configs}) == 1
    assert len({config["input_sha256"]["ligand_manifest"] for config in configs}) == 1
    assert all(
        config["inputs"]["ligand_manifest"].endswith(
            "stage04_cdk2_development_80a80d_pdbqt_manifest.csv"
        )
        for config in configs
    )


def test_stage04_aggregate_config_preserves_three_seed_protocol():
    config = load_aggregate_config(AGGREGATE_CONFIG)

    assert config["expected_receptor_count"] == 8
    assert config["expected_ligand_count"] == 160
    assert config["expected_label_counts"] == {"active": 80, "decoy": 80}
    assert [source["expected_base_seed"] for source in config["source_runs"]] == [
        20260901,
        20360901,
        20460901,
    ]
    assert config["aggregation"]["primary_method"] == "median_score"
    assert config["aggregation"]["sensitivity_method"] == "minimum_score"
    assert all(
        file_sha256(Path(source["config_path"])) == source["config_sha256"]
        for source in config["source_runs"]
    )
