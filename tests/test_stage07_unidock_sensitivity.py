import json
from pathlib import Path

import pytest

from scripts.experimental.unidock.build_stage07_unidock_sensitivity_bundle import (
    CONFIG,
    FIXED_PATHS,
)
from scripts.experimental.unidock.evaluate_stage07_unidock_sensitivity import (
    finite_spearman,
    profile_summary_rows,
    top_fraction_overlap,
)
from scripts.experimental.unidock.run_stage07_unidock_sensitivity import (
    PROFILE_ORDER,
    merged_protocol,
    validate_config,
    validate_inputs,
)
from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
    file_sha256,
    read_json,
)


CONFIG_PATH = Path(
    "configs/stage07_mk14_unidock113_train160_search_sensitivity.json"
)


def test_stage07_config_uses_official_profile_ladder():
    config = read_json(CONFIG_PATH)

    validate_config(config)

    observed = {
        profile_id: (
            merged_protocol(config, profile_id)["exhaustiveness"],
            merged_protocol(config, profile_id)["max_step"],
        )
        for profile_id in PROFILE_ORDER
    }
    assert observed == {
        "fast": (128, 20),
        "balance": (384, 40),
        "detail": (512, 40),
    }


def test_stage07_actual_inputs_are_train_only_and_macrocycle_safe():
    config = read_json(CONFIG_PATH)

    receptors, ligands, audit = validate_inputs(Path.cwd(), config)

    assert len(receptors) == 4
    assert len(ligands) == 160
    assert audit["macrocycle_closure_pseudoatom_ligand_count"] == 0
    assert audit["preparation_variant_counts"] == {
        "meeko_rigid_macrocycles": 4,
        "original_meeko_flexible": 156,
    }
    assert audit["validation_rows"] == audit["test_rows"] == 0


def test_rank_comparison_helpers_use_lower_scores_as_better():
    ligand_ids = ["A", "B", "C", "D"]
    first = {"A": -9.0, "B": -8.0, "C": -7.0, "D": -6.0}
    second = {"A": -9.1, "B": -8.1, "C": -6.0, "D": -7.0}

    assert finite_spearman(
        [first[value] for value in ligand_ids],
        [second[value] for value in ligand_ids],
    ) == pytest.approx(0.8)
    assert top_fraction_overlap(ligand_ids, first, second, 0.25) == 1.0


def test_profile_summary_applies_every_gate_check():
    config = read_json(CONFIG_PATH)
    comparisons = []
    stability = []
    batches = []
    for profile_id, rho, overlap, bedroc_delta in (
        ("fast", 0.94, 0.875, 0.01),
        ("balance", 0.97, 0.875, 0.01),
        ("detail", 1.0, 1.0, 0.0),
    ):
        comparisons.append(
            {
                "profile_id": profile_id,
                "spearman_vs_detail": rho,
                "top5pct_overlap_vs_detail": overlap,
                "bedroc_delta_vs_detail": bedroc_delta,
            }
        )
        stability.append(
            {
                "profile_id": profile_id,
                "spearman": 0.96,
                "absolute_bedroc_delta": 0.01,
            }
        )
        for index in range(12):
            batches.append(
                {
                    "profile_id": profile_id,
                    "elapsed_seconds": str(10 + index),
                    "engine_warning_count": "0",
                }
            )

    summaries = profile_summary_rows(
        config, comparisons, stability, batches
    )

    assert not summaries[0]["all_gate_checks_passed"]
    assert summaries[1]["all_gate_checks_passed"]
    assert summaries[2]["all_gate_checks_passed"]


def test_stage07_config_hashes_match_implementations():
    config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))

    for descriptor in config["implementation"].values():
        assert file_sha256(Path(descriptor["path"])) == descriptor["sha256"]


def test_stage07_bundle_includes_runner_evaluator_and_execution_guide():
    assert CONFIG in FIXED_PATHS
    assert any("run_stage07_unidock_sensitivity.py" in path for path in FIXED_PATHS)
    assert any(
        "evaluate_stage07_unidock_sensitivity.py" in path for path in FIXED_PATHS
    )
    assert any("sensitivity_execution.md" in path for path in FIXED_PATHS)
