import json
from pathlib import Path

import pytest

from scripts.experimental.vinagpu.run_vinagpu_deterministic_batch import (
    batch_command,
)
from scripts.experimental.vinagpu.run_vinagpu_search_depth_diagnostic import (
    assess_profile,
    heuristic_depth,
    runtime_projection,
    validate_approved_runtime,
    validate_diagnostic_inputs,
)


CONFIG_PATH = Path("configs/stage06_mk14_vinagpu21_search_depth_diagnostic.json")


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="ascii"))


def test_diagnostic_inputs_are_bounded_and_bridge_authorized():
    config = load_config()

    receptors, ligands, reference, cpu_scores, audit = validate_diagnostic_inputs(
        Path.cwd().resolve(), config
    )

    assert len(receptors) == 5
    assert len(ligands) == 160
    assert len(reference) == 2400
    assert len(cpu_scores) == 320
    assert audit["bridge_status"] == "deterministic_batch_bridge_passed"
    assert audit["profile_depths"] == [16, 24, 32]
    assert audit["diagnostic_pair_count_per_profile"] == 320
    assert audit["diagnostic_chunk_count_per_profile"] == 40
    assert audit["validation_rows"] == 0
    assert audit["test_rows"] == 0
    assert audit["labels_used_for_profile_selection"] is False


def test_first_fixed_depth_covers_every_heuristic_ligand_depth():
    config = load_config()
    _, ligands, _, _, _ = validate_diagnostic_inputs(Path.cwd().resolve(), config)
    depths = sorted(heuristic_depth(row) for row in ligands)

    assert depths[0] == 2
    assert depths[len(depths) // 2] == 9
    assert depths[-1] == 15
    assert config["search_depth_diagnostic"]["profiles"][0]["search_depth"] == 16


def test_batch_command_includes_the_fixed_profile_depth(tmp_path: Path):
    config = load_config()
    protocol = {**config["vinagpu"], "search_depth": 16}

    command = batch_command(
        tmp_path / "vina-gpu",
        tmp_path / "kernels",
        tmp_path / "receptor.pdbqt",
        tmp_path / "inputs",
        tmp_path / "outputs",
        protocol,
        20260802,
    )

    assert command[command.index("--search_depth") + 1] == "16"
    assert command[command.index("--seed") + 1] == "20260802"


def synthetic_rows(reverse_second_group: bool = False):
    rows = []
    for receptor_index, receptor_id in enumerate(("R1", "R2")):
        for index in range(160):
            cpu = float(index)
            gpu = (
                float(159 - index)
                if reverse_second_group and receptor_index == 1
                else cpu
            )
            rows.append(
                {
                    "seed_id": "seed1",
                    "receptor_id": receptor_id,
                    "ligand_id": f"L{index:03d}",
                    "cpu_vina_e32_score": cpu,
                    "gpu_vinagpu21_score": gpu,
                    "score_delta_gpu_minus_cpu": gpu - cpu,
                }
            )
    return rows


def frozen_groups():
    return {
        ("seed1", receptor_id): {
            "spearman": 0.93,
            "top5pct_overlap": 0.875,
        }
        for receptor_id in ("R1", "R2")
    }


def test_profile_gate_selects_exact_fast_cpu_equivalence():
    summary, groups = assess_profile(
        load_config(),
        {"profile_id": "fixed_depth_16", "search_depth": 16},
        synthetic_rows(),
        400.0,
        frozen_groups(),
    )

    assert len(groups) == 2
    assert summary["status"] == "search_depth_candidate_selected"
    assert summary["all_gate_checks_passed"] is True
    assert summary["continue_to_next_profile"] is False


def test_profile_gate_continues_only_when_accuracy_fails_but_speed_passes():
    inaccurate, _ = assess_profile(
        load_config(),
        {"profile_id": "fixed_depth_16", "search_depth": 16},
        synthetic_rows(reverse_second_group=True),
        400.0,
        frozen_groups(),
    )
    slow, _ = assess_profile(
        load_config(),
        {"profile_id": "fixed_depth_16", "search_depth": 16},
        synthetic_rows(),
        500.0,
        frozen_groups(),
    )

    assert inaccurate["all_gate_checks_passed"] is False
    assert inaccurate["continue_to_next_profile"] is True
    assert slow["all_gate_checks_passed"] is False
    assert (
        slow["gate_checks"]["throughput_speedup_vs_recorded_32vcpu"]["passed"] is False
    )
    assert slow["continue_to_next_profile"] is False


def test_runtime_approval_ignores_paths_but_not_binary_identity():
    approved = json.loads(
        Path(
            "data/stage06_mk14_vinagpu21_deterministic_batch_runtime_lock.json"
        ).read_text(encoding="ascii")
    )
    relocated = {
        **approved,
        "executable_path": "/different/vina-gpu",
        "opencl_binary_path": "/different/kernels",
        "deterministic_batch_patch": {
            **approved["deterministic_batch_patch"],
            "manifest_path": "/different/patch.json",
        },
    }

    assert runtime_projection(relocated) == runtime_projection(approved)
    validate_approved_runtime(relocated, approved)
    with pytest.raises(ValueError, match="runtime differs"):
        validate_approved_runtime(
            {**relocated, "executable_sha256": "0" * 64}, approved
        )
