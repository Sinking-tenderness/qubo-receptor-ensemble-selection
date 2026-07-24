import json
from pathlib import Path

import pytest

from scripts.experimental.vinagpu.audit_vinagpu_equivalence import (
    average_ranks,
    group_metrics,
    pearson,
    quantile,
    spearman,
    top_overlap,
)
from scripts.experimental.vinagpu.run_vinagpu_equivalence import (
    ensure_runtime_lock,
    makefile_settings,
    pair_seed,
    parse_vina_pose,
    validate_inputs,
    vinagpu_command,
)


CONFIG_PATH = Path(
    "configs/stage06_mk14_vinagpu21_train160_equivalence.json"
)


def test_frozen_inputs_are_consumed_train_only_and_complete():
    config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))

    receptors, ligands, audit = validate_inputs(Path.cwd().resolve(), config)

    assert len(receptors) == 5
    assert len(ligands) == 160
    assert audit["gpu_pair_count"] == 2400
    assert audit["validation_rows"] == 0
    assert audit["test_rows"] == 0
    assert audit["macrocycle_closure_pseudoatom_ligand_count"] == 0


def test_pair_seed_matches_cpu_ligand_offset():
    ligand = {"ligand_id": "L7", "seed_offset": "7"}

    assert pair_seed(20260801, ligand) == 20260808


def test_vinagpu_command_freezes_single_ligand_seed_and_heuristic_depth(
    tmp_path: Path,
):
    protocol = {
        "thread": 8000,
        "search_depth": "heuristic",
        "rilc_bfgs": 1,
        "num_modes": 9,
        "energy_range": 3,
        "box": {
            "center_x": -0.49,
            "center_y": 3.26,
            "center_z": 21.83,
            "size_x": 22,
            "size_y": 24,
            "size_z": 32,
        },
    }

    command = vinagpu_command(
        tmp_path / "vina-gpu",
        tmp_path / "kernels",
        tmp_path / "receptor.pdbqt",
        tmp_path / "ligand.pdbqt",
        tmp_path / "pose_out.pdbqt",
        protocol,
        20260808,
    )

    assert command[command.index("--seed") + 1] == "20260808"
    assert command[command.index("--thread") + 1] == "8000"
    assert "--ligand" in command
    assert "--ligand_directory" not in command
    assert "--search_depth" not in command


def test_parse_vina_pose_reads_first_mode_and_counts_all_modes(tmp_path: Path):
    pose = tmp_path / "pose.pdbqt"
    pose.write_text(
        "MODEL 1\n"
        "REMARK VINA RESULT:      -8.9     0.000     0.000\n"
        "ENDMDL\n"
        "MODEL 2\n"
        "REMARK VINA RESULT:      -8.4     1.000     2.000\n"
        "ENDMDL\n",
        encoding="ascii",
    )

    score, count = parse_vina_pose(pose)

    assert score == -8.9
    assert count == 2


def test_makefile_settings_reads_the_frozen_small_box_profile(tmp_path: Path):
    makefile = tmp_path / "Makefile"
    makefile.write_text(
        "GPU_PLATFORM=-DNVIDIA_PLATFORM\n"
        "OPENCL_VERSION=-DOPENCL_3_0\n"
        "DOCKING_BOX_SIZE=-DSMALL_BOX\n",
        encoding="ascii",
    )

    assert makefile_settings(makefile) == {
        "GPU_PLATFORM": "-DNVIDIA_PLATFORM",
        "OPENCL_VERSION": "-DOPENCL_3_0",
        "DOCKING_BOX_SIZE": "-DSMALL_BOX",
    }


def test_runtime_lock_is_immutable_after_first_write(tmp_path: Path):
    path = tmp_path / "runtime_lock.json"
    evidence = {"schema_version": "1.0", "status": "locked", "hash": "ABC"}

    ensure_runtime_lock(path, evidence)
    ensure_runtime_lock(path, evidence)

    with pytest.raises(ValueError, match="immutable lock"):
        ensure_runtime_lock(path, {**evidence, "hash": "DEF"})


def test_metric_helpers_preserve_lower_is_better_ranking():
    assert average_ranks([1.0, 1.0, 3.0]) == [1.5, 1.5, 3.0]
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert spearman([1.0, 1.0, 3.0], [2.0, 2.0, 5.0]) == pytest.approx(1.0)
    assert quantile([0.0, 10.0], 0.95) == 9.5
    ligand_ids = [f"L{index}" for index in range(20)]
    overlap, count = top_overlap(
        ligand_ids,
        [float(index) for index in range(20)],
        [float(index) for index in range(20)],
        0.05,
    )
    assert count == 1
    assert overlap == 1.0


def test_group_metrics_use_explicit_vinagpu_score_column():
    rows = [
        {
            "seed_id": "seed0",
            "receptor_id": "R1",
            "ligand_id": f"L{index:02d}",
            "cpu_vina_e32_score": float(index),
            "gpu_vinagpu21_score": float(index),
            "score_delta_gpu_minus_cpu": 0.0,
        }
        for index in range(20)
    ]

    metrics = group_metrics(rows)

    assert metrics[0]["spearman"] == pytest.approx(1.0)
    assert metrics[0]["top5pct_overlap"] == 1.0
