import numpy as np
import pytest

from scripts.analyze_md_trajectory import (
    direct_rmsd_angstrom,
    finite_summary,
    load_config,
    per_atom_rmsf_angstrom,
)


def test_direct_rmsd_converts_nm_to_angstrom():
    reference = np.zeros((2, 3))
    frames = np.zeros((1, 2, 3))
    frames[0, 0, 0] = 0.1
    value = direct_rmsd_angstrom(frames, reference, np.array([0]))
    assert value[0] == pytest.approx(1.0)


def test_per_atom_rmsf_uses_trajectory_mean():
    frames = np.array([[[0.0, 0.0, 0.0]], [[0.2, 0.0, 0.0]]])
    value = per_atom_rmsf_angstrom(frames, np.array([0]))
    assert value[0] == pytest.approx(1.0)


def test_finite_summary_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        finite_summary(np.array([1.0, np.nan]))


def test_load_config_requires_complete_output_paths(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """{
          "schema_version": "1.0",
          "experiment_id": "test",
          "production_experiment_id": "production",
          "purpose": "test",
          "inputs": {"topology_pdb": "topology.pdb", "trajectory_glob": "*.dcd"},
          "frame_interval_ps": 20.0,
          "expected_frame_count": 100,
          "alignment_selection": "protein and backbone",
          "pocket_residue_numbers": [10],
          "outputs": {"summary_json": "summary.json"},
          "interpretation_boundary": "test only"
        }""",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="outputs"):
        load_config(config_path)
