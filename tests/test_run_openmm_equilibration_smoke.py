import numpy as np
import pytest
from pathlib import Path

from scripts.run_openmm_equilibration_smoke import (
    GAS_CONSTANT_KJ_PER_MOL_K,
    centered_rmsd,
    load_smoke_config,
    platform_properties,
    steps_for_duration,
    temperature_from_kinetic_energy,
)


def test_steps_for_duration_uses_ps_and_fs_consistently():
    assert steps_for_duration(10.0, 2.0) == 5000


def test_load_smoke_config_accepts_derived_protocol():
    config = load_smoke_config(Path("configs/stage03_cdk2_af2_md_cuda_equilibration_smoke.json"))
    assert config["platform"]["precision"] == "mixed"


def test_platform_properties_support_cpu_and_opencl():
    assert platform_properties("CPU", None, 8) == {"Threads": "8"}
    assert platform_properties("OpenCL", "single", None) == {"Precision": "single"}
    assert platform_properties("CUDA", "mixed", None) == {"Precision": "mixed"}


def test_temperature_from_kinetic_energy_matches_equipartition():
    dof = 12
    temperature = 300.0
    kinetic = 0.5 * dof * GAS_CONSTANT_KJ_PER_MOL_K * temperature
    assert temperature_from_kinetic_energy(kinetic, dof) == pytest.approx(temperature)


def test_centered_rmsd_removes_translation():
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mobile = reference + np.array([2.0, 3.0, 4.0])
    assert centered_rmsd(reference, mobile) == pytest.approx(0.0, abs=1e-12)
