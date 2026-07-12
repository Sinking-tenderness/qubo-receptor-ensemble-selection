import pytest

from scripts.benchmark_openmm_platform import build_platform_properties, ns_per_day


def test_ns_per_day_uses_fixed_units():
    assert ns_per_day(1000, 2.0, 2.0) == pytest.approx(86.4)


def test_ns_per_day_rejects_invalid_elapsed_time():
    with pytest.raises(ValueError, match="elapsed_seconds"):
        ns_per_day(1000, 2.0, 0.0)


def test_platform_properties_are_specific_to_requested_platform():
    assert build_platform_properties("CPU", 8, "mixed") == {"Threads": "8"}
    assert build_platform_properties("OpenCL", 8, "mixed") == {"Precision": "mixed"}
    assert build_platform_properties("CUDA", None, "mixed") == {"Precision": "mixed"}
