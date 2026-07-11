import pytest


qiskit = pytest.importorskip("qiskit")


def test_qaoa_module_dependencies_are_available() -> None:
    import qiskit_aer

    assert qiskit.__version__ == "2.5.0"
    assert qiskit_aer.__version__ == "0.17.2"
