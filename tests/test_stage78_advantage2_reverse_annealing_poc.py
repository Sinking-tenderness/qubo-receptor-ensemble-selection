import hashlib
import json
from pathlib import Path

import dimod
import pytest

import scripts.experimental.quantum.run_stage78_advantage2_reverse_annealing_poc as hw


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="ascii"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_stage78_implementations_are_hash_locked_and_audited():
    config = read_json("configs/stage78_advantage2_reverse_annealing_poc.json")
    for descriptor in config["implementation"].values():
        path = ROOT / descriptor["path"]
        assert sha256(path) == descriptor["sha256"]
        assert path.stat().st_size == descriptor["size_bytes"]
    result = read_json("data/stage78_advantage2_reverse_annealing_poc_result.json")
    audit = read_json("data/stage78_advantage2_reverse_annealing_poc_audit.json")
    assert result["status"] == "stage78_advantage2_reverse_annealing_poc_frozen"
    assert audit["status"].endswith("independent_audit_ok")
    assert audit["exact_milp_objectives_independently_matched"] == 6
    assert audit["qpu_jobs_observed"] == 0


def test_stage78_exact_panel_has_two_positives_three_negatives_and_calibration():
    result = read_json("data/stage78_advantage2_reverse_annealing_poc_result.json")
    by_role = {}
    for instance in result["instances"]:
        by_role.setdefault(instance["role"], []).append(instance)
    assert len(by_role["confirmation_positive"]) == 2
    assert len(by_role["confirmation_negative"]) == 3
    assert len(by_role["calibration_diagnostic"]) == 1
    assert all(
        row["exact_reference"]["improvement_from_warm"] < 0
        for row in by_role["confirmation_positive"]
        + by_role["calibration_diagnostic"]
    )
    assert all(
        row["exact_reference"]["improvement_from_warm"] == pytest.approx(0.0)
        for row in by_role["confirmation_negative"]
    )


def test_stage78_local_execution_bundle_validates_without_cloud_access(tmp_path):
    result = read_json("data/stage78_advantage2_reverse_annealing_poc_result.json")
    validation = hw.local_validate(ROOT, result)
    assert validation["status"] == "stage78_local_execution_bundle_valid"
    assert validation["instance_count"] == 6
    assert validation["cloud_queries"] == 0
    assert validation["qpu_jobs"] == 0
    assert all(row["warm_identity"] for row in validation["instances"])
    assert all(row["exact_certificate"] for row in validation["instances"])


def test_stage78_manual_spin_gauge_transforms_initial_state_and_round_trips():
    original = dimod.BinaryQuadraticModel.from_ising(
        {"a": -0.25, "b": 0.5}, {("a", "b"): -1.0}
    )
    gauge = [-1, 1]
    transformed, initial = hw.gauged_spin_bqm(original, gauge)
    assert initial == {"a": 1, "b": -1}
    for a in (-1, 1):
        for b in (-1, 1):
            transformed_sample = {"a": a, "b": b}
            binary = hw.unflip_spin_sample(["a", "b"], transformed_sample, gauge)
            original_spin = {
                name: 2 * value - 1 for name, value in binary.items()
            }
            assert transformed.energy(transformed_sample) == pytest.approx(
                original.energy(original_spin)
            )


def test_stage78_primary_analysis_counts_chain_breaks_as_failures():
    base = {
        "instance_id": "example",
        "condition_id": "confirm_reverse",
        "mode": "reverse",
        "embedding_index": 0,
        "gauge_index": 0,
        "num_occurrences": 1,
        "feasible": True,
        "strict_improvement": True,
        "exact_optimum": True,
        "guarded_energy": -2.0,
        "energy_below_certified_optimum": False,
    }
    intact = {
        **base,
        "read_index": 0,
        "chain_break_fraction": 0.0,
        "main_analysis_eligible": True,
        "guarded_strict_improvement": True,
        "guarded_exact_optimum": True,
    }
    broken = {
        **base,
        "read_index": 1,
        "chain_break_fraction": 0.25,
        "main_analysis_eligible": False,
        "guarded_strict_improvement": False,
        "guarded_exact_optimum": False,
    }
    summary = hw.block_summaries([intact, broken])[0]
    assert summary["read_count"] == 2
    assert summary["intact_chain_read_fraction"] == pytest.approx(0.5)
    assert summary["guarded_strict_improvement_read_fraction"] == pytest.approx(0.5)
    assert summary["strict_improvement_recovered"] is True


def test_stage78_paid_sampling_has_a_double_interlock(monkeypatch):
    monkeypatch.delenv("STAGE78_QPU_ACK", raising=False)
    with pytest.raises(PermissionError):
        hw.paid_authorized(True)
    monkeypatch.setenv("STAGE78_QPU_ACK", hw.PAID_ACKNOWLEDGEMENT)
    with pytest.raises(PermissionError):
        hw.paid_authorized(False)
    hw.paid_authorized(True)


def test_stage78_protocol_and_external_environment_are_frozen():
    config = read_json("configs/stage78_advantage2_reverse_annealing_poc.json")
    protocol = config["hardware_protocol"]
    assert protocol["solver_selector"] == {"topology__type": "zephyr"}
    assert protocol["confirmation"]["embedding_count"] == 2
    assert protocol["confirmation"]["gauge_count"] == 8
    assert protocol["planned_default_total_qpu_jobs"] == 232
    assert protocol["planned_default_total_qpu_reads"] == 23_200
    assert protocol["maximum_planned_qpu_access_time_seconds"] == 20.0
    environment = (
        ROOT / "environment/stage78_dwave_advantage2.yml"
    ).read_text(encoding="ascii")
    assert "dwave-ocean-sdk==9.4.0" in environment
    assert 'numpy>=2,<3' in environment
    source = (
        ROOT
        / "scripts/experimental/quantum/run_stage78_advantage2_reverse_annealing_poc.py"
    ).read_text(encoding="ascii")
    assert "SpinReversalTransformComposite" not in source
    assert '"auto_scale": False' in source
