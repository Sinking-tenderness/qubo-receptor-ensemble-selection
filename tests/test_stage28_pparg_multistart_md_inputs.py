import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_openmm_system import load_protocol
from scripts.run_openmm_equilibration_smoke import load_smoke_config


def test_stage28_preparation_freezes_eight_complete_qpu_independent_starts() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/stage28_pparg_multistart_md_ensemble.json").read_text(encoding="ascii"))
    result = json.loads((root / "data/stage28_pparg_multistart_md_input_preparation_result.json").read_text(encoding="ascii"))
    assert result["status"] == "stage28_pparg_multistart_md_inputs_ready"
    assert result["starting_structure_count"] == 8
    assert result["expected_total_frames"] == 1200
    assert [row["selection_rank"] for row in result["starting_structures"]] == [1, 2, 3, 4, 5, 6, 8, 10]
    assert result["data_boundary"]["stage27_qubo_subsets_read"] == 0
    assert config["evidence_timing"]["stage28_md_outcomes_known_before_freeze"] is False
    assert config["evidence_timing"]["docking_scores_permitted"] is False


def test_stage28_generated_openmm_configs_are_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    import csv
    with (root / "data/processed/stage28_pparg_multistart_md_start_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    for row in rows:
        protocol = load_protocol(root / row["protocol_config"])
        equilibration = load_smoke_config(root / row["equilibration_config"])
        production = load_smoke_config(root / row["production_config"])
        qc = json.loads((root / row["trajectory_qc_config"]).read_text(encoding="ascii"))
        assert Path(protocol["starting_structure"]["pdb_path"]).is_absolute() is False
        assert production["dynamics"]["production_duration_ns"] == 3.0
        assert production["dynamics"]["frame_interval_ps"] == 20.0
        assert equilibration["platform"]["name"] == "CUDA"
        assert equilibration["parent_protocol"] == row["protocol_config"]
        assert production["parent_protocol"] == row["protocol_config"]
        assert qc["expected_frame_count"] == 150
        assert int(row["expected_frame_count"]) == 150


def test_stage28_remote_environment_pins_cuda_and_executes_context_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = (root / "environment/stage03_openmm.yml").read_text(encoding="ascii")
    runner = (root / "scripts/run_stage28_pparg_multistart_md_remote.sh").read_text(encoding="ascii")
    assert "cuda-version=12" in environment
    assert "Context(" in runner
    assert "cuda_context_smoke=ok" in runner


def test_stage28_result_audit_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data/stage28_pparg_multistart_md_ensemble_audit.json"
    if not path.exists():
        return
    audit = json.loads(path.read_text(encoding="ascii"))
    assert audit["status"] == "stage28_pparg_multistart_md_ensemble_audit_ok"
    assert audit["decision"]["stage29_solver_scaling_authorized"] is True
