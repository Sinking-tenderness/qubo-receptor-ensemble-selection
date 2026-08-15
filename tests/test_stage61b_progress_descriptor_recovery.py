import hashlib
import json
from pathlib import Path

from scripts.repair_stage61b_progress_descriptor import repair


ROOT = Path(__file__).resolve().parents[1]


def descriptor(root: Path, path: Path):
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        "size_bytes": path.stat().st_size,
    }


def test_stage61b_repair_changes_only_stale_progress_descriptor(tmp_path):
    config_source = ROOT / "configs/stage61b_ppard_remaining144_unidock113_production.json"
    config_path = tmp_path / "configs/stage61b_ppard_remaining144_unidock113_production.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(config_source.read_bytes())
    config = json.loads(config_path.read_text())
    outputs = config["outputs"]
    generated = {}
    for key in (
        "scores_csv", "batch_runs_csv", "median_matrix_csv", "minimum_matrix_csv"
    ):
        path = tmp_path / outputs[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{key}\n")
        generated[key] = descriptor(tmp_path, path)
    progress = tmp_path / outputs["progress_json"]
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text('{"status":"stage61b_production_complete"}\n')
    stale = {
        "path": progress.relative_to(tmp_path).as_posix(),
        "sha256": "0" * 64,
        "size_bytes": 1,
    }
    summary_path = tmp_path / outputs["summary_json"]
    summary = {
        "status": "stage61b_ppard_remaining144_unidock_matrix_ok",
        "config": {
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest().upper()
        },
        "batch_count": 87,
        "pair_count": 12528,
        "unresolved_warning_event_count": 0,
        "pose_integrity_failure_count": 0,
        "outputs": {**generated, "progress_json": stale},
    }
    summary_path.write_text(json.dumps(summary) + "\n")
    amendment = repair(
        tmp_path,
        Path("configs/stage61b_ppard_remaining144_unidock113_production.json"),
        Path("data/amendment.json"),
    )
    repaired = json.loads(summary_path.read_text())
    assert amendment["docking_jobs_reexecuted"] == 0
    assert amendment["changed_descriptor_fields"] == ["sha256", "size_bytes"]
    assert repaired["outputs"]["progress_json"] == descriptor(tmp_path, progress)
    for key in generated:
        assert repaired["outputs"][key] == generated[key]
