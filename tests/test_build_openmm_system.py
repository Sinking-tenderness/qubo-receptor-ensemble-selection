import json
from pathlib import Path

import pytest

from scripts.build_openmm_system import load_protocol


def test_load_protocol_accepts_versioned_pilot_config():
    protocol = load_protocol(Path("configs/stage03_cdk2_af2_md_pilot.json"))
    assert protocol["experiment_id"] == "stage03-cdk2-af2-apo-md-pilot-v1"
    assert protocol["dynamics"]["pilot_production_duration_ns"] == 2.0


def test_load_protocol_rejects_missing_sections(tmp_path: Path):
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps({"schema_version": "1.0"}), encoding="ascii")
    with pytest.raises(ValueError, match="missing required keys"):
        load_protocol(path)
