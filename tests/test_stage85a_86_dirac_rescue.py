import math
from pathlib import Path

import scripts.prepare_stage86_nonnegative_gauge_dirac_rescue as s86
import scripts.run_stage81_dirac_global_qubo_formulation_gate as s81
import scripts.run_stage84_mixed_radix_dirac_iqp_gate as s84


ROOT = Path(__file__).resolve().parents[1]


def stage86_context():
    config = s84.read_json(ROOT / "configs/stage86_nonnegative_gauge_dirac_rescue.json")
    stage84_config = s84.read_json(ROOT / config["inputs"]["stage84_config"])
    cells = s81.canonical_cells(stage84_config, ROOT)
    selection = config["rescue_instance"]
    cell = next(
        cell
        for cell in cells
        if cell["model"]["record"]["target_id"] == selection["target_id"]
        and int(cell["model"]["record"]["outer_fold"])
        == int(selection["outer_fold"])
    )
    return config, selection, cell


def test_stage85_physical_failure_is_frozen():
    result = s84.read_json(ROOT / "data/stage85a_qci_dirac3_failure_adjudication.json")
    assert result["status"] == "stage85_physical_calibration_failed_stop_hardware"
    assert result["completed_device_jobs"] == 2
    assert result["completed_device_samples"] == 50
    assert result["recorded_device_usage_seconds"] == 71.0
    assert all(item["feasible_sample_count"] == 0 for item in result["summaries"])
    assert result["additional_stage85_device_jobs_authorized"] == 0


def test_stage86_nonnegative_gauge_has_global_exact_penalty():
    _, selection, cell = stage86_context()
    k = int(selection["k"])
    encoding = s86.encode_cell(cell, k)
    assert encoding["pair_span"] > 0
    assert encoding["objective_upper_bound"] == math.comb(k, 2)
    assert encoding["constraint_weight"] == math.comb(k, 2) + 1
    assert encoding["global_penalty_margin"] == 1.0
    assert encoding["coefficient_retention_fraction"] == 1.0


def test_stage86_rescue_instance_is_free_tier_compatible_and_exact():
    config, selection, cell = stage86_context()
    encoding = s86.encode_cell(cell, int(selection["k"]))
    result = s84.read_json(ROOT / config["outputs"]["result_json"])
    mapping = s84.read_json(ROOT / result["rescue_instance"]["mapping"]["path"])
    selected = set(mapping["quantized_exact"]["selected_subset"].split("+"))
    subset = tuple(
        index
        for index, receptor_id in enumerate(mapping["receptor_ids"])
        if receptor_id in selected
    )
    _, residuals = s86.assignment_for_subset(encoding, cell["model"], subset)
    assert len(encoding["names"]) == 27
    assert len(encoding["names"]) <= config["local_gate"]["free_tier_quadratic_variable_limit"]
    assert len(subset) == int(selection["k"])
    assert residuals == [0, 0, 0, 0]
