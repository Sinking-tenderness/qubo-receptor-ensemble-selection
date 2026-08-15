from scripts.prepare_stage13e_egfr_redocking_inputs import derive_common_box


def test_fa10_common_box_keeps_minimum_axis_size() -> None:
    import numpy as np

    result = derive_common_box(
        [np.array([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])],
        minimum_margin_angstrom=3.5,
        size_increment_angstrom=2.0,
        minimum_axis_size_angstrom=22.0,
        center_decimals=2,
    )
    assert result["size"] == {"x": 22.0, "y": 22.0, "z": 22.0}
