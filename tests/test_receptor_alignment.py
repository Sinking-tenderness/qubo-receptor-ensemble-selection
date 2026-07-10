import numpy as np

from scripts.align_receptor_structure import (
    calculate_kabsch_transform,
    rmsd,
    transform_coordinates,
)


def test_kabsch_recovers_rigid_body_transform() -> None:
    mobile = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    known_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    known_translation = np.array([4.0, -2.0, 7.0])
    reference = mobile @ known_rotation + known_translation

    rotation, translation = calculate_kabsch_transform(mobile, reference)
    aligned = transform_coordinates(mobile, rotation, translation)

    assert rmsd(aligned, reference) < 1e-12
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)
