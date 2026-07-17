from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.transform_sdf_coordinates import (
    load_transform,
    maximum_distance_error,
    transform_coordinates,
)


def test_transform_uses_row_vector_convention_and_preserves_distances() -> None:
    coordinates = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([4.0, 5.0, 6.0])

    transformed = transform_coordinates(coordinates, rotation, translation)

    np.testing.assert_allclose(transformed, [[4.0, 6.0, 6.0], [2.0, 5.0, 6.0]])
    assert maximum_distance_error(coordinates, transformed) < 1e-12


def test_load_transform_rejects_reflection(tmp_path: Path) -> None:
    summary = tmp_path / "reflection.json"
    summary.write_text(
        json.dumps(
            {
                "rotation_matrix_row_vector_convention": [
                    [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "translation_vector_angstrom": [0.0, 0.0, 0.0],
            }
        ),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match=r"determinant is not \+1"):
        load_transform(summary)


def test_identity_transform_requires_no_summary() -> None:
    rotation, translation, summary = load_transform(None)

    assert rotation == pytest.approx(np.eye(3))
    assert translation == pytest.approx(np.zeros(3))
    assert summary == {"method": "identity"}
