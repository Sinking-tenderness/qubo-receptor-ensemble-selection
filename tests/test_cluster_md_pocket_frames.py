import numpy as np
import pytest

from scripts.cluster_md_pocket_frames import (
    medoid_indices,
    pairwise_point_distances_angstrom,
    relabel_by_medoid_time,
    silhouette_from_distances,
    standardize_features,
)


def test_pairwise_point_distances_are_in_angstrom():
    points = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    values = pairwise_point_distances_angstrom(points)
    assert values[0] == pytest.approx([10.0, 10.0, np.sqrt(200.0)])


def test_standardize_features_drops_low_variance_columns():
    values = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    scaled, kept, means, standard_deviations = standardize_features(values, minimum_sd=0.01)
    assert kept.tolist() == [False, True]
    assert scaled[:, 0].mean() == pytest.approx(0.0)
    assert means.tolist() == pytest.approx([1.0, 1.0])
    assert standard_deviations[0] == pytest.approx(0.0)


def test_medoid_and_relabeling_are_deterministic():
    labels = np.array([5, 5, 2, 2])
    distances = np.array([
        [0.0, 1.0, 8.0, 9.0],
        [1.0, 0.0, 7.0, 8.0],
        [8.0, 7.0, 0.0, 2.0],
        [9.0, 8.0, 2.0, 0.0],
    ])
    assert medoid_indices(labels, distances) == {2: 2, 5: 0}
    assert relabel_by_medoid_time(labels, distances).tolist() == [0, 0, 1, 1]


def test_silhouette_is_high_for_well_separated_clusters():
    labels = np.array([0, 0, 1, 1])
    distances = np.array([
        [0.0, 1.0, 9.0, 10.0],
        [1.0, 0.0, 8.0, 9.0],
        [9.0, 8.0, 0.0, 1.0],
        [10.0, 9.0, 1.0, 0.0],
    ])
    assert silhouette_from_distances(labels, distances) > 0.85
