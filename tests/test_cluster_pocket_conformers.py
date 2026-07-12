import numpy as np

from scripts.cluster_pocket_conformers import medoids


def test_medoids_choose_lowest_total_within_cluster_distance():
    labels = np.array([0, 0, 1])
    distances = np.array([[0.0, 1.0, 5.0], [1.0, 0.0, 4.0], [5.0, 4.0, 0.0]])
    assert medoids(labels, distances) == {0: 0, 1: 2}
