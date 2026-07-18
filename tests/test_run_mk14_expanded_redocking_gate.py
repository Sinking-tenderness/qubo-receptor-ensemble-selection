import numpy as np

from scripts.run_mk14_expanded_redocking_gate import (
    box_audit,
    build_modelserver_url,
    point_set_rmsd_by_element,
)


def test_modelserver_url_uses_auth_identifiers():
    url = build_modelserver_url(
        "https://models.rcsb.org/v1/{pdb_id}/ligand", "3UVR", "A", 361
    )

    assert url == (
        "https://models.rcsb.org/v1/3UVR/ligand?"
        "auth_asym_id=A&auth_seq_id=361&encoding=sdf"
    )


def test_point_set_rmsd_matches_equal_elements_without_pose_fitting():
    first = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    second = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

    rmsd, maximum = point_set_rmsd_by_element(
        first, ["C", "C", "N"], second, ["C", "C", "N"]
    )

    assert rmsd == 0.0
    assert maximum == 0.0


def test_box_audit_reports_smallest_face_margin():
    coordinates = np.array([[-1.0, -2.0, -3.0], [2.0, 1.0, 3.0]])
    result = box_audit(
        coordinates,
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 10.0, "y": 10.0, "z": 10.0},
    )

    assert result["minimum_margin_angstrom"] == 2.0
