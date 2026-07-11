from scripts.extend_score_matrix import parse_raw_tables


def test_parse_raw_tables_requires_successful_pairs(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text(
        "target_id,receptor_id,ligand_id,label,pose_rank,docking_score,status\n"
        "CDK2,R1,L1,active,1,-8.0,ok\n",
        encoding="utf-8",
    )
    rows, receptors = parse_raw_tables([path])
    assert receptors == {"R1"}
    assert rows[0]["representative_score"] == -8.0
