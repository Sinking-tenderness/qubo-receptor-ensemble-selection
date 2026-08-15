from pathlib import Path

from scripts.build_stage08c_mk14_current15_manifest import read_json


def test_current15_manifest_has_no_failed_receptors() -> None:
    path = Path("data/stage08c_mk14_current15_receptor_manifest_summary.json")
    if not path.exists():
        return
    result = read_json(path)
    assert result["status"] == "stage08c_current15_manifest_ok"
    assert result["current_receptor_count"] == 15
    assert "MK14_3ITZ_aligned" in result["current_receptor_ids"]
    assert not set(result["excluded_receptor_ids"]).intersection(
        result["current_receptor_ids"]
    )
