from __future__ import annotations

from scripts.build_dude_ligand_manifest import build_rows, duplicate_id_audit
from scripts.make_dude_subset import DudeRecord


def test_build_rows_preserves_duplicate_source_ids_and_uses_line_ids() -> None:
    actives = [DudeRecord("CCO", "A1", "X", 3)]
    decoys = [
        DudeRecord("CCN", "D1", "", 7),
        DudeRecord("CCC", "D1", "", 9),
    ]

    rows = build_rows(actives, decoys, "MK14", "DUD-E")
    audit = duplicate_id_audit(rows)

    assert [row["ligand_id"] for row in rows] == [
        "MK14_active_L000003",
        "MK14_decoy_L000007",
        "MK14_decoy_L000009",
    ]
    assert len(rows) == 3
    assert audit["duplicate_label_source_id_count"] == 1
    assert audit["rows_with_duplicate_label_source_id"] == 2
