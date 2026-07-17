from scripts.check_ligand_smiles import audit_row, build_summary


def test_large_identifier_lists_are_counted_and_bounded() -> None:
    rows = []
    for index in range(105):
        rows.append(
            audit_row(
                {
                    "ligand_id": f"charged_{index}",
                    "smiles": "C[NH3+]",
                    "label": "decoy",
                    "source": "test",
                    "target_id": "target",
                }
            )
        )

    summary = build_summary(rows)

    assert summary["charged_ligand_count"] == 105
    assert len(summary["charged_ligand_ids_preview"]) == 100
    assert summary["preview_limit"] == 100
