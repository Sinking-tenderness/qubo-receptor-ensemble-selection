from scripts.build_stage05_mk14_train696_panel import select_panel


def row(ligand_id: str, label: str, group: str) -> dict[str, str]:
    return {
        "ligand_id": ligand_id,
        "label": label,
        "split": "train",
        "split_group_id": group,
        "scaffold_smiles": group,
        "smiles": "C",
        "target_id": "MK14",
    }


def test_train_panel_keeps_existing_and_adds_group_diverse_decoys() -> None:
    split_rows = [
        row("A1", "active", "GA1"),
        row("A2", "active", "GA2"),
        row("D1", "decoy", "GD1"),
        row("D2", "decoy", "GD2"),
        row("D3", "decoy", "GD3"),
        row("D4", "decoy", "GD3"),
    ]
    existing = [split_rows[0], split_rows[2]]

    panel, new_rows = select_panel(
        split_rows,
        existing,
        active_count=2,
        decoy_count=2,
        seed=7,
        selection_role="expanded",
    )

    assert {row["ligand_id"] for row in existing}.issubset(
        {row["ligand_id"] for row in panel}
    )
    assert len(panel) == 4
    assert len(new_rows) == 2
    assert len({row["split_group_id"] for row in panel}) == 4
    assert {row["selection_role"] for row in panel} == {"expanded"}


def test_train_panel_selection_is_deterministic() -> None:
    split_rows = [
        row("A1", "active", "GA1"),
        row("A2", "active", "GA2"),
        *[row(f"D{i}", "decoy", f"GD{i}") for i in range(1, 8)],
    ]
    existing = [split_rows[0], split_rows[2]]

    first, _ = select_panel(split_rows, existing, 2, 3, 11, "expanded")
    second, _ = select_panel(split_rows, existing, 2, 3, 11, "expanded")

    assert [row["ligand_id"] for row in first] == [
        row["ligand_id"] for row in second
    ]
