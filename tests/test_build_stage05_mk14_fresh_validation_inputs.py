from scripts.build_stage05_mk14_fresh_validation_inputs import (
    freeze_normalization_bounds,
    group_order_key,
    select_fresh_panel,
)


def row(ligand_id: str, label: str, group: str) -> dict[str, str]:
    return {
        "ligand_id": ligand_id,
        "label": label,
        "split": "validation",
        "split_group_id": group,
    }


def test_fresh_panel_excludes_consumed_groups_and_preserves_selected_groups():
    validation = [
        row("old-a", "active", "g-old"),
        row("old-d", "decoy", "g-old-d"),
        row("a1", "active", "g-a1"),
        row("hard-d", "decoy", "g-a1"),
        row("a2", "active", "g-a2"),
        row("d1", "decoy", "g-d1"),
        row("d2", "decoy", "g-d2"),
        row("d3", "decoy", "g-d3"),
        row("d4", "decoy", "g-d4"),
    ]
    consumed = [validation[0], validation[1]]
    selection = {
        "selection_seed": 17,
        "selection_role": "fresh_validation_preregistered",
        "target_decoy_to_active_ratio": 2,
        "minimum_selected_active_count": 2,
        "maximum_selected_ligand_count": 8,
    }

    panel, audit = select_fresh_panel(validation, consumed, selection)
    selected_groups = {item["split_group_id"] for item in panel}

    assert {"g-old", "g-old-d"}.isdisjoint(selected_groups)
    assert {"g-a1", "g-a2"}.issubset(selected_groups)
    assert {item["ligand_id"] for item in panel}.issuperset(
        {"a1", "a2", "hard-d"}
    )
    assert audit["selected_label_counts"] == {"active": 2, "decoy": 4}
    assert audit["consumed_group_overlap_count"] == 0
    assert all(
        item["selection_role"] == "fresh_validation_preregistered"
        for item in panel
    )


def test_group_order_is_deterministic_and_seed_specific():
    assert group_order_key(11, "g1") == group_order_key(11, "g1")
    assert group_order_key(11, "g1") != group_order_key(12, "g1")


def test_freeze_normalization_bounds_uses_each_matrix_and_seed():
    receptors = ["R1", "R2"]
    primary = []
    sensitivity = []
    long_rows = []
    for index in range(696):
        ligand_id = f"L{index:03d}"
        label = "active" if index < 348 else "decoy"
        primary.append(
            {
                "ligand_id": ligand_id,
                "label": label,
                "R1": str(-10.0 + index / 1000),
                "R2": str(-8.0 + index / 1000),
            }
        )
        sensitivity.append(
            {
                "ligand_id": ligand_id,
                "label": label,
                "R1": str(-11.0 + index / 1000),
                "R2": str(-9.0 + index / 1000),
            }
        )
        for receptor_index, receptor in enumerate(
            ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
        ):
            base = -12.0 + receptor_index + index / 1000
            long_rows.append(
                {
                    "ligand_id": ligand_id,
                    "receptor_id": receptor,
                    "seed0_representative_score": str(base),
                    "seed1_representative_score": str(base + 0.1),
                    "seed2_representative_score": str(base + 0.2),
                }
            )

    bounds, audit = freeze_normalization_bounds(
        primary, sensitivity, long_rows, receptors
    )

    assert bounds["primary"]["R1"]["minimum"] == -10.0
    assert bounds["sensitivity"]["R1"]["minimum"] == -11.0
    assert bounds["seed0"]["R1"]["minimum"] == -12.0
    assert bounds["seed2"]["R2"]["minimum"] == -10.8
    assert audit["train_ligand_count"] == 696
    assert audit["matrix_ids"] == [
        "primary",
        "sensitivity",
        "seed0",
        "seed1",
        "seed2",
    ]
