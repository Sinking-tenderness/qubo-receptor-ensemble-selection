from scripts.sample_split_pilot import sample_rows


def test_pilot_is_balanced_deterministic_and_group_diverse() -> None:
    rows = []
    for label in ("active", "decoy"):
        for index in range(5):
            rows.append(
                {
                    "ligand_id": f"{label}_{index}",
                    "label": label,
                    "split": "train",
                    "split_group_id": f"{label}_group_{index}",
                    "scaffold_smiles": f"{label}_scaffold_{index}",
                }
            )

    first = sample_rows(rows, "train", count_per_label=3, seed=17)
    second = sample_rows(rows, "train", count_per_label=3, seed=17)

    assert first == second
    assert sum(row["label"] == "active" for row in first) == 3
    assert sum(row["label"] == "decoy" for row in first) == 3
    assert len({row["split_group_id"] for row in first}) == 6
    assert {row["pilot_role"] for row in first} == {"execution_smoke_only"}
