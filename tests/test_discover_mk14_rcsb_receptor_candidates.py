from pathlib import Path

from scripts.discover_mk14_rcsb_receptor_candidates import (
    load_config,
    normalize_entry,
    read_json,
    search_payload,
    validate_preregistration,
)


CONFIG_PATH = Path("configs/stage05_mk14_rcsb_candidate_discovery.json")


def sample_entry(mutation_count=0, ligand_id="LIG", ligand_weight=0.4):
    return {
        "rcsb_id": "TEST",
        "struct": {"title": "MAPK14 test structure"},
        "exptl": [{"method": "X-RAY DIFFRACTION"}],
        "rcsb_entry_info": {"resolution_combined": [2.0]},
        "rcsb_accession_info": {"initial_release_date": "2020-01-01T00:00:00Z"},
        "polymer_entities": [
            {
                "rcsb_id": "TEST_1",
                "entity_poly": {
                    "rcsb_mutation_count": mutation_count,
                    "rcsb_sample_sequence_length": 360,
                },
                "rcsb_polymer_entity": {"pdbx_mutation": None},
                "rcsb_polymer_entity_container_identifiers": {
                    "auth_asym_ids": ["A"],
                    "reference_sequence_identifiers": [
                        {
                            "database_name": "UniProt",
                            "database_accession": "Q16539",
                            "entity_sequence_coverage": 0.99,
                            "reference_sequence_coverage": 0.99,
                        }
                    ],
                },
            }
        ],
        "nonpolymer_entities": [
            {
                "rcsb_id": "TEST_2",
                "pdbx_entity_nonpoly": {"comp_id": ligand_id, "name": "inhibitor"},
                "rcsb_nonpolymer_entity": {"formula_weight": ligand_weight},
                "rcsb_nonpolymer_entity_container_identifiers": {
                    "auth_asym_ids": ["A"],
                    "nonpolymer_comp_id": ligand_id,
                },
            }
        ],
    }


def test_discovery_config_preserves_label_and_test_boundaries():
    config = load_config(CONFIG_PATH)
    preregistration = read_json(Path(config["preregistration"]["path"]))
    validate_preregistration(preregistration)

    assert preregistration["data_boundary"]["test"] == "locked_unreleased"
    assert not preregistration["data_boundary"][
        "labels_allowed_during_structural_selection"
    ]
    assert preregistration["pool_expansion"]["new_receptor_count"] == 4
    assert preregistration["future_validation"]["status"] == "not_authorized"


def test_search_request_uses_current_paginate_schema_and_frozen_target():
    payload = search_payload("Q16539", "X-RAY DIFFRACTION")

    assert payload["request_options"]["paginate"] == {"start": 0, "rows": 1000}
    assert "pager" not in payload["request_options"]
    values = [node["parameters"]["value"] for node in payload["query"]["nodes"]]
    assert values == ["Q16539", "X-RAY DIFFRACTION"]


def test_wild_type_high_resolution_same_chain_ligand_is_metadata_eligible():
    preregistration = read_json(
        Path(load_config(CONFIG_PATH)["preregistration"]["path"])
    )
    eligibility = preregistration["rcsb_discovery"]["metadata_eligibility"]
    row = normalize_entry(sample_entry(), "Q16539", eligibility, set())

    assert row["status"] == "metadata_eligible"
    assert row["selected_auth_chain"] == "A"
    assert row["qualifying_ligand_ids"] == "LIG"


def test_mutation_and_common_ion_are_fail_closed():
    preregistration = read_json(
        Path(load_config(CONFIG_PATH)["preregistration"]["path"])
    )
    eligibility = preregistration["rcsb_discovery"]["metadata_eligibility"]
    row = normalize_entry(
        sample_entry(mutation_count=1, ligand_id="SO4", ligand_weight=0.4),
        "Q16539",
        eligibility,
        set(),
    )

    assert row["status"] == "metadata_excluded"
    assert "mutation_count_differs" in row["exclusion_reasons"]
    assert "missing_same_chain_drug_like_nonpolymer" in row["exclusion_reasons"]
