import json
from collections import Counter
from pathlib import Path

from scripts.prepare_stage43_pparg_md96_inputs import frozen_frames, read_csv
from scripts.experimental.unidock.run_stage43_pparg_md96_production import (
    FROZEN_PROFILE,
    FROZEN_SEEDS,
    batch_signatures,
    load_rescue_amendment,
    validate_config,
)


def test_stage43_config_freezes_transfer_without_protected_data() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/stage43_pparg_md96_rank_sensitive_replication.json").read_text(encoding="ascii"))
    validate_config(config)
    assert config["objective_freeze"]["source_objective_id"] == "bedroc20_rank_sensitive_pair_complementarity_v1"
    assert config["objective_freeze"]["weight_search_on_stage43_outcomes"] is False
    assert config["evidence_timing"]["fresh_validation_rows_permitted"] is False
    assert config["evidence_timing"]["test_rows_permitted"] is False
    assert FROZEN_PROFILE == ("enhanced", 1024, 80)
    assert len(FROZEN_SEEDS) == 3


def test_stage43_panel_is_balanced_and_reuses_only_stage32_frames() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/stage43_pparg_md96_rank_sensitive_replication.json").read_text(encoding="ascii"))
    rows = frozen_frames(config, root)
    assert len(rows) == 96
    assert Counter(int(row["start_index"]) for row in rows) == Counter({index: 12 for index in range(8)})
    assert Counter(row["evidence_role"] for row in rows) == Counter({"historical_stage32_reuse": 16, "new_stage43_docking": 80})
    assert {int(row["temporal_maximin_rank"]) for row in rows} == set(range(12))


def test_stage43_historical_score_coverage_is_exact() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/stage43_pparg_md96_rank_sensitive_replication.json").read_text(encoding="ascii"))
    frames = frozen_frames(config, root)
    history = {row["conformer_id"] for row in frames if row["evidence_role"] == "historical_stage32_reuse"}
    ligands = {row["ligand_id"] for row in read_csv(root / config["inputs"]["stage32_ligand_manifest"])}
    scores = read_csv(root / config["inputs"]["stage32_scores"])
    observed = {(row["seed_id"], row["receptor_id"], row["ligand_id"]) for row in scores}
    expected = {(seed, receptor, ligand) for seed, _ in FROZEN_SEEDS for receptor in history for ligand in ligands}
    assert len(scores) == 7680
    assert observed == expected


def test_stage43_rescue_is_limited_to_one_named_batch() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/stage43_pparg_md96_rank_sensitive_replication.json").read_text(encoding="ascii"))
    frames = frozen_frames(config, root)
    ligands = read_csv(root / config["inputs"]["stage32_ligand_manifest"])
    amendment = load_rescue_amendment(root)
    seed = next(row for row in config["inputs"]["seeds"] if row["seed_id"] == "seed2")
    trigger = next(row for row in frames if row["conformer_id"] == "PPARG_MD_00177_8CPI")
    ordinary = next(row for row in frames if row["conformer_id"] == "PPARG_MD_00196_8CPI")
    config_hash = "0" * 64
    assert len(batch_signatures(config_hash, seed, trigger, ligands, config["unidock"], amendment)) == 2
    assert len(batch_signatures(config_hash, seed, ordinary, ligands, config["unidock"], amendment)) == 1
    assert amendment["rescue"]["effective_base_seed"] == 20365532
    assert amendment["rescue"]["maximum_rescue_attempt_count"] == 1
