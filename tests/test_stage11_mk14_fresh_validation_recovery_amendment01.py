import json
from pathlib import Path

from scripts.experimental.unidock import (
    run_stage11_mk14_fresh_validation_recovery_amendment01 as recovery,
)
from scripts.experimental.unidock.run_unidock_gpu_equivalence import read_json


CONFIG = Path("configs/stage11_mk14_fresh_validation_unidock113_confirmation.json")
AMENDMENT = Path(
    "configs/stage11_mk14_fresh_validation_score_guard_amendment01.json"
)


def test_stage11_recovery_amendment_matches_the_frozen_source_config() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / CONFIG
    amendment_path = root / AMENDMENT
    amendment = recovery.validate_amendment(
        root,
        config_path,
        read_json(config_path),
        amendment_path,
    )
    assert amendment["original_score_guard_kcal_per_mol"] == 100.0
    assert amendment["amended_score_guard_kcal_per_mol"] == 1000.0
    assert amendment["failure_evidence"]["observed_score_kcal_per_mol"] == 172.351


def test_stage11_recovery_retains_raw_outlier_without_changing_signature_protocol(
    monkeypatch,
) -> None:
    observed = {}

    def fake_run_batch(
        root,
        paths,
        executable,
        receptor,
        ligands,
        protocol,
        seed_id,
        base_seed,
        signature,
    ):
        observed["execution_guard"] = protocol[
            "maximum_absolute_score_kcal_per_mol"
        ]
        return (
            [
                {"ligand_id": "outlier", "gpu_score": 172.351},
                {"ligand_id": "ordinary", "gpu_score": -8.0},
            ],
            {"status": "ok", "signature": signature},
        )

    monkeypatch.setattr(recovery, "ORIGINAL_RUN_BATCH", fake_run_batch)
    monkeypatch.setattr(
        recovery,
        "AMENDMENT_DESCRIPTOR",
        {"amendment_id": "amendment01"},
    )
    rows, summary = recovery.amended_run_batch(
        Path.cwd(),
        {},
        "unidock",
        {"conformer_id": "R"},
        [],
        {"maximum_absolute_score_kcal_per_mol": 100.0},
        "seed1",
        20260802,
        "frozen-signature",
    )

    assert observed["execution_guard"] == 1000.0
    assert rows[0]["gpu_score"] == 172.351
    assert rows[0]["score_outlier_over_original_guard"] is True
    assert rows[1]["score_outlier_over_original_guard"] is False
    assert summary["checkpoint_signature_score_guard_kcal_per_mol"] == 100.0
    assert summary["execution_score_guard_kcal_per_mol"] == 1000.0
    assert summary["score_outlier_over_original_guard_ligand_ids"] == ["outlier"]


def test_stage11_recovery_finalizer_collects_original_signatures_but_audits_1000(
    monkeypatch, tmp_path: Path
) -> None:
    observed = {}

    def fake_collect(root, config, receptors, ligands, config_sha256):
        observed["collection_guard"] = config["unidock"][
            "maximum_absolute_score_kcal_per_mol"
        ]
        return [], [], []

    def fake_finalize(
        root,
        config_path,
        config,
        receptors,
        ligands,
        input_audit,
        executable_info,
        executed_batches,
        resumed_batches,
        invocation_elapsed,
        selected_seed_ids,
        selected_receptor_ids,
    ):
        observed["final_guard"] = config["unidock"][
            "maximum_absolute_score_kcal_per_mol"
        ]
        recovery.production.stage09.collect_batches(
            root, config, receptors, ligands, "config-sha"
        )
        return {
            "status": "stage11_fresh_validation_unidock_matrix_ok",
            "frozen_protocol": dict(config["unidock"]),
        }

    monkeypatch.setattr(recovery, "ORIGINAL_COLLECT_BATCHES", fake_collect)
    monkeypatch.setattr(recovery, "ORIGINAL_FINALIZE", fake_finalize)
    monkeypatch.setattr(
        recovery,
        "AMENDMENT_DESCRIPTOR",
        {"amendment_id": "amendment01"},
    )
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="ascii")
    config = {
        "unidock": {"maximum_absolute_score_kcal_per_mol": 100.0},
        "outputs": {
            "summary_json": "summary.json",
            "progress_json": "progress.json",
        },
    }
    result = recovery.amended_finalize(
        tmp_path,
        config_path,
        config,
        [],
        [],
        {},
        None,
        0,
        8,
        0.0,
        [],
        [],
    )

    assert observed == {"final_guard": 1000.0, "collection_guard": 100.0}
    assert result["execution_score_guard_kcal_per_mol"] == 1000.0
    written = json.loads((tmp_path / "summary.json").read_text(encoding="ascii"))
    assert written["technical_amendment"]["amendment_id"] == "amendment01"
