"""Amend the Stage110 closure without changing the historical record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_verified(root: Path, record: dict[str, str]) -> dict[str, Any]:
    path = root / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"input hash mismatch: {record['path']}")
    return json.loads(path.read_text(encoding="utf-8"))


def run(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    if any(value != 0 for value in config["data_boundary"].values()):
        raise ValueError("Stage112 crossed its protected data boundary")

    closure = load_verified(root, config["inputs"]["stage110_closure"])
    adjudication = load_verified(root, config["inputs"]["stage111_identity_adjudication"])
    old_thrb = next(item for item in closure["registry_entries"] if item["target_id"] == "THRB")
    if closure["status"] != "stage110_historical_candidate_pool_closed":
        raise ValueError("Stage110 closure status differs")
    if old_thrb["status"] != "ineligible" or "19 metadata-eligible" not in old_thrb["reason"]:
        raise ValueError("historical THRB entry differs")
    if adjudication["status"] != "stage111_thrb_identity_mismatch_confirmed":
        raise ValueError("Stage111 identity correction is absent")
    identity = adjudication["authoritative_identity"]
    if identity["dude_catalog_description"] != "Thrombin" or identity["rcsb_uniprot_accession"] != "P00734":
        raise ValueError("corrected thrombin identity differs")

    result = {
        "schema_version": "1.0",
        "status": "stage112_historical_candidate_pool_amendment01_ok",
        "scope": "Amends only the THRB entry in the Stage110 historical registry closure; all other Stage110 entries remain historical records without reinterpretation.",
        "historical_record": {
            "stage110_file_preserved": True,
            "stage110_original_remaining_outcome_unseen_protocol_eligible_candidates": closure["counts"]["remaining_outcome_unseen_protocol_eligible_candidates"],
            "superseded_for_target_ids": ["THRB"]
        },
        "amended_thrb_entry": {
            "historical_target_interpretation": "Thyroid hormone receptor beta (P10828)",
            "historical_metadata_eligible_count": 19,
            "corrected_target_interpretation": "Thrombin (F2)",
            "corrected_uniprot_accession": identity["rcsb_uniprot_accession"],
            "corrected_reference_pdb": identity["rcsb_reference_pdb"],
            "corrected_metadata_eligible_count": None,
            "reason": "The historical value 19 was computed for P10828 and is not evidence about thrombin P00734. The corrected target has not yet undergone an outcome-unseen metadata screen."
        },
        "corrected_registry_state": {
            "remaining_outcome_unseen_candidates_requiring_new_preregistration": 1,
            "candidate_target_ids": ["THRB"],
            "candidate_target_names": ["Thrombin (F2)"],
            "protocol_eligibility_established": False
        },
        "data_boundary": config["data_boundary"],
        "decision": {
            "new_thrombin_preregistration_required": True,
            "new_target_source_download_authorized": False,
            "new_coordinate_audit_authorized": False,
            "new_docking_authorized": False,
            "quantum_hardware_authorized": False,
            "next_action": "Freeze a new thrombin/F2 preregistration with a wild-type source-reference identity gate. Only then conduct a result-unseen RCSB metadata screen; do not reuse the historical THRB count of 19."
        }
    }
    outputs = config["outputs"]
    summary_path = root / outputs["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    report = "\n".join([
        "# Stage112: 历史候选池闭合修正 01",
        "",
        "Stage111 已确认 DUD-E 的 `THRB` 是凝血酶（F2，`P00734`），而 Stage47b/Stage110 把它错误映射成甲状腺激素受体 beta（`P10828`）。因此旧的 19 个结构不能作为凝血酶的结构池证据。",
        "",
        "本修正不改写 Stage110：除 `THRB` 外其余条目保持原样。修正后，历史登记册中有 1 个需要从头预注册的未测试候选，即凝血酶；其是否满足 32 个元数据候选和 16 个最终受体门槛尚未知。",
        "",
        "下一步只能创建并冻结独立的 thrombin/F2 预注册，再做结果未知的来源身份与 RCSB 元数据审计。本阶段没有下载数据、没有对接、没有量子硬件任务。",
        "",
    ])
    report_path = root / outputs["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage112_historical_candidate_pool_amendment01.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.config), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
