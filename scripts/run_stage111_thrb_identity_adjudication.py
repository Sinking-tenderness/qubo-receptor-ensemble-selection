"""Adjudicate an incorrect historical DUD-E-to-UniProt mapping without overwriting it."""

from __future__ import annotations

import argparse
import csv
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


def verified(root: Path, record: dict[str, str]) -> Path:
    path = root / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"input hash mismatch: {record['path']}")
    return path


def run(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    inputs = {key: verified(root, value) for key, value in config["inputs"].items()}
    if any(value != 0 for value in config["data_boundary"].values()):
        raise ValueError("Stage111 crossed its data boundary")

    historical = json.loads(inputs["historical_config"].read_text(encoding="utf-8"))
    mapping = next(item for item in historical["candidate_targets"] if item["target_id"] == "THRB")
    expected_mapping = {
        key: config["historical_mapping"][key]
        for key in ("target_id", "protein_name", "uniprot_accession", "dude_target_url")
    }
    if any(mapping[key] != value for key, value in expected_mapping.items()):
        raise ValueError("historical THRB mapping differs")

    with inputs["historical_screen"].open(encoding="utf-8", newline="") as handle:
        screen = next(row for row in csv.DictReader(handle) if row["target_id"] == "THRB")
    if int(screen["metadata_eligible_count"]) != 19 or screen["uniprot_accession"] != "P10828":
        raise ValueError("historical THRB screen record differs")

    closure = json.loads(inputs["historical_closure"].read_text(encoding="utf-8"))
    if closure["status"] != "stage110_historical_candidate_pool_closed":
        raise ValueError("Stage110 closure record differs")

    source = config["authoritative_sources"]
    if (
        source["dude_catalog_description"] != "Thrombin"
        or source["rcsb_observed_title"] != "Thrombin Inhibitor Complex"
        or source["rcsb_uniprot_accession"] != "P00734"
    ):
        raise ValueError("authoritative THRB identity evidence differs")

    result = {
        "schema_version": "1.0",
        "status": "stage111_thrb_identity_mismatch_confirmed",
        "evidence_status": "metadata-only provenance correction; no source archive, ligand row, coordinate, docking, validation, or hardware outcome was accessed",
        "config": {"path": str(config_path).replace("\\", "/"), "sha256": sha256(root / config_path)},
        "historical_mapping": config["historical_mapping"],
        "authoritative_identity": source,
        "finding": "DUD-E target slug THRB denotes thrombin, while the historical screen queried thyroid hormone receptor beta P10828. The old 19-structure result is therefore not evidence about the DUD-E thrombin target.",
        "data_boundary": config["data_boundary"],
        "decision": {
            "historical_record_overwritten": False,
            "stage110_registry_requires_amendment": True,
            "thrombin_new_preregistration_authorized": True,
            "thrombin_source_download_authorized": False,
            "thrombin_coordinate_download_authorized": False,
            "thrombin_docking_authorized": False,
            "quantum_hardware_authorized": False,
            "next_action": "Create and freeze a separate thrombin (F2) preregistration with a wild-type DUD-E reference-identity gate, then conduct a new outcome-unseen RCSB metadata screen. Do not use the old THRB result as thrombin evidence.",
        },
    }
    outputs = config["outputs"]
    result_path = root / outputs["summary_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    report = "\n".join([
        "# Stage111: THRB 靶点身份裁决",
        "",
        "DUD-E 目录将 `THRB` 描述为 **thrombin**，参考 PDB 为 `1YPE`；RCSB 将 `1YPE` 标记为 **Thrombin Inhibitor Complex**，主蛋白 UniProt 为 `P00734`。",
        "",
        "历史 Stage47b 配置却把同一 DUD-E slug 配对为 thyroid hormone receptor beta（`P10828`），因此其 19 个结构的筛查结果属于错误蛋白，不能代表凝血酶。",
        "",
        "## 决定",
        "",
        "保留旧记录，不作覆盖。`THRB` 作为凝血酶候选被重新打开，但必须以新的、独立预注册的 thrombin/F2 协议从头进行来源身份和元数据审计；本阶段不授权任何下载、对接、验证或量子硬件计算。",
        "",
    ])
    report_path = root / outputs["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage111_thrb_identity_adjudication.json"))
    args = parser.parse_args()
    result = run(args.root.resolve(), args.config)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
