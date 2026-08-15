"""Close the historical target registry without making a global no-target claim."""

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


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def report(result: dict[str, Any]) -> str:
    entries = result["registry_entries"]
    table = ["| Candidate | Status under the historical protocol | Reason |", "| --- | --- | --- |"]
    table.extend(f"| {entry['target_id']} | {entry['status']} | {entry['reason']} |" for entry in entries)
    return "\n".join(
        [
            "# Stage110: 历史候选靶点池关闭审计",
            "",
            "## 范围",
            "",
            "本报告只关闭 Stage47/47b 的旧候选表。它不声称未来不能发现新靶点，也不把已完成蛋白的结果重新用于调参。",
            "",
            "## 结论",
            "",
            "在同一“先有至少 32 个野生型、可解释公开候选，再经坐标/重对接硬门保留至少 16 个受体，且来源参考合格”的协议下，旧候选表中没有剩余的、未使用且可启动的确认靶点。",
            "",
            *table,
            "",
            "## 后续边界",
            "",
            "如要继续确认性路线，必须先另行冻结一个新的候选发现集合与来源参考规则，再做结果盲元数据审计；不能回到这个表中换阈值、替换参考或按预期表现挑靶点。",
            "",
        ]
    )


def run(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads((root / config_path).read_text(encoding="utf-8"))
    paths = {key: verified(root, value) for key, value in config["inputs"].items()}
    if any(value != 0 for value in config["data_boundary"].values()):
        raise ValueError("Stage110 must not access new target data")
    initial = rows(paths["stage47_screen"])
    expanded = rows(paths["stage47b_screen"])
    if [row["target_id"] for row in initial] != ["ABL1", "AKT1", "KIF11"]:
        raise ValueError("Stage47 candidate registry differs")
    expanded_ids = [row["target_id"] for row in expanded]
    if expanded_ids != ["RXRA", "THRB", "PPARA", "PPARD", "ESR2"]:
        raise ValueError("Stage47b candidate registry differs")
    if any(row["eligible_for_selection"] == "True" for row in initial):
        raise ValueError("initial registry should contain no eligible candidate")

    selection = json.loads(paths["ppard_selection"].read_text(encoding="utf-8"))
    ppard = json.loads(paths["ppard_completion"].read_text(encoding="utf-8"))
    esr1 = json.loads(paths["esr1_coordinate_gate"].read_text(encoding="utf-8"))
    parp1 = json.loads(paths["parp1_catalytic_domain_gate"].read_text(encoding="utf-8"))
    esr2 = json.loads(paths["esr2_reference_gate"].read_text(encoding="utf-8"))
    if selection["selection"]["eligible_target_order"] != ["PPARA", "PPARD", "ESR2"]:
        raise ValueError("historical eligible order differs")
    if ppard["decision"]["fresh_validation_authorized"]:
        raise ValueError("PPARD did not close as a negative transfer test")
    if esr1["status"] != "stage20c_esr1_coordinate_pool_insufficient_stop":
        raise ValueError("ESR1 closure differs")
    if parp1["counts"]["metadata_eligible_count"] != 10:
        raise ValueError("PARP1 catalytic-domain feasibility differs")
    if esr2["gate_checks"]["gate_passes"]:
        raise ValueError("ESR2 reference gate differs")

    registry_entries = [
        {"target_id": "ABL1", "status": "ineligible", "reason": "3 metadata-eligible structures, below the 32-candidate intake gate"},
        {"target_id": "AKT1", "status": "ineligible", "reason": "5 metadata-eligible structures, below the 32-candidate intake gate"},
        {"target_id": "KIF11", "status": "ineligible", "reason": "0 metadata-eligible structures, below the 32-candidate intake gate"},
        {"target_id": "RXRA", "status": "ineligible", "reason": "131 DUD-E clustered actives, below the historical source-size gate"},
        {"target_id": "THRB", "status": "ineligible", "reason": "19 metadata-eligible structures, below the 32-candidate intake gate"},
        {"target_id": "PPARA", "status": "already used", "reason": "completed development branch; cannot be reused as an unseen confirmation target"},
        {"target_id": "PPARD", "status": "already used and closed", "reason": "prospective transferred-QUBO application gate failed; fresh validation remained locked"},
        {"target_id": "ESR2", "status": "reference No-Go", "reason": "DUD-E reference 2FSZ carries C334S, C369S, and C481S"},
        {"target_id": "ESR1", "status": "historical exploratory No-Go", "reason": "only 9 coordinate-eligible receptors after a label-blind coordinate audit"},
        {"target_id": "PARP1", "status": "historical exploratory No-Go", "reason": "10 catalytic-domain metadata-eligible candidates, below even the final 16-receptor gate"},
    ]
    result = {"schema_version": "1.0", "status": "stage110_historical_candidate_pool_closed", "scope": "Only the Stage47/47b registry and earlier recorded exploratory candidates; not a global target-discovery claim.", "protocol_gate": config["protocol_gate"], "registry_entries": registry_entries, "counts": {"initial_registry_candidates": len(initial), "expanded_registry_candidates": len(expanded), "remaining_outcome_unseen_protocol_eligible_candidates": 0}, "data_boundary": config["data_boundary"], "decision": {"historical_candidate_pool_closed": True, "new_target_source_intake_authorized": False, "new_coordinate_audit_authorized": False, "new_docking_authorized": False, "quantum_hardware_authorized": False, "next_action": "Before any further confirmation experiment, freeze a new candidate-discovery registry and source-reference integrity criteria independently of docking outcomes."}}
    output = config["outputs"]
    summary_path = root / output["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report_path = root / output["report_md"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/stage110_historical_candidate_pool_closure.json"))
    args = parser.parse_args()
    result = run(args.root.resolve(), args.config)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
