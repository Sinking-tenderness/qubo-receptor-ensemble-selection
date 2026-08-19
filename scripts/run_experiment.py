"""Run the configurable local source-to-QUBO experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qubo_receptor_ensemble.experiment import FullExperimentRunner, validate_front_inputs
from qubo_receptor_ensemble.full_workflow import load_full_experiment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument(
            "--data-root",
            type=Path,
            default=None,
            help="External experiment data root; defaults to the config directory.",
        )
        command.add_argument("--from", dest="start_stage", default=None)
        command.add_argument("--to", dest="end_stage", default=None)
        if name == "run":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_full_experiment_config(args.config, data_root=args.data_root)
    if args.command == "validate":
        result = validate_front_inputs(config, start_stage=args.start_stage)
    else:
        result = FullExperimentRunner(config).run(
            dry_run=args.command == "plan",
            start_stage=args.start_stage,
            end_stage=args.end_stage,
            resume=bool(getattr(args, "resume", False)),
            overwrite=bool(getattr(args, "overwrite", False)),
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
