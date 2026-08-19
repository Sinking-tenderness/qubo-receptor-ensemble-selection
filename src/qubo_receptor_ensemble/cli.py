"""Command-line interface for the canonical experiment pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import load_pipeline_config, verify_input_artifacts
from .pipeline import PipelineRunner, format_summary


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    _add_config_arguments(validate)

    plan = subparsers.add_parser("plan")
    _add_config_arguments(plan)
    plan.add_argument("--from", dest="start_stage", default=None)
    plan.add_argument("--to", dest="end_stage", default=None)

    run = subparsers.add_parser("run")
    _add_config_arguments(run)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--from", dest="start_stage", default=None)
    run.add_argument("--to", dest="end_stage", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_pipeline_config(args.config, args.root)
    if args.command == "validate":
        records = verify_input_artifacts(config)
        print(format_summary({"status": "valid", "inputs": records}))
        return 0

    dry_run = args.command == "plan" or bool(getattr(args, "dry_run", False))
    summary = PipelineRunner(config).run(
        dry_run=dry_run,
        start_stage=getattr(args, "start_stage", None),
        end_stage=getattr(args, "end_stage", None),
    )
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
