#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export STAGE28_CONFIG="configs/stage28b_pparg_md_ready_multistart_md_ensemble.json"
export STAGE28_RESULT_ARCHIVE="${STAGE28_RESULT_ARCHIVE:-$(dirname "$root")/stage28b_pparg_md_ready_multistart_md_ensemble_core_v1.tar.gz}"
exec bash "$root/scripts/run_stage28_pparg_multistart_md_remote.sh"
