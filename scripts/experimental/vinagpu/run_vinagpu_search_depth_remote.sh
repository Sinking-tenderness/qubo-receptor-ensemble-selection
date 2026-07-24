#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

mode="${1:-gpu-all}"
case "$mode" in
  audit|preflight|run|package|gpu-all) ;;
  *) echo "usage: $0 [audit|preflight|run|package|gpu-all]" >&2; exit 2 ;;
esac

python_bin="${PYTHON_BIN:-python}"
base="${VINA_GPU_BASE:-/root/autodl-tmp/vina_gpu_stage06}"
source_patched="${VINA_GPU_SOURCE_PATCHED:-$base/src/Vina-GPU-2.1-deterministic-batch-v1}"
engine_dir="$source_patched/AutoDock-Vina-GPU-2.1"
vinagpu_bin="$engine_dir/AutoDock-Vina-GPU-2-1"
patch_manifest="$source_patched/deterministic_batch_patch_manifest.json"
config="configs/stage06_mk14_vinagpu21_search_depth_diagnostic.json"
run_dir="results/runs/stage06_mk14_vinagpu21_search_depth_diagnostic"
environment_dir="$run_dir/environment"
archive="${VINA_GPU_SEARCH_DEPTH_RESULT_ARCHIVE:-/root/autodl-tmp/stage06_mk14_vinagpu21_search_depth_diagnostic_core_v1.tar.gz}"

common_args=(
  --config "$config"
  --vinagpu "$vinagpu_bin"
  --opencl-binary-path "$engine_dir"
  --source-tree "$source_patched"
  --patch-manifest "$patch_manifest"
)

record_gpu_environment() {
  test -x "$vinagpu_bin"
  test -s "$engine_dir/Kernel1_Opt.bin"
  test -s "$engine_dir/Kernel2_Opt.bin"
  test -s "$patch_manifest"
  mkdir -p /etc/OpenCL/vendors "$environment_dir"
  printf '%s\n' 'libnvidia-opencl.so.1' > /etc/OpenCL/vendors/nvidia.icd
  ulimit -s 8192
  nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv > "$environment_dir/nvidia_smi.csv"
  nvidia-smi > "$environment_dir/nvidia_smi_full.txt"
  clinfo -l > "$environment_dir/clinfo_platforms.txt"
  sha256sum \
    "$vinagpu_bin" \
    "$engine_dir/Kernel1_Opt.bin" \
    "$engine_dir/Kernel2_Opt.bin" \
    "$engine_dir/Makefile" \
    "$patch_manifest" > "$environment_dir/runtime_files.sha256"
  "$vinagpu_bin" --version > "$environment_dir/vinagpu_version.txt"
  printf '%s\n' "$(ulimit -s)" > "$environment_dir/stack_size_kib.txt"
}

audit_runtime() {
  "$python_bin" scripts/experimental/vinagpu/run_vinagpu_search_depth_diagnostic.py \
    --config "$config" --audit-only
  record_gpu_environment
  "$python_bin" scripts/experimental/vinagpu/run_vinagpu_search_depth_diagnostic.py \
    "${common_args[@]}" --lock-runtime-only
}

run_preflight() {
  record_gpu_environment
  "$python_bin" scripts/experimental/vinagpu/run_vinagpu_search_depth_diagnostic.py \
    "${common_args[@]}" --preflight-only --resume
}

run_ladder() {
  record_gpu_environment
  "$python_bin" scripts/experimental/vinagpu/run_vinagpu_search_depth_diagnostic.py \
    "${common_args[@]}" --resume
}

package_results() {
  "$python_bin" scripts/experimental/vinagpu/package_vinagpu_search_depth_results.py \
    --output "$archive"
  sha256sum "$archive"
  sync
  du -sh "$base"
  echo "stage06_vinagpu_search_depth_diagnostic_complete"
}

case "$mode" in
  audit) audit_runtime ;;
  preflight) run_preflight ;;
  run) run_ladder ;;
  package) package_results ;;
  gpu-all) audit_runtime; run_preflight; run_ladder; package_results ;;
esac
