#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

mode="${1:-all}"
case "$mode" in
  lock|smoke|run|audit|all) ;;
  *) echo "usage: $0 [lock|smoke|run|audit|all]" >&2; exit 2 ;;
esac

python_bin="${PYTHON_BIN:-python}"
base="${VINA_GPU_BASE:-/root/autodl-tmp/vina_gpu_stage06}"
source_tree="${VINA_GPU_SOURCE_TREE:-$base/src/Vina-GPU-2.1}"
engine_dir="${VINA_GPU_ENGINE_DIR:-$source_tree/AutoDock-Vina-GPU-2.1}"
vinagpu_bin="${VINAGPU_BIN:-$engine_dir/AutoDock-Vina-GPU-2-1}"
config="configs/stage06_mk14_vinagpu21_train160_equivalence.json"
run_dir="results/runs/stage06_mk14_vinagpu21_train160_equivalence"
environment_dir="$run_dir/environment"
archive="${VINA_GPU_RESULT_ARCHIVE:-/root/autodl-tmp/stage06_mk14_vinagpu21_train160_core_v1.tar.gz}"

common_args=(
  --config "$config"
  --vinagpu "$vinagpu_bin"
  --opencl-binary-path "$engine_dir"
  --source-tree "$source_tree"
)

lock_runtime() {
  "$python_bin" scripts/experimental/vinagpu/run_vinagpu_equivalence.py \
    --config "$config" --audit-only
  "$python_bin" scripts/experimental/vinagpu/run_vinagpu_equivalence.py \
    "${common_args[@]}" --lock-runtime-only
}

record_gpu_environment() {
  mkdir -p /etc/OpenCL/vendors "$environment_dir"
  printf '%s\n' 'libnvidia-opencl.so.1' > /etc/OpenCL/vendors/nvidia.icd
  ulimit -s 8192
  nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv > "$environment_dir/nvidia_smi.csv"
  nvidia-smi > "$environment_dir/nvidia_smi_full.txt"
  clinfo -l > "$environment_dir/clinfo_platforms.txt"
  clinfo | grep -E \
    'Platform Name|Platform Version|Device Name|Device Version|Driver Version|Global memory size' \
    > "$environment_dir/clinfo_nvidia_details.txt"
  git -C "$source_tree" rev-parse HEAD > "$environment_dir/vinagpu_source_commit.txt"
  grep -E \
    '^(GPU_PLATFORM|OPENCL_VERSION|DOCKING_BOX_SIZE)=' \
    "$engine_dir/Makefile" > "$environment_dir/vinagpu_build_settings.txt"
  sha256sum \
    "$vinagpu_bin" \
    "$engine_dir/Kernel1_Opt.bin" \
    "$engine_dir/Kernel2_Opt.bin" \
    "$engine_dir/Makefile" > "$environment_dir/vinagpu_runtime_files.sha256"
  "$vinagpu_bin" --version > "$environment_dir/vinagpu_version.txt"
  printf '%s\n' "$(ulimit -s)" > "$environment_dir/stack_size_kib.txt"
}

run_smoke() {
  record_gpu_environment
  "$python_bin" scripts/experimental/vinagpu/run_vinagpu_equivalence.py \
    "${common_args[@]}" --smoke-only --resume
}

run_full() {
  record_gpu_environment
  "$python_bin" scripts/experimental/vinagpu/run_vinagpu_equivalence.py \
    "${common_args[@]}" --resume
}

audit_and_package() {
  "$python_bin" scripts/experimental/vinagpu/audit_vinagpu_equivalence.py \
    --config "$config" --overwrite
  "$python_bin" scripts/experimental/vinagpu/package_vinagpu_results.py \
    --output "$archive"
  sha256sum "$archive"
  sync
  du -sh "$base"
  echo "stage06_vinagpu_train160_complete"
}

case "$mode" in
  lock) lock_runtime ;;
  smoke) lock_runtime; run_smoke ;;
  run) lock_runtime; run_full ;;
  audit) audit_and_package ;;
  all) lock_runtime; run_smoke; run_full; audit_and_package ;;
esac
