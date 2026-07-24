#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

mode="${1:-gpu-all}"
case "$mode" in
  prepare|preflight|run|package|gpu-all) ;;
  *) echo "usage: $0 [prepare|preflight|run|package|gpu-all]" >&2; exit 2 ;;
esac

python_bin="${PYTHON_BIN:-python}"
base="${VINA_GPU_BASE:-/root/autodl-tmp/vina_gpu_stage06}"
source_original="${VINA_GPU_SOURCE_ORIGINAL:-$base/src/Vina-GPU-2.1}"
source_patched="${VINA_GPU_SOURCE_PATCHED:-$base/src/Vina-GPU-2.1-deterministic-batch-v1}"
engine_dir="$source_patched/AutoDock-Vina-GPU-2.1"
vinagpu_bin="$engine_dir/AutoDock-Vina-GPU-2-1"
patch_manifest="$source_patched/deterministic_batch_patch_manifest.json"
boost_dir="$base/src/boost_1_84_0"
config="configs/stage06_mk14_vinagpu21_deterministic_batch_bridge.json"
run_dir="results/runs/stage06_mk14_vinagpu21_deterministic_batch_bridge"
environment_dir="$run_dir/environment"
archive="${VINA_GPU_BATCH_RESULT_ARCHIVE:-/root/autodl-tmp/stage06_mk14_vinagpu21_deterministic_batch_bridge_core_v1.tar.gz}"

common_args=(
  --config "$config"
  --vinagpu "$vinagpu_bin"
  --opencl-binary-path "$engine_dir"
  --source-tree "$source_patched"
  --patch-manifest "$patch_manifest"
)

prepare_engine() {
  test -d "$source_original/.git"
  test -s "$source_original/AutoDock-Vina-GPU-2.1/Kernel1_Opt.bin"
  test -s "$source_original/AutoDock-Vina-GPU-2.1/Kernel2_Opt.bin"
  test -d "$boost_dir"
  if [[ ! -e "$source_patched" ]]; then
    cp -a "$source_original" "$source_patched"
  fi
  "$python_bin" scripts/experimental/vinagpu/apply_deterministic_batch_patch.py \
    --source-tree "$source_patched" \
    --manifest-output "$patch_manifest"
  sed -i \
    -e "s|^WORK_DIR=.*|WORK_DIR=$engine_dir|" \
    -e "s|^BOOST_LIB_PATH=.*|BOOST_LIB_PATH=$boost_dir|" \
    -e "s|^OPENCL_LIB_PATH=.*|OPENCL_LIB_PATH=/usr/local/cuda|" \
    -e "s|^OPENCL_VERSION=.*|OPENCL_VERSION=-DOPENCL_3_0|" \
    -e "s|^GPU_PLATFORM=.*|GPU_PLATFORM=-DNVIDIA_PLATFORM|" \
    -e "s|^DOCKING_BOX_SIZE=.*|DOCKING_BOX_SIZE=-DSMALL_BOX|" \
    "$engine_dir/Makefile"
  cd "$engine_dir"
  if [[ -f AutoDock-Vina-GPU-2-1 ]]; then
    make clean
  fi
  make
  cd "$repo_root"
  test -x "$vinagpu_bin"
  sha256sum \
    "$vinagpu_bin" \
    "$engine_dir/Kernel1_Opt.bin" \
    "$engine_dir/Kernel2_Opt.bin" \
    "$patch_manifest"
  "$python_bin" scripts/experimental/vinagpu/run_vinagpu_deterministic_batch.py \
    --config "$config" --audit-only
  "$python_bin" scripts/experimental/vinagpu/run_vinagpu_deterministic_batch.py \
    "${common_args[@]}" --lock-runtime-only
  sync
}

record_gpu_environment() {
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
    "$patch_manifest" > "$environment_dir/runtime_files.sha256"
  "$vinagpu_bin" --version > "$environment_dir/vinagpu_version.txt"
  printf '%s\n' "$(ulimit -s)" > "$environment_dir/stack_size_kib.txt"
}

run_preflight() {
  record_gpu_environment
  "$python_bin" scripts/experimental/vinagpu/run_vinagpu_deterministic_batch.py \
    "${common_args[@]}" --preflight-only --resume
}

run_full() {
  record_gpu_environment
  "$python_bin" scripts/experimental/vinagpu/run_vinagpu_deterministic_batch.py \
    "${common_args[@]}" --resume
}

package_results() {
  "$python_bin" scripts/experimental/vinagpu/package_vinagpu_deterministic_batch_results.py \
    --output "$archive"
  sha256sum "$archive"
  sync
  du -sh "$base"
  echo "stage06_vinagpu_deterministic_batch_bridge_complete"
}

case "$mode" in
  prepare) prepare_engine ;;
  preflight) run_preflight ;;
  run) run_full ;;
  package) package_results ;;
  gpu-all) run_preflight; run_full; package_results ;;
esac
