#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx qubo-receptor-md; then
  conda activate qubo-receptor-md
else
  conda env create -f environment/stage03_openmm.yml
  conda activate qubo-receptor-md
fi

echo "=== GPU ==="
nvidia-smi
python - <<'PY'
from openmm import Context, NonbondedForce, Platform, System, VerletIntegrator, unit
import openmm
print("openmm_version=" + openmm.version.version)
print("platforms=" + ",".join(Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())))
if not any(Platform.getPlatform(i).getName() == "CUDA" for i in range(Platform.getNumPlatforms())):
    raise SystemExit("CUDA platform is unavailable")

# Platform discovery alone does not detect a CUDA/PTX runtime mismatch.  Build a
# real context and execute one kernel before any long Stage28 work begins.
system = System()
system.addParticle(39.9)
system.addParticle(39.9)
force = NonbondedForce()
force.addParticle(0.0, 0.3, 0.1)
force.addParticle(0.0, 0.3, 0.1)
system.addForce(force)
integrator = VerletIntegrator(1.0 * unit.femtoseconds)
context = Context(
    system,
    integrator,
    Platform.getPlatformByName("CUDA"),
    {"Precision": "mixed"},
)
context.setPositions([[0, 0, 0], [0.4, 0, 0]] * unit.nanometer)
energy = context.getState(getEnergy=True).getPotentialEnergy()
print("cuda_context_smoke=ok")
print("cuda_context_energy=" + str(energy))
del context, integrator
PY

config="${STAGE28_CONFIG:-configs/stage28_pparg_multistart_md_ensemble.json}"
args=(--config "$config")
if [[ -n "${START_ID:-}" ]]; then
  args+=(--start-id "$START_ID" --skip-collect)
fi
python scripts/run_stage28_pparg_multistart_md.py "${args[@]}"

if [[ -z "${START_ID:-}" ]]; then
  output="${STAGE28_RESULT_ARCHIVE:-$(dirname "$root")/stage28_pparg_multistart_md_ensemble_core_v1.tar.gz}"
  python scripts/build_stage28_pparg_multistart_md_result_bundle.py \
    --root "$root" \
    --config "$config" \
    --output "$output"
  sync
  ls -lh "$output"
  sha256sum "$output"
fi
