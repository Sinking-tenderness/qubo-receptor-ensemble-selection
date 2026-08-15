# Stage79 QCI Dirac-3 External Execution

## What is frozen

Stage79 contains six independently audited local move-QUBOs inherited from
Stage78. Each QCI payload has at most 40 binary variables and 780 quadratic
interactions. The all-zero vector is the classical warm solution. QCI receives
an offset-free, max-absolute-normalized float32 polynomial; every returned
solution is evaluated again with the original float64 BQM.

Local preparation made zero QCI queries and zero Dirac-3 submissions.

## 1. Obtain free Dirac-3 access

Open the official Dirac-3 page and select `Get started for free`:

https://quantumcomputinginc.com/products/commercial-products/dirac-3

Create the account, activate the free 10-minute cloud allocation, and obtain
the API token. The official client documentation expects the production API at
`https://api.qci-prod.com` and reports the allocation through
`client.get_allocations()`.

Do not paste the API token into chat, source code, a shell history file, or a
result archive. It must exist only as the process environment variable
`QCI_TOKEN`.

## 2. Upload and verify the execution bundle

Upload `stage79_qci_dirac3_local_move_qubo_poc_external_input_v1.tar.gz` to a
small Linux host. A GPU is not required. Two CPU cores, 4 GB RAM, 5 GB free
disk, Python 3.11, and outbound HTTPS are enough.

```bash
cd /root/autodl-tmp

sha256sum stage79_qci_dirac3_local_move_qubo_poc_external_input_v1.tar.gz

mkdir -p stage79_qci_dirac3_v1
tar -xzf stage79_qci_dirac3_local_move_qubo_poc_external_input_v1.tar.gz \
  -C stage79_qci_dirac3_v1

cd stage79_qci_dirac3_v1
sha256sum -c bundle_manifest.sha256
```

## 3. Create the isolated client environment

This environment is independent of Uni-Dock and OpenMM environments.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx qubo-receptor-stage79-qci; then
  conda activate qubo-receptor-stage79-qci
else
  conda env create -f environment/stage79_qci_dirac3.yml
  conda activate qubo-receptor-stage79-qci
fi

python - <<'PY'
import importlib.metadata as metadata
print("qci-client:", metadata.version("qci-client"))
print("dimod:", metadata.version("dimod"))
PY
```

Expected versions are `qci-client 5.0.0` and `dimod 0.12.22`.

## 4. Run allocation-only preflight

Set the token without printing it:

```bash
read -rsp "QCI token: " QCI_TOKEN
export QCI_TOKEN
echo

export STAGE79_PYTHON="$(command -v python)"

bash scripts/experimental/quantum/run_stage79_qci_preflight_remote.sh \
  2>&1 | tee stage79_qci_preflight.log
```

This step performs local validation and one `get_allocations()` query. It does
not upload a QUBO and does not submit a Dirac-3 job. It refuses a paid
allocation and requires at least 300 free seconds.

Return these two files for review before running the device:

- `stage79_qci_dirac3_preflight_results_v1.tar.gz`
- `stage79_qci_preflight.log`

## 5. Run calibration only after preflight review

Calibration uses one diagnostic instance, four relaxation schedules, and 25
samples per schedule. It consumes 100 of the planned 600 device samples.

```bash
export STAGE79_QCI_ACK=I_ACCEPT_STAGE79_QCI_DEVICE_USAGE

bash scripts/experimental/quantum/run_stage79_qci_device_remote.sh calibration \
  2>&1 | tee stage79_qci_calibration.log
```

Return the calibration archive and log. Do not run confirmation until the
selected schedule and remaining allocation are reviewed.

## 6. Run frozen confirmation after calibration review

Confirmation applies the selected schedule to two positive and three negative
instances, with 100 samples per instance.

```bash
bash scripts/experimental/quantum/run_stage79_qci_device_remote.sh confirmation \
  2>&1 | tee stage79_qci_confirmation.log
```

The runner writes the upload record, job body, raw response, and metrics after
each completed job. Re-running a phase reuses existing completed raw responses
instead of resubmitting those jobs.

## Interpretation

A pass means that Dirac-3 physically recovered both certified local
improvements while producing no false improvement on the three exact negative
controls. It is a cross-hardware feasibility result, not evidence of quantum
advantage or new drug discovery.
