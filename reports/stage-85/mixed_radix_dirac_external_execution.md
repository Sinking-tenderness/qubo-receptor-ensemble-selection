# Stage85 Dirac-3 external execution

## Scope

This package contains three exact-oracle mixed-radix integer quadratic
calibration instances. It does not authorize production, efficacy, speedup, or
quantum-advantage claims.

## Environment

Use the existing Stage79 QCI environment with Python 3.11 and `qci-client`
5.0.0. No GPU is required.

## 1. Extract and verify

```bash
tar -xzf stage85_mixed_radix_dirac_calibration_external_input_v1.tar.gz
cd stage85_mixed_radix_dirac_calibration_external_input_v1
sha256sum -c bundle_manifest.sha256
```

## 2. Local validation

This step does not contact QCI.

```bash
python scripts/experimental/quantum/run_stage85_mixed_radix_dirac_calibration.py \
  local-validate --root .
```

The required status is `stage85_external_local_validation_ok` for all three
instances.

## 3. Allocation-only preflight

Set the token only in the current shell. Never write it into a file or send it
through chat.

```bash
export QCI_TOKEN='replace-with-your-token'

python scripts/experimental/quantum/run_stage85_mixed_radix_dirac_calibration.py \
  preflight --root . \
  2>&1 | tee stage85_qci_preflight.log
```

This performs one authenticated allocation query and zero device jobs. Stop
after preflight and return these two files for review:

```text
external_results/stage85_mixed_radix_dirac_calibration/preflight.json
stage85_qci_preflight.log
```

The preflight requires an unpaid Dirac allocation with at least 150 seconds.

## 4. Device calibration

Do not run this command until the preflight has been reviewed. The explicit
flag is required and authorizes exactly three jobs, 25 samples each, relaxation
schedule 1.

```bash
python scripts/experimental/quantum/run_stage85_mixed_radix_dirac_calibration.py \
  calibration --root . --authorized-device-run \
  2>&1 | tee stage85_qci_calibration.log
```

The runner stops before a job if fewer than 30 free seconds remain and rejects
cumulative recorded device use above 150 seconds. A passing calibration still
does not authorize additional jobs.
