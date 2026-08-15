# Stage86 Dirac-3 nonnegative-gauge rescue

Stage86 authorizes only one allocation preflight. The physical rescue job must
not be run until that preflight has been returned and reviewed.

## Local validation on Windows

Extract the bundle beside the existing Stage85 directory. Reuse the isolated
Stage85 QCI Python environment:

```powershell
cd stage86_nonnegative_gauge_dirac_rescue_external_input_v1

$PY = "..\stage85_mixed_radix_dirac_calibration_external_input_v1\.venv-stage85-qci\Scripts\python.exe"

& $PY scripts\experimental\quantum\run_stage86_nonnegative_gauge_dirac_rescue.py `
  local-validate --root .
```

The required status is `stage86_external_local_validation_ok` with 27 variables.
This step makes zero QCI queries and zero device submissions.

## Allocation-only preflight

Set the token only in the current PowerShell process:

```powershell
$secure = Read-Host "QCI Token" -AsSecureString
$env:QCI_TOKEN = [System.Net.NetworkCredential]::new("", $secure).Password

& $PY scripts\experimental\quantum\run_stage86_nonnegative_gauge_dirac_rescue.py `
  preflight --root . 2>&1 |
  Tee-Object stage86_qci_preflight.log
```

The preflight performs one allocation query and no device job. It requires an
unpaid allocation with at least 60 seconds. Return `preflight.json` and the log
for review. Do not run the rescue mode yet.

## Separately authorized rescue

After review, the frozen rescue consists of one 27-variable job with 25 samples
and relaxation schedule 1. Even a pass cannot authorize production or quantum
advantage claims.
