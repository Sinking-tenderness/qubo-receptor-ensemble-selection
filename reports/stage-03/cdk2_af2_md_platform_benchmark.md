# CDK2 AF2 MD Platform Benchmark

## Purpose

This benchmark estimates local OpenMM throughput for the already built 46,071-atom CDK2 AF2 apo-like solvated system. It uses 100 warmup steps and 1,000 timed 2 fs Langevin steps at 300 K, writes no trajectory, and is not an equilibration or sampling result.

## Controlled Inputs

- OpenMM: `8.5.2.dev-36a30cb` from the Conda `openmm=8.5.2` package.
- System: 46,071 atoms; Amber14SB, TIP3P-FB, PME; same generated System XML and solvated coordinates in both successful runs.
- Integrator: Langevin Middle, 2 fs, 300 K, friction 1/ps, seed 20260712.
- Warmup / timed steps: 100 / 1,000.
- No trajectory, minimization, NVT assessment, NPT assessment, or production sampling was performed.

## Results

| Platform | Precision | Throughput (steps/s) | Estimated ns/day | Result |
| --- | --- | ---: | ---: | --- |
| CPU, 8 threads | CPU numerical path | 13.2543 | 2.290338 | success |
| Intel Iris Xe OpenCL | mixed | n/a | n/a | Context creation failed |
| Intel Iris Xe OpenCL | single | 90.6032 | 15.656240 | success |

The OpenCL single-precision run is about 6.84 times faster than the eight-thread CPU run for this fixed benchmark. Linear extrapolation gives approximately 55 minutes for 600 ps and 3.1 hours for 2 ns. Real equilibration and production runtimes may differ because minimization, barostat moves, output, and changing system state add overhead.

## Precision Finding

OpenMM discovered `Intel(R) Iris(R) Xe Graphics` through Intel OpenCL Graphics. Its OpenCL `mixed` precision Context could not be created, whereas `single` precision succeeded. This limitation is explicitly retained rather than silently falling back. Single precision is allowed only as a candidate pilot platform until a short stability check verifies finite energies, temperature, pressure/density behaviour, and absence of gross structural instability.

The CPU and OpenCL potential energies after stochastic warmup are not expected to be bitwise identical, even with the same seed. Their finite values only show that neither short run produced a numerical explosion; they are not an accuracy comparison.

## Records

- CPU result: `data/stage03_cdk2_af2_md_pilot_cpu_benchmark.json`
- OpenCL single result: `data/stage03_cdk2_af2_md_pilot_opencl_single_benchmark.json`
- The failed mixed-precision attempt is documented here but has no generated result file because Context creation failed.

## Decision

Use Intel Iris Xe OpenCL single precision for the next short stability module, with CPU retained as a verification platform. Do not start the 2 ns pilot yet.
