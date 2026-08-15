# Stage 28 PPARG multi-start MD remote execution

This input package freezes eight hard-gate-passing starts from the pre-existing
classical max-min order. Each start runs 100 ps NVT, 500 ps NPT, and 3 ns NPT
production with a 20 ps frame interval. The expected combined pool is 1200
time-correlated frames.

## Run

Extract the input archive under the AutoDL data disk, then enter its repository
directory. The runner creates or reuses the `qubo-receptor-md` environment and
resumes any existing equilibration or production checkpoints.

The environment is pinned to the CUDA 12 build family. If an environment made
from an earlier unpinned bundle already exists, update it once before running:

```bash
conda install -n qubo-receptor-md -y -c conda-forge \
  "openmm=8.5.2" "cuda-version=12"
```

The remote runner creates and evaluates a real CUDA Context before starting MD;
listing the CUDA platform alone is not treated as a sufficient runtime check.

```bash
nohup bash scripts/run_stage28_pparg_multistart_md_remote.sh \
  > stage28_pparg_multistart_md.log 2>&1 &
echo $! | tee stage28.pid
```

Monitor with:

```bash
tail -f stage28_pparg_multistart_md.log
```

Resume after an interruption by running the same `nohup` command again. Completed
systems, equilibrations, productions, and trajectory QC steps are skipped.

For a single start only:

```bash
START_ID=PPARG_2GTK_reference \
  bash scripts/run_stage28_pparg_multistart_md_remote.sh
```

Single-start mode does not aggregate or package the ensemble. Run the normal
command after all starts finish to collect, audit, and package the full result.

## Boundary

This is a short multi-start structural sampling experiment, not evidence of MD
convergence. It does not read docking scores or ligand labels and does not start
quantum-hardware work.
