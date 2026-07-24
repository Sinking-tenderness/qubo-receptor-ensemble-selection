# AutoDock Vina-GPU 2.1 Pilot

This directory isolates the Stage 6 AutoDock Vina-GPU 2.1 experiment from the
official CPU Vina evidence and the historical Uni-Dock diagnostics.

The pilot uses 160 consumed training ligands, five receptors, and three seeds.
Each receptor-ligand pair is launched separately with the exact CPU reference
seed (`base_seed + seed_offset`). This avoids Vina-GPU 2.1 virtual-screening
mode's unsorted filesystem iteration and continuous batch RNG stream.

Execution has two gates:

1. A real MAPK14 receptor-ligand smoke test must pass with the frozen search
   box and precompiled `SMALL_BOX` OpenCL kernels.
2. All 2,400 pairs must complete before score/rank equivalence and throughput
   are audited against the frozen AutoDock Vina 1.2.7 CPU matrices.

A pass is train-only engine evidence. It does not validate QUBO and does not
permit CPU and GPU docking matrices to be mixed.
