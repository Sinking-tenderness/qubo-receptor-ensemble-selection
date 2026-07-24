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

The single-pair v1 pilot completed all 2,400 pairs and preserved aggregate
scores well, but failed minimum per-group Spearman and the frozen 5x speed
threshold. Its formal status remains failed.

The deterministic-batch bridge is a follow-up execution diagnostic. It applies
an audited host-only patch that sorts staged ligands and resets the RNG for each
ligand. Every batch score and complete pose hash must exactly match v1 before a
speed result is accepted. A bridge pass still does not validate QUBO and does
not permit CPU and GPU docking matrices to be mixed.

The bridge passed with 2,400 exact score and pose-hash matches and 7.536x
throughput over the recorded 32-vCPU reference. The next bounded diagnostic
tests fixed search depths 16, 24, and 32 only on the two preregistered failing
seed1/receptor groups. It selects the first profile passing both rank and speed
gates and stops as soon as speed fails. A selected profile must still undergo a
complete uniform-depth Train-160 confirmation before any larger GPU run.
