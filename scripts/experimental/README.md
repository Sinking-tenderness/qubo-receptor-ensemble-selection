# Experimental Scripts

These directories contain bounded method experiments, not supported production
workflow entry points. Their outputs must not be mixed with the official CPU
Vina score matrices unless a preregistered equivalence gate explicitly passes.

- `unidock/`: consumed-train Uni-Dock compatibility and equivalence work. The
  tested profiles failed the frozen CPU Vina equivalence gate.
- `vinagpu/`: consumed-train AutoDock Vina-GPU 2.1 compatibility, exact-seed
  execution, equivalence auditing, and deterministic evidence packaging.
