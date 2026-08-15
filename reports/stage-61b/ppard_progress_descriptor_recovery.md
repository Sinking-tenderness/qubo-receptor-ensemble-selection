# Stage61b progress descriptor recovery

All 87 Uni-Dock batches and 12,528 receptor-ligand-seed jobs completed. The first independent audit stopped because finalization changed the status inside `progress.json` after its descriptor had already been recorded in `summary.json`.

The approved amendment changes only the `progress_json` SHA-256 and size descriptor in the generated summary. It does not change the executed config, score rows, poses, batch checkpoints, docking parameters, or protected-data boundary. The unchanged independent auditor is then rerun before result packaging.
