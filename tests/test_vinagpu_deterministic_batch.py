import json
from pathlib import Path

from scripts.experimental.vinagpu.apply_deterministic_batch_patch import (
    patch_main_cpp,
    patch_main_procedure,
)
from scripts.experimental.vinagpu.run_vinagpu_deterministic_batch import (
    batch_command,
    build_bridge_summary,
    ligand_chunks,
    staged_ligand_name,
    validate_bridge_inputs,
)


CONFIG_PATH = Path(
    "configs/stage06_mk14_vinagpu21_deterministic_batch_bridge.json"
)


def test_bridge_inputs_are_complete_consumed_train_reference():
    config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))

    receptors, ligands, reference, audit = validate_bridge_inputs(
        Path.cwd().resolve(), config
    )

    assert len(receptors) == 5
    assert len(ligands) == 160
    assert len(reference) == 2400
    assert audit["chunk_size"] == 8
    assert audit["chunk_count"] == 300
    assert audit["frozen_v1_status"] == "gpu_equivalence_gate_failed"
    assert audit["validation_rows"] == 0
    assert audit["test_rows"] == 0


def test_ligand_chunks_and_names_preserve_global_seed_offsets():
    ligands = [
        {"ligand_id": f"L{index}", "seed_offset": str(index)}
        for index in range(16)
    ]

    chunks = ligand_chunks(ligands, 8)

    assert len(chunks) == 2
    assert [int(row["seed_offset"]) for row in chunks[1]] == list(range(8, 16))
    assert staged_ligand_name(chunks[1][0]) == "000008__L8.pdbqt"


def test_batch_command_uses_directory_mode_and_chunk_seed(tmp_path: Path):
    config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))

    command = batch_command(
        tmp_path / "vina-gpu",
        tmp_path / "kernels",
        tmp_path / "receptor.pdbqt",
        tmp_path / "inputs",
        tmp_path / "outputs",
        config["vinagpu"],
        20260809,
    )

    assert command[command.index("--seed") + 1] == "20260809"
    assert command[command.index("--thread") + 1] == "8000"
    assert "--ligand_directory" in command
    assert "--output_directory" in command
    assert "--ligand" not in command
    assert "--search_depth" not in command


def test_bridge_gate_requires_exact_outputs_and_both_speedups():
    config = json.loads(CONFIG_PATH.read_text(encoding="ascii"))

    passed = build_bridge_summary(config, 2400, 2400, 2400, 0.0, 3000.0)
    slow = build_bridge_summary(config, 2400, 2400, 2400, 0.0, 5000.0)
    mismatch = build_bridge_summary(config, 2400, 2399, 2400, 0.1, 3000.0)

    assert passed["status"] == "deterministic_batch_bridge_passed"
    assert slow["gate_checks"]["speedup_vs_recorded_32vcpu"]["passed"] is False
    assert mismatch["gate_checks"]["exact_score_matches"]["passed"] is False
    assert mismatch["gate_checks"]["maximum_absolute_score_delta"]["passed"] is False


def test_patch_sorts_paths_and_resets_rng_per_ligand():
    main = """\
#include <vector> // ligand paths
#include <cmath>
\t\t\tfor (const auto& entry : std::experimental::filesystem::directory_iterator(ligand_directory)) {
\t\t\t\tstd::vector<std::string> tmp = { entry.path().string() };
\t\t\t\tligand_names.push_back(tmp);
\t\t\t\tstd::string delimiter1 = ".pdbqt";
\t\t\t\tstd::string delimiter2 = ligand_directory;
\t\t\t\tstd::string tmp2 = entry.path().string();
\t\t\t\tstd::string tmp3 = tmp2.substr(0, tmp2.find(delimiter1)) + "_out.pdbqt";
\t\t\t\tstd::string tmp4 = out_dir + tmp3.substr(ligand_directory.length(), tmp3.length());
\t\t\t\tout_names.push_back(tmp4);
\t\t\t}
"""
    procedure = """\
\trng generator(static_cast<rng::result_type>(seed));
\ttry {
\t\tmodel m = ms[ligand_count];
"""

    patched_main = patch_main_cpp(main)
    patched_procedure = patch_main_procedure(procedure)

    assert "#include <algorithm>" in patched_main
    assert "std::sort(sorted_ligand_paths.begin(), sorted_ligand_paths.end())" in patched_main
    assert "seed + ligand_count" in patched_procedure
    assert patched_procedure.count("rng generator") == 1
