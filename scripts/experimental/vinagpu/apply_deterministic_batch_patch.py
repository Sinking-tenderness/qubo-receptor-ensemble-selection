"""Apply the audited deterministic-batch patch to Vina-GPU 2.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SOURCE_COMMIT = "180272b8a5265d6ed9664178345933cebe2cd349"
PATCH_ID = "vinagpu21-sorted-per-ligand-seed-v1"
SOURCE_FILES = {
    "AutoDock-Vina-GPU-2.1/main/main.cpp": {
        "original_canonical_lf_sha256": (
            "8E3537E286E09770CB2A0F537B57D429BCDDC3FAF2B1CAD5716FF4CA582D8D06"
        ),
        "patched_canonical_lf_sha256": (
            "33696A3735539436506A50A557DF6E51C7EA8DD352B57CE81B1C5B0698D5ABF0"
        ),
    },
    "AutoDock-Vina-GPU-2.1/lib/main_procedure_cl.cpp": {
        "original_canonical_lf_sha256": (
            "1535F8FD2C2484DB9882816B5A0C9270FF360362B7DBFB0C62B5A545900C0137"
        ),
        "patched_canonical_lf_sha256": (
            "9B567A77A78D2294B3218D9196B98F5F552B53E91654C2B54E2978C9A4A7E30C"
        ),
    },
}


def canonical_lf_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest().upper()


def git_head(source_tree: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_tree), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot read source commit: {result.stdout.strip()}")
    return result.stdout.strip()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected one {label} replacement, observed {count}")
    return text.replace(old, new, 1)


def patch_main_cpp(text: str) -> str:
    text = replace_once(
        text,
        "#include <vector> // ligand paths\n#include <cmath>",
        "#include <algorithm>\n#include <vector> // ligand paths\n#include <cmath>",
        "algorithm include",
    )
    original = """\
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
    patched = """\
\t\t\t// Stable order is required for ligand_count-based deterministic seeds.
\t\t\tstd::vector<std::string> sorted_ligand_paths;
\t\t\tfor (const auto& entry : std::experimental::filesystem::directory_iterator(ligand_directory))
\t\t\t\tsorted_ligand_paths.push_back(entry.path().string());
\t\t\tstd::sort(sorted_ligand_paths.begin(), sorted_ligand_paths.end());
\t\t\tfor (const auto& ligand_path : sorted_ligand_paths) {
\t\t\t\tstd::vector<std::string> tmp = { ligand_path };
\t\t\t\tligand_names.push_back(tmp);
\t\t\t\tstd::string delimiter1 = ".pdbqt";
\t\t\t\tstd::string tmp3 = ligand_path.substr(0, ligand_path.find(delimiter1)) + "_out.pdbqt";
\t\t\t\tstd::string tmp4 = out_dir + tmp3.substr(ligand_directory.length(), tmp3.length());
\t\t\t\tout_names.push_back(tmp4);
\t\t\t}
"""
    return replace_once(text, original, patched, "sorted ligand path")


def patch_main_procedure(text: str) -> str:
    text = replace_once(
        text,
        "\trng generator(static_cast<rng::result_type>(seed));\n",
        "",
        "shared RNG removal",
    )
    return replace_once(
        text,
        "\ttry {\n\t\tmodel m = ms[ligand_count];\n",
        "\ttry {\n"
        "\t\t// Batch ligand i must reproduce a standalone run with seed + i.\n"
        "\t\trng generator(static_cast<rng::result_type>(seed + ligand_count));\n"
        "\t\tmodel m = ms[ligand_count];\n",
        "per-ligand RNG insertion",
    )


def apply_patch(source_tree: Path) -> dict[str, object]:
    source_tree = source_tree.resolve()
    observed_head = git_head(source_tree)
    if observed_head != SOURCE_COMMIT:
        raise ValueError(f"source commit differs: {observed_head}")

    observed_before = {
        relative: canonical_lf_sha256(source_tree / relative)
        for relative in SOURCE_FILES
    }
    expected_after = {
        relative: values["patched_canonical_lf_sha256"]
        for relative, values in SOURCE_FILES.items()
    }
    if observed_before == expected_after:
        pass
    else:
        expected_before = {
            relative: values["original_canonical_lf_sha256"]
            for relative, values in SOURCE_FILES.items()
        }
        if observed_before != expected_before:
            raise ValueError(
                "source file hashes are neither pristine nor the approved patch"
            )
        main_path = source_tree / "AutoDock-Vina-GPU-2.1/main/main.cpp"
        procedure_path = (
            source_tree
            / "AutoDock-Vina-GPU-2.1/lib/main_procedure_cl.cpp"
        )
        main_path.write_bytes(
            patch_main_cpp(main_path.read_text(encoding="utf-8")).encode("utf-8")
        )
        procedure_path.write_bytes(
            patch_main_procedure(
                procedure_path.read_text(encoding="utf-8")
            ).encode("utf-8")
        )

    observed_after = {
        relative: canonical_lf_sha256(source_tree / relative)
        for relative in SOURCE_FILES
    }
    if observed_after != expected_after:
        raise ValueError(f"patched source hashes differ: {observed_after}")
    return {
        "schema_version": "1.0",
        "status": "ok",
        "patch_id": PATCH_ID,
        "source_commit": SOURCE_COMMIT,
        "source_files": {
            relative: {
                "original_canonical_lf_sha256": SOURCE_FILES[relative][
                    "original_canonical_lf_sha256"
                ],
                "patched_canonical_lf_sha256": observed_after[relative],
            }
            for relative in sorted(SOURCE_FILES)
        },
        "behavioral_scope": [
            "sort virtual-screening ligand paths lexicographically",
            "seed batch ligand i with batch_seed plus ligand index",
        ],
        "opencl_kernel_sources_modified": False,
        "scoring_or_search_code_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    result = apply_patch(args.source_tree)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_bytes(
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("ascii")
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
