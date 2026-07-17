import hashlib
import tarfile

from scripts.build_stage05_mk14_remote_bundle import write_bundle


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_is_deterministic_and_manifest_is_linux_compatible(tmp_path) -> None:
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "data" / "input.txt").write_text("input\n", encoding="ascii")
    script = root / "scripts" / "run.sh"
    script.write_text("#!/usr/bin/env bash\ntrue\n", encoding="ascii")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    paths = ["scripts/run.sh", "data/input.txt"]

    first_result = write_bundle(root, first, paths)
    second_result = write_bundle(root, second, list(reversed(paths)))

    assert first_result["manifest_line_ending"] == "LF"
    assert sha256(first) == sha256(second)
    with tarfile.open(first, "r:gz") as archive:
        manifest = archive.extractfile("bundle_manifest.sha256").read()
        script_info = archive.getmember("scripts/run.sh")
    assert b"\r" not in manifest
    assert manifest.endswith(b"\n")
    assert script_info.mode == 0o755
