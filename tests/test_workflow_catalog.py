from pathlib import Path

import pytest

from scripts.workflow import WORKFLOWS, main, repository_root


def test_catalog_paths_exist_and_names_are_normalized():
    root = repository_root()
    assert root == Path(__file__).resolve().parents[1]
    assert WORKFLOWS
    for name, workflow in WORKFLOWS.items():
        assert name == name.lower()
        assert "_" not in name
        assert (root / workflow.path).is_file(), name


def test_catalog_list_is_safe(capsys):
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "dock-vina" in output
    assert "fresh-validation-evaluate" in output


def test_restricted_workflow_cannot_be_launched():
    with pytest.raises(SystemExit, match="restricted"):
        main(["run", "fresh-validation-evaluate"])
