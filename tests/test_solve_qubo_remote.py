import json

from scripts.solve_qubo_remote import main


def test_remote_runner_defaults_to_dry_run(tmp_path, monkeypatch) -> None:
    qubo = tmp_path / "qubo.json"
    qubo.write_text(
        json.dumps(
            {
                "qubo_coefficients": {
                    "constant": 0,
                    "linear": {"a": -1},
                    "quadratic": {},
                }
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "remote.json"
    monkeypatch.setattr(
        "sys.argv",
        ["solve_qubo_remote.py", "--qubo-json", str(qubo), "--output", str(output)],
    )

    assert main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "dry_run"
