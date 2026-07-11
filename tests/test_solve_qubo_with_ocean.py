from scripts.solve_qubo_with_ocean import load_bqm


def test_load_bqm_reads_explicit_coefficients(tmp_path) -> None:
    path = tmp_path / "qubo.json"
    path.write_text(
        '{"qubo_coefficients": {"constant": 0, "linear": {"a": -1, "b": -1}, '
        '"quadratic": {"a__b": 2}}}',
        encoding="utf-8",
    )

    bqm, variables = load_bqm(path)

    assert variables == ["a", "b"]
    assert bqm.energy({"a": 1, "b": 0}) == -1
