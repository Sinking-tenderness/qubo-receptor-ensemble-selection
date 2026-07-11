from scripts.evaluate_matrix_by_split import read_csv


def test_read_csv_rejects_empty_files(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    try:
        read_csv(path)
    except ValueError as exc:
        assert "empty CSV" in str(exc)
    else:
        raise AssertionError("empty CSV should be rejected")
