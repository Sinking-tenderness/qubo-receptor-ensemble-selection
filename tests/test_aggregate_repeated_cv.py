def test_aggregate_script_is_importable():
    from scripts import aggregate_repeated_cv

    assert callable(aggregate_repeated_cv.main)
