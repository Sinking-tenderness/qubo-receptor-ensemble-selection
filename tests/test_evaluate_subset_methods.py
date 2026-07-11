def test_subset_method_module_imports() -> None:
    from scripts import evaluate_subset_methods

    assert callable(evaluate_subset_methods.main)
