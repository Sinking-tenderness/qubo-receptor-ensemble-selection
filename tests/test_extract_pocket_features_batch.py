from scripts.extract_pocket_features_batch import safe_filename


def test_batch_safe_filename_removes_path_characters():
    assert safe_filename("CDK2/AF:1") == "CDK2_AF_1"
