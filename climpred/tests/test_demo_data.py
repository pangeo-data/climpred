import pytest

from climpred.tutorial import FILE_ALIAS_DICT, load_dataset

filepaths = list(FILE_ALIAS_DICT.keys())


@pytest.mark.parametrize('filepath', filepaths)
def test_open_dataset_locally(filepath):
    """Opens all files listed in file_alias_dict."""
    print(filepath)
    d = load_dataset(filepath)
    assert d.nbytes > 0


def test_load_datasets_empty():
    actual = load_dataset()
    assert actual is None
