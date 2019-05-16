import pytest
from climpred.loadutils import FILE_ALIAS_DICT, open_dataset


filepaths = list(FILE_ALIAS_DICT.keys())


@pytest.mark.parametrize('filepath', filepaths)
def test_open_dataset_locally(filepath):
    """Opens all files listed in file_alias_dict."""
    print(filepath)
    d = open_dataset(filepath)
    assert d.nbytes > 0
