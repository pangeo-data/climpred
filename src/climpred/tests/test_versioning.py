import pytest

from climpred.versioning.print_versions import show_versions


@pytest.mark.parametrize("as_json", [True, False])
def test_show_versions(as_json):
    show_versions(as_json=as_json)
