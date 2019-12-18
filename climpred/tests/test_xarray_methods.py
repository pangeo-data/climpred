import pytest

from climpred.tutorial import load_dataset


@pytest.fixture
def initialized_ds():
    da = load_dataset('CESM-DP-SST')
    da = da - da.mean('init')
    return da


def test_drop_vars(initialized_ds):
    """Tests that ds.drop_vars() from version 0.14.1 of xarray works."""
    assert initialized_ds.drop_vars('lead')
