import dask
import numpy as np
import pytest

from climpred.bootstrap import bootstrap_hindcast
from climpred.constants import (
    CLIMPRED_DIMS,
    DETERMINISTIC_HINDCAST_METRICS,
    HINDCAST_COMPARISONS,
)
from climpred.prediction import (
    compute_hindcast,
    compute_persistence,
    compute_uninitialized,
)
from climpred.tutorial import load_dataset


@pytest.fixture
def initialized_ds():
    da = load_dataset('CESM-DP-SST')
    da = da - da.mean('init')
    return da


def test_drop_vars(initialized_ds):
    """Tests that ds.drop_vars() from version 0.14.1 of xarray works."""
    assert initialized_ds.drop_vars("lead")
