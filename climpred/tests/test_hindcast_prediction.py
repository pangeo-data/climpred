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

# uacc is sqrt(MSSS), fails when MSSS negative
DETERMINISTIC_HINDCAST_METRICS.remove('uacc')


@pytest.fixture
def initialized_ds():
    da = load_dataset('CESM-DP-SST')
    da = da - da.mean('init')
    return da


@pytest.fixture
def initialized_ds_lead0():
    da = load_dataset('CESM-DP-SST')
    da = da - da.mean('init')
    # Change to a lead-0 framework
    da['init'] += 1
    da['lead'] -= 1
    return da


@pytest.fixture
def initialized_da():
    da = load_dataset('CESM-DP-SST')['SST']
    da = da - da.mean('init')
    return da


@pytest.fixture
def observations_ds():
    da = load_dataset('ERSST')
    da = da - da.mean('time')
    return da


@pytest.fixture
def observations_da():
    da = load_dataset('ERSST')['SST']
    da = da - da.mean('time')
    return da


@pytest.fixture
def reconstruction_ds():
    da = load_dataset('FOSI-SST')
    da = da - da.mean('time')
    return da


@pytest.fixture
def reconstruction_da():
    da = load_dataset('FOSI-SST')['SST']
    da = da - da.mean('time')
    return da


@pytest.fixture
def uninitialized_ds():
    da = load_dataset('CESM-LE')
    # add member coordinate
    da['member'] = np.arange(1, 1 + da.member.size)
    da = da - da.mean('time')
    return da


@pytest.fixture
def uninitialized_da():
    da = load_dataset('CESM-LE')['SST']
    # add member coordinate
    da['member'] = np.arange(1, 1 + da.member.size)
    da = da - da.mean('time')
    return da


@pytest.fixture
def hind_3d():
    da = load_dataset('CESM-DP-SST-3D')['SST'].isel(
        nlon=slice(0, 10), nlat=slice(0, 12)
    )
    da = da - da.mean('init')
    return da


@pytest.fixture
def fosi_3d():
    da = load_dataset('FOSI-SST-3D')['SST'].isel(nlon=slice(0, 10), nlat=slice(0, 12))
    da = da - da.mean('time')
    return da


@pytest.mark.skip(reason='less not properly implemented')
def test_compute_hindcast_less_m2o(initialized_da, reconstruction_da):
    """Test LESS m2o runs through."""
    actual = (
        compute_hindcast(
            initialized_da, reconstruction_da, metric='less', comparison='m2o'
        )
        .isnull()
        .any()
    )
    assert not actual


@pytest.mark.parametrize('metric', DETERMINISTIC_HINDCAST_METRICS)
@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
def test_compute_hindcast(initialized_ds, reconstruction_ds, metric, comparison):
    """
    Checks that compute hindcast works without breaking.
    """
    res = (
        compute_hindcast(
            initialized_ds, reconstruction_ds, metric=metric, comparison=comparison
        )
        .isnull()
        .any()
    )
    for var in res.data_vars:
        assert not res[var]


def test_compute_hindcast_lead0_lead1(
    initialized_ds, initialized_ds_lead0, reconstruction_ds
):
    """
    Checks that compute hindcast returns the same results with a lead-0 and lead-1
    framework.
    """
    res1 = compute_hindcast(
        initialized_ds, reconstruction_ds, metric='rmse', comparison='e2o'
    )
    res2 = compute_hindcast(
        initialized_ds_lead0, reconstruction_ds, metric='rmse', comparison='e2o'
    )
    assert (res1.SST.values == res2.SST.values).all()


@pytest.mark.parametrize('metric', DETERMINISTIC_HINDCAST_METRICS)
def test_persistence(initialized_da, reconstruction_da, metric):
    """
    Checks that compute persistence works without breaking.
    """
    res = (
        compute_persistence(initialized_da, reconstruction_da, metric=metric)
        .isnull()
        .any()
    )
    assert not res


def test_persistence_lead0_lead1(
    initialized_ds, initialized_ds_lead0, reconstruction_ds
):
    """
    Checks that compute persistence returns the same results with a lead-0 and lead-1
    framework.
    """
    res1 = compute_persistence(initialized_ds, reconstruction_ds, metric='rmse')
    res2 = compute_persistence(initialized_ds_lead0, reconstruction_ds, metric='rmse')
    assert (res1.SST.values == res2.SST.values).all()


def test_uninitialized(uninitialized_da, reconstruction_da):
    """
    Checks that compute uninitialized works without breaking.
    """
    res = (
        compute_uninitialized(
            uninitialized_da, reconstruction_da, metric='rmse', comparison='e2o'
        )
        .isnull()
        .any()
    )
    assert not res


def test_bootstrap_hindcast_da1d_not_nan(
    initialized_da, uninitialized_da, reconstruction_da
):
    """
    Checks that there are no NaNs on bootstrap hindcast of 1D da.
    """
    actual = bootstrap_hindcast(
        initialized_da,
        uninitialized_da,
        reconstruction_da,
        metric='rmse',
        comparison='e2o',
        sig=50,
        bootstrap=2,
    )
    actual_init_skill = actual.sel(kind='init', results='skill').isnull().any()
    assert not actual_init_skill
    actual_uninit_p = actual.sel(kind='uninit', results='p').isnull().any()
    assert not actual_uninit_p


@pytest.mark.parametrize('metric', ('AnomCorr', 'test', 'None'))
def test_compute_hindcast_metric_keyerrors(initialized_ds, reconstruction_ds, metric):
    """
    Checks that wrong metric names get caught.
    """
    with pytest.raises(KeyError) as excinfo:
        compute_hindcast(
            initialized_ds, reconstruction_ds, comparison='e2o', metric=metric
        )
    assert 'Specify metric from' in str(excinfo.value)


@pytest.mark.parametrize('comparison', ('ensemblemean', 'test', 'None'))
def test_compute_hindcast_comparison_keyerrors(
    initialized_ds, reconstruction_ds, comparison
):
    """
    Checks that wrong comparison names get caught.
    """
    with pytest.raises(KeyError) as excinfo:
        compute_hindcast(
            initialized_ds, reconstruction_ds, comparison=comparison, metric='mse'
        )
    assert 'Specify comparison from' in str(excinfo.value)


@pytest.mark.parametrize('metric', ('rmse', 'pearson_r'))
def test_compute_hindcast_dask_spatial(hind_3d, fosi_3d, metric):
    """Chunking along spatial dims."""
    # chunk over dims in both
    for dim in hind_3d.dims:
        if dim in fosi_3d.dims:
            step = 5
            res_chunked = compute_hindcast(
                hind_3d.chunk({dim: step}),
                fosi_3d.chunk({dim: step}),
                comparison='e2o',
                metric=metric,
            )
            # check for chunks
            assert dask.is_dask_collection(res_chunked)
            assert res_chunked.chunks is not None


@pytest.mark.skip(reason='not yet implemented')
@pytest.mark.parametrize('metric', ('rmse', 'pearson_r'))
def test_compute_hindcast_dask_climpred_dims(hind_3d, fosi_3d, metric):
    """Chunking along climpred dims if available."""
    step = 5
    for dim in CLIMPRED_DIMS:
        if dim in hind_3d.dims:
            hind_3d = hind_3d.chunk({dim: step})
        if dim in fosi_3d.dims:
            fosi_3d = fosi_3d.chunk({dim: step})
        res_chunked = compute_hindcast(
            hind_3d, fosi_3d, comparison='e2o', metric=metric
        )
        # check for chunks
        assert dask.is_dask_collection(res_chunked)
        assert res_chunked.chunks is not None


def test_compute_hindcast_CESM_3D_keep_coords(hind_3d, fosi_3d):
    """Test that no coords are lost in compute_hindcast with the CESM sample data."""
    s = compute_hindcast(hind_3d, fosi_3d)
    for c in hind_3d.drop('init').coords:
        assert c in s.coords


def test_bootstrap_hindcast_keeps_lead_units(
    initialized_da, uninitialized_da, observations_da
):
    """Test that lead units is kept in compute."""
    sig = 95
    units = 'years'
    initialized_da['lead'].attrs['units'] = units
    actual = bootstrap_hindcast(
        initialized_da,
        uninitialized_da,
        observations_da,
        metric='mse',
        bootstrap=2,
        comparison='e2o',
        sig=sig,
        dim='init',
    )
    assert actual.lead.attrs['units'] == units
