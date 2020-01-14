import time

import dask
import pytest
from xarray.testing import assert_allclose

from climpred.bootstrap import bootstrap_hindcast, bootstrap_perfect_model, my_quantile
from climpred.constants import HINDCAST_COMPARISONS, PM_COMPARISONS
from climpred.tutorial import load_dataset


@pytest.fixture
def pm_da_ds1d():
    da = load_dataset('MPI-PM-DP-1D')
    da = da['tos'].isel(area=1, period=-1)
    return da


@pytest.fixture
def pm_da_control1d():
    da = load_dataset('MPI-control-1D')
    da = da['tos'].isel(area=1, period=-1)
    return da


@pytest.fixture
def initialized_da():
    da = load_dataset('CESM-DP-SST')['SST']
    da = da - da.mean('init')
    return da


@pytest.fixture
def uninitialized_da():
    da = load_dataset('CESM-LE')['SST']
    # add member coordinate
    da['member'] = range(1, 1 + da.member.size)
    da = da - da.mean('time')
    return da


@pytest.fixture
def observations_da():
    da = load_dataset('ERSST')['SST']
    da = da - da.mean('time')
    return da


@pytest.fixture
def ds3d():
    """ds3d"""
    ds3d = load_dataset('MPI-PM-DP-3D')['tos']
    return ds3d


@pytest.fixture
def control3d():
    """control 3d"""
    control3d = load_dataset('MPI-control-3D')['tos']
    return control3d


# @pytest.mark.parametrize('dims', (('x', 'time'), ('time', 'x')))
@pytest.mark.parametrize('chunk', [True, False])
def test_dask_percentile_implemented_faster_xr_quantile(control3d, chunk):
    chunk_dim, dim = 'x', 'time'
    # chunk_dim, dim = 'time', 'x' # fails, why?
    # chunk_dim, dim = dims
    print(f'chunk_dim={chunk_dim}, dim={dim}')
    if chunk:
        chunks = {chunk_dim: 24}
        control3d = control3d.chunk(chunks).persist()
    start_time = time.time()
    actual = my_quantile(control3d, q=0.95, dim=dim)
    elapsed_time_my_quantile = time.time() - start_time

    start_time = time.time()
    expected = control3d.compute().quantile(q=0.95, dim=dim)
    elapsed_time_xr_quantile = time.time() - start_time
    if chunk:
        assert dask.is_dask_collection(actual)
        assert not dask.is_dask_collection(expected)
        start_time = time.time()
        actual = actual.compute()
        elapsed_time_my_quantile = elapsed_time_my_quantile + time.time() - start_time
    else:
        assert not dask.is_dask_collection(actual)
        assert not dask.is_dask_collection(expected)
    assert actual.shape == expected.shape
    assert_allclose(actual, expected)
    actual.plot()
    expected.plot()
    print(
        elapsed_time_my_quantile,
        elapsed_time_xr_quantile,
        'my_quantile is',
        elapsed_time_xr_quantile / elapsed_time_my_quantile,
        'times faster than xr.quantile',
    )
    assert elapsed_time_xr_quantile > elapsed_time_my_quantile


@pytest.mark.parametrize('comparison', PM_COMPARISONS)
@pytest.mark.parametrize('chunk', [True, False])
def test_bootstrap_PM_no_lazy_results(ds3d, control3d, chunk, comparison):
    bootstrap = 2
    if chunk:
        ds3d = ds3d.chunk({'lead': 2}).persist()
        control3d = control3d.chunk({'time': -1}).persist()
    else:
        ds3d = ds3d.compute()
        control3d = control3d.compute()
    s = bootstrap_perfect_model(
        ds3d, control3d, bootstrap=bootstrap, comparison=comparison, metric='mse',
    )
    assert dask.is_dask_collection(s) == chunk


@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
@pytest.mark.parametrize('chunk', [True, False])
def test_bootstrap_hindcast_lazy(
    initialized_da, uninitialized_da, observations_da, chunk, comparison
):
    bootstrap = 2
    if chunk:
        initialized_da = initialized_da.chunk({'lead': 2}).persist()
        uninitialized_da = uninitialized_da.chunk({'time': -1}).persist()
        observations_da = observations_da.chunk({'time': -1}).persist()
    else:
        initialized_da = initialized_da.compute()
        uninitialized_da = uninitialized_da.compute()
        observations_da = observations_da.compute()
    s = bootstrap_hindcast(
        initialized_da,
        uninitialized_da,
        observations_da,
        bootstrap=bootstrap,
        comparison=comparison,
        metric='mse',
    )
    assert dask.is_dask_collection(s) == chunk
