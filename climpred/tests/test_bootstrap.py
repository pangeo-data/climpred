import time

import dask
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from climpred.bootstrap import (
    bootstrap_by_reshape,
    bootstrap_hindcast,
    bootstrap_perfect_model,
    bootstrap_uninit_pm_ensemble_from_control_cftime,
    my_quantile,
)
from climpred.comparisons import HINDCAST_COMPARISONS, PM_COMPARISONS

BOOTSTRAP = 2


@pytest.mark.parametrize('chunk', [True, False])
def test_dask_percentile_implemented_faster_xr_quantile(PM_da_control_3d, chunk):
    chunk_dim, dim = 'x', 'time'
    if chunk:
        chunks = {chunk_dim: 24}
        PM_da_control_3d = PM_da_control_3d.chunk(chunks).persist()
    start_time = time.time()
    actual = my_quantile(PM_da_control_3d, q=0.95, dim=dim)
    elapsed_time_my_quantile = time.time() - start_time

    start_time = time.time()
    expected = PM_da_control_3d.compute().quantile(q=0.95, dim=dim)
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
def test_bootstrap_PM_no_lazy_results(
    PM_da_initialized_3d, PM_da_control_3d, chunk, comparison
):
    if chunk:
        PM_da_initialized_3d = PM_da_initialized_3d.chunk({'lead': 2}).persist()
        PM_da_control_3d = PM_da_control_3d.chunk({'time': -1}).persist()
    else:
        PM_da_initialized_3d = PM_da_initialized_3d.compute()
        PM_da_control_3d = PM_da_control_3d.compute()
    s = bootstrap_perfect_model(
        PM_da_initialized_3d,
        PM_da_control_3d,
        bootstrap=BOOTSTRAP,
        comparison=comparison,
        metric='mse',
    )
    assert dask.is_dask_collection(s) == chunk


@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
@pytest.mark.parametrize('chunk', [True, False])
def test_bootstrap_hindcast_lazy(
    hind_da_initialized_1d,
    hist_da_uninitialized_1d,
    observations_da_1d,
    chunk,
    comparison,
):
    if chunk:
        hind_da_initialized_1d = hind_da_initialized_1d.chunk({'lead': 2}).persist()
        hist_da_uninitialized_1d = hist_da_uninitialized_1d.chunk(
            {'time': -1}
        ).persist()
        observations_da_1d = observations_da_1d.chunk({'time': -1}).persist()
    else:
        hind_da_initialized_1d = hind_da_initialized_1d.compute()
        hist_da_uninitialized_1d = hist_da_uninitialized_1d.compute()
        observations_da_1d = observations_da_1d.compute()
    s = bootstrap_hindcast(
        hind_da_initialized_1d,
        hist_da_uninitialized_1d,
        observations_da_1d,
        bootstrap=BOOTSTRAP,
        comparison=comparison,
        metric='mse',
    )
    assert dask.is_dask_collection(s) == chunk


@pytest.mark.parametrize('resample_dim', ['member', 'init'])
def test_bootstrap_hindcast_resample_dim(
    hind_da_initialized_1d, hist_da_uninitialized_1d, observations_da_1d, resample_dim
):
    bootstrap_hindcast(
        hind_da_initialized_1d,
        hist_da_uninitialized_1d,
        observations_da_1d,
        bootstrap=BOOTSTRAP,
        comparison='e2o',
        metric='mse',
        resample_dim=resample_dim,
    )


def test_bootstrap_uninit_pm_ensemble_from_control_cftime_annual_identical(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
):
    """Test ``bootstrap_uninit_pm_ensemble_from_control_cftime`` cftime identical to
    ``bootstrap_uninit_pm_ensemble_from_control`` for annual data."""
    cftime_res = bootstrap_uninit_pm_ensemble_from_control_cftime(
        PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
    )
    noncftime_res = bootstrap_uninit_pm_ensemble_from_control_cftime(
        PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
    )
    # lead and member identical
    for d in ['lead', 'member']:
        assert (cftime_res[d] == noncftime_res[d]).all()
    # init same size
    assert cftime_res['init'].size == noncftime_res['init'].size


def test_bootstrap_uninit_pm_ensemble_from_control_cftime_annual(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
):
    """Test bootstrap_uninit_pm_ensemble_from_control_cftime for annual data."""
    uninit = bootstrap_uninit_pm_ensemble_from_control_cftime(
        PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
    )
    # lead and member identical
    for d in ['lead', 'member']:
        assert (uninit[d] == PM_ds_initialized_1d_ym_cftime[d]).all()
    # init same size
    assert uninit['init'].size == PM_ds_initialized_1d_ym_cftime['init'].size


def test_bootstrap_uninit_pm_ensemble_from_control_cftime_monthly(
    PM_ds_initialized_1d_mm_cftime, PM_ds_control_1d_mm_cftime
):
    """Test bootstrap_uninit_pm_ensemble_from_control_cftime for monthly data."""
    uninit = bootstrap_uninit_pm_ensemble_from_control_cftime(
        PM_ds_initialized_1d_mm_cftime, PM_ds_control_1d_mm_cftime
    )
    # lead and member identical
    for d in ['lead', 'member']:
        assert (uninit[d] == PM_ds_initialized_1d_mm_cftime[d]).all()
    # init same size
    assert uninit['init'].size == PM_ds_initialized_1d_mm_cftime['init'].size


def test_bootstrap_uninit_pm_ensemble_from_control_cftime_daily(
    PM_ds_initialized_1d_dm_cftime, PM_ds_control_1d_dm_cftime
):
    """Test bootstrap_uninit_pm_ensemble_from_control_cftime for daily data."""
    uninit = bootstrap_uninit_pm_ensemble_from_control_cftime(
        PM_ds_initialized_1d_dm_cftime, PM_ds_control_1d_dm_cftime
    )
    # lead and member identical
    for d in ['lead', 'member']:
        assert (uninit[d] == PM_ds_initialized_1d_dm_cftime[d]).all()
    # init same size
    assert uninit['init'].size == PM_ds_initialized_1d_dm_cftime['init'].size


def test_bootstrap_by_reshape_dataset(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_dm_cftime
):
    res = bootstrap_by_reshape(
        PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_dm_cftime
    )
    assert res.lead.attrs['units'] == 'years'
    assert isinstance(res, xr.Dataset)


def test_bootstrap_by_reshape_dataarray(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_dm_cftime
):
    v = list(PM_ds_initialized_1d_ym_cftime.data_vars)[0]
    res = bootstrap_by_reshape(
        PM_ds_initialized_1d_ym_cftime[v], PM_ds_control_1d_dm_cftime[v]
    )
    assert res.lead.attrs['units'] == 'years'
    assert isinstance(res, xr.DataArray)


def test_bootstrap_by_reshape_chunked(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_dm_cftime
):
    res_chunked = bootstrap_by_reshape(
        PM_ds_initialized_1d_ym_cftime.chunk(), PM_ds_control_1d_dm_cftime.chunk()
    )
    assert dask.is_dask_collection(res_chunked)
    res_chunked = res_chunked.compute()
    res = bootstrap_by_reshape(
        PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_dm_cftime
    )
    for d in ['lead', 'member']:
        assert (res_chunked[d] == res[d]).all()
    # init same size
    assert res_chunked['init'].size == res['init'].size


def test_bootstrap_by_reshape_two_var_dataset(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_dm_cftime
):
    """Test test_bootstrap_by_reshape when init_pm and control two variable dataset."""
    PM_ds_initialized_1d_ym_cftime['sos'] = PM_ds_initialized_1d_ym_cftime['tos']
    PM_ds_control_1d_dm_cftime['sos'] = PM_ds_control_1d_dm_cftime['tos']
    res = bootstrap_by_reshape(
        PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_dm_cftime
    )
    assert len(list(res.data_vars)) == 2
