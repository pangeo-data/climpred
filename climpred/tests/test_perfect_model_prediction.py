import dask
import pytest

from climpred.bootstrap import bootstrap_perfect_model
from climpred.constants import CLIMPRED_DIMS, DETERMINISTIC_PM_METRICS
from climpred.prediction import compute_perfect_model, compute_persistence
from climpred.tutorial import load_dataset

# uacc is sqrt(MSSS), fails when MSSS negative
DETERMINISTIC_PM_METRICS_LUACC = DETERMINISTIC_PM_METRICS.copy()
DETERMINISTIC_PM_METRICS_LUACC.remove('uacc')

# run less tests
PM_COMPARISONS = {'m2c': '', 'e2c': ''}


@pytest.fixture
def pm_da_ds1d():
    da = load_dataset('MPI-PM-DP-1D')
    da = da['tos'].isel(area=1, period=-1)
    return da


@pytest.fixture
def pm_da_ds1d_lead0():
    da = load_dataset('MPI-PM-DP-1D')
    da = da['tos'].isel(area=1, period=-1)
    # Convert to lead zero for testing
    da['lead'] -= 1
    da['init'] += 1
    return da


@pytest.fixture
def pm_da_control1d():
    da = load_dataset('MPI-control-1D')
    da = da['tos'].isel(area=1, period=-1)
    return da


@pytest.fixture
def pm_ds_ds1d():
    ds = load_dataset('MPI-PM-DP-1D').isel(area=1, period=-1)
    return ds


@pytest.fixture
def pm_ds_control1d():
    ds = load_dataset('MPI-control-1D').isel(area=1, period=-1)
    return ds


@pytest.fixture
def ds_3d_NA():
    """ds North Atlantic"""
    ds = load_dataset('MPI-PM-DP-3D')['tos'].sel(x=slice(120, 130), y=slice(50, 60))
    return ds


@pytest.fixture
def control_3d_NA():
    """control North Atlantic"""
    ds = load_dataset('MPI-control-3D')['tos'].sel(x=slice(120, 130), y=slice(50, 60))
    return ds


@pytest.mark.parametrize('metric', ('rmse', 'pearson_r'))
def test_pvalue_from_bootstrapping(pm_da_ds1d, pm_da_control1d, metric):
    """Test that pvalue of initialized ensemble first lead is close to 0."""
    sig = 95
    actual = (
        bootstrap_perfect_model(
            pm_da_ds1d,
            pm_da_control1d,
            metric=metric,
            bootstrap=5,
            comparison='e2c',
            sig=sig,
            dim='init',
        )
        .sel(kind='uninit', results='p')
        .isel(lead=0)
    )
    assert actual.values < 2 * (1 - sig / 100)


@pytest.mark.parametrize('metric', DETERMINISTIC_PM_METRICS_LUACC)
def test_compute_persistence_ds1d_not_nan(pm_ds_ds1d, pm_ds_control1d, metric):
    """
    Checks that there are no NaNs on persistence forecast of 1D time series.
    """
    actual = (
        compute_persistence(pm_ds_ds1d, pm_ds_control1d, metric=metric).isnull().any()
    )
    for var in actual.data_vars:
        assert not actual[var]


@pytest.mark.parametrize('metric', DETERMINISTIC_PM_METRICS_LUACC)
def test_compute_persistence_lead0_lead1(
    pm_da_ds1d, pm_da_ds1d_lead0, pm_da_control1d, metric
):
    """
    Checks that persistence forecast results are identical for a lead 0 and lead 1 setup
    """
    res1 = compute_persistence(pm_da_ds1d, pm_da_control1d, metric=metric)
    res2 = compute_persistence(pm_da_ds1d_lead0, pm_da_control1d, metric=metric)
    assert (res1.values == res2.values).all()


@pytest.mark.parametrize('comparison', PM_COMPARISONS)
@pytest.mark.parametrize('metric', DETERMINISTIC_PM_METRICS_LUACC)
def test_compute_perfect_model_da1d_not_nan(
    pm_da_ds1d, pm_da_control1d, comparison, metric
):
    """
    Checks that there are no NaNs on perfect model metrics of 1D time series.
    """
    actual = (
        compute_perfect_model(
            pm_da_ds1d, pm_da_control1d, comparison=comparison, metric=metric
        )
        .isnull()
        .any()
    )
    assert not actual


@pytest.mark.parametrize('comparison', PM_COMPARISONS)
@pytest.mark.parametrize('metric', DETERMINISTIC_PM_METRICS_LUACC)
def test_compute_perfect_model_lead0_lead1(
    pm_da_ds1d, pm_da_ds1d_lead0, pm_da_control1d, comparison, metric
):
    """
    Checks that metric results are identical for a lead 0 and lead 1 setup.
    """
    res1 = compute_perfect_model(
        pm_da_ds1d, pm_da_control1d, comparison=comparison, metric=metric
    )
    res2 = compute_perfect_model(
        pm_da_ds1d_lead0, pm_da_control1d, comparison=comparison, metric=metric
    )
    assert (res1.values == res2.values).all()


def test_bootstrap_perfect_model_da1d_not_nan(pm_da_ds1d, pm_da_control1d):
    """
    Checks that there are no NaNs on bootstrap perfect_model of 1D da.
    """
    actual = bootstrap_perfect_model(
        pm_da_ds1d,
        pm_da_control1d,
        metric='rmse',
        comparison='e2c',
        sig=50,
        bootstrap=2,
    )
    actual_init_skill = actual.sel(kind='init', results='skill').isnull().any()
    assert not actual_init_skill
    actual_uninit_p = actual.sel(kind='uninit', results='p').isnull().any()
    assert not actual_uninit_p


def test_bootstrap_perfect_model_ds1d_not_nan(pm_ds_ds1d, pm_ds_control1d):
    """
    Checks that there are no NaNs on bootstrap perfect_model of 1D ds.
    """
    actual = bootstrap_perfect_model(
        pm_ds_ds1d,
        pm_ds_control1d,
        metric='rmse',
        comparison='e2c',
        sig=50,
        bootstrap=2,
    )
    for var in actual.data_vars:
        actual_init_skill = actual[var].sel(kind='init', results='skill').isnull().any()
        assert not actual_init_skill
    for var in actual.data_vars:
        actual_uninit_p = actual[var].sel(kind='uninit', results='p').isnull().any()
        assert not actual_uninit_p


@pytest.mark.parametrize('metric', ('AnomCorr', 'test', 'None'))
def test_compute_perfect_model_metric_keyerrors(pm_da_ds1d, pm_da_control1d, metric):
    """
    Checks that wrong metric names get caught.
    """
    with pytest.raises(KeyError) as excinfo:
        compute_perfect_model(
            pm_da_ds1d, pm_da_control1d, comparison='e2c', metric=metric
        )
    assert 'Specify metric from' in str(excinfo.value)


@pytest.mark.parametrize('comparison', ('ensemblemean', 'test', 'None'))
def test_compute_perfect_model_comparison_keyerrors(
    pm_da_ds1d, pm_da_control1d, comparison
):
    """
    Checks that wrong comparison names get caught.
    """
    with pytest.raises(KeyError) as excinfo:
        compute_perfect_model(
            pm_da_ds1d, pm_da_control1d, comparison=comparison, metric='mse'
        )
    assert 'Specify comparison from' in str(excinfo.value)


@pytest.mark.parametrize('metric', ('rmse', 'pearson_r'))
@pytest.mark.parametrize('comparison', PM_COMPARISONS)
def test_compute_pm_dask_spatial(ds_3d_NA, control_3d_NA, comparison, metric):
    """Chunking along spatial dims."""
    # chunk over dims in both
    for dim in ds_3d_NA.dims:
        if dim in control_3d_NA.dims:
            step = 5
            res_chunked = compute_perfect_model(
                ds_3d_NA.chunk({dim: step}),
                control_3d_NA.chunk({dim: step}),
                comparison=comparison,
                metric=metric,
                dim='init',
            )
            # check for chunks
            assert dask.is_dask_collection(res_chunked)
            assert res_chunked.chunks is not None


@pytest.mark.parametrize('metric', ('rmse', 'pearson_r'))
@pytest.mark.parametrize('comparison', PM_COMPARISONS)
def test_compute_pm_dask_climpred_dims(ds_3d_NA, control_3d_NA, comparison, metric):
    """Chunking along climpred dims if available."""
    step = 5
    for dim in CLIMPRED_DIMS:
        if dim in ds_3d_NA.dims:
            ds_3d_NA = ds_3d_NA.chunk({dim: step})
        if dim in control_3d_NA.dims:
            control_3d_NA = control_3d_NA.chunk({dim: step})
        res_chunked = compute_perfect_model(
            ds_3d_NA, control_3d_NA, comparison=comparison, metric=metric, dim='init'
        )
        # check for chunks
        assert dask.is_dask_collection(res_chunked)
        assert res_chunked.chunks is not None


def test_bootstrap_perfect_model_keeps_lead_units(pm_da_ds1d, pm_da_control1d):
    """Test that lead units is kept in compute."""
    sig = 95
    units = 'years'
    pm_da_ds1d.lead.attrs['units'] = 'years'
    actual = bootstrap_perfect_model(
        pm_da_ds1d,
        pm_da_control1d,
        metric='mse',
        bootstrap=2,
        comparison='e2c',
        sig=sig,
        dim='init',
    )
    assert actual.lead.attrs['units'] == units
