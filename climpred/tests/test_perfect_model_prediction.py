import pytest
from climpred.loadutils import open_dataset
from climpred.bootstrap import bootstrap_perfect_model
from climpred.prediction import compute_perfect_model, compute_persistence_pm


xskillscore_metrics = ('pearson_r', 'rmse', 'mse', 'mae')
xskillscore_distance_metrics = ('rmse', 'mse', 'mae')
PM_only_metrics = ('nrmse', 'nmse', 'nmae')
PM_comparisons = ('e2c', 'm2c', 'm2e', 'm2m')
all_metrics = xskillscore_metrics + PM_only_metrics


@pytest.fixture
def PM_da_ds3d():
    da = open_dataset('MPI-PM-DP-3D')
    # Box in South Atlantic with no NaNs.
    da = da.isel(x=slice(0, 50), y=slice(125, 150))
    return da['tos']


@pytest.fixture
def PM_da_control3d():
    da = open_dataset('MPI-control-3D')
    da = da.isel(x=slice(0, 50), y=slice(125, 150))
    return da['tos']


@pytest.fixture
def PM_ds_ds3d():
    ds = open_dataset('MPI-PM-DP-3D')
    ds = ds.isel(x=slice(0, 50), y=slice(125, 150))
    return ds


@pytest.fixture
def PM_ds_control3d():
    ds = open_dataset('MPI-control-3D')
    ds = ds.isel(x=slice(0, 50), y=slice(125, 150))
    return ds


@pytest.fixture
def PM_da_ds1d():
    da = open_dataset('MPI-PM-DP-1D')
    da = da['tos']
    return da


@pytest.fixture
def PM_da_control1d():
    da = open_dataset('MPI-control-1D')
    da = da['tos']
    return da


@pytest.fixture
def PM_ds_ds1d():
    ds = open_dataset('MPI-PM-DP-1D')
    return ds


@pytest.fixture
def PM_ds_control1d():
    ds = open_dataset('MPI-control-1D')
    return ds


@pytest.mark.parametrize('comparison', PM_comparisons)
@pytest.mark.parametrize('metric', all_metrics)
def test_compute_perfect_model_da_not_nan(PM_da_ds3d, PM_da_control3d, metric,
                                          comparison):
    """
    Checks that there are no NaNs on perfect model comparison for DataArray.
    """
    actual = compute_perfect_model(PM_da_ds3d, PM_da_control3d, metric=metric,
                                   comparison=comparison).isnull().any()
    # most pythonic way to assert value is False.
    # https://stackoverflow.com/questions/14733883/
    # best-practice-for-python-assert-command-false
    assert not actual


@pytest.mark.parametrize('comparison', PM_comparisons)
@pytest.mark.parametrize('metric', all_metrics)
def test_compute_perfect_model_ds_not_nan(PM_ds_ds3d, PM_ds_control3d, metric,
                                          comparison):
    """
    Checks that there are no NaNs on perfect model comparison for Dataset.
    """
    actual = compute_perfect_model(PM_ds_ds3d, PM_ds_control3d, metric=metric,
                                   comparison=comparison).isnull().any()
    for var in actual.data_vars:
        assert not actual[var]


@pytest.mark.parametrize('metric', xskillscore_distance_metrics)
def test_compute_persistence_pm_ds_not_nan(PM_ds_ds1d, PM_ds_control1d,
                                           metric):
    """
    Checks that there are no NaNs on persistence forecast of 1D time series.
    """
    actual = compute_persistence_pm(PM_ds_ds1d, PM_ds_control1d,
                                    metric=metric).isnull().any()
    for var in actual.data_vars:
        assert not actual[var]


# NOTE: bootstrap_perfect_model breaks on 'm2m'. It should be added back into
# the test once that is resolved. Also, it breaks when any PM-only metrics are
# used, since `compute_persistence_pm` breaks on non-xskillscore metrics.
@pytest.mark.parametrize('comparison', PM_comparisons)
@pytest.mark.parametrize('metric', xskillscore_metrics)
def test_bootstrap_perfect_model_da_not_nan(PM_da_ds1d, PM_da_control1d,
                                            metric, comparison):
    """
    Checks that there are no NaNs on bootstrap of 1D time series.
    """
    actual = bootstrap_perfect_model(PM_da_ds1d, PM_da_control1d,
                                     metric=metric, comparison=comparison,
                                     sig=50, bootstrap=2).isnull().any()
    for var in actual.data_vars:
        assert not actual[var]
