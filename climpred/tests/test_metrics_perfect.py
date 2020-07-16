import cftime
import pytest
import xarray as xr

from climpred.comparisons import HINDCAST_COMPARISONS, PM_COMPARISONS
from climpred.metrics import HINDCAST_METRICS, METRIC_ALIASES, PM_METRICS
from climpred.utils import get_metric_class


@pytest.mark.parametrize('how', ['constant', 'increasing_by_lead'])
@pytest.mark.parametrize('comparison', PM_COMPARISONS)
@pytest.mark.parametrize('metric', PM_METRICS)
def test_PerfectModelEnsemble_constant_forecasts(
    perfectModelEnsemble_initialized_control, metric, comparison, how
):
    """Test that PerfectModelEnsemble.verify() returns a perfect score for a perfectly
    identical forecasts."""
    pe = perfectModelEnsemble_initialized_control.isel(lead=[0, 1, 2])
    if how == 'constant':  # replaces the variable with all 1's
        pe = pe.apply(xr.ones_like)
    elif (
        how == 'increasing_by_lead'
    ):  # sets variable values to cftime index in days to have increase over time.
        pe = pe.apply(xr.zeros_like)
        pe._datasets['initialized'] = (
            pe._datasets['initialized'] + pe._datasets['initialized'].lead
        )
    # get metric and comparison strings incorporating alias
    metric = METRIC_ALIASES.get(metric, metric)
    Metric = get_metric_class(metric, PM_METRICS)
    if metric == 'brier_score':

        def f(x):
            return x > 0.5

        metric_kwargs = {'logical': f}
    elif metric == 'threshold_brier_score':
        metric_kwargs = {'threshold': 0.5}
    else:
        metric_kwargs = {'useless_kwargs': 'to_ignore'}
    if Metric.probabilistic:
        skill = pe.verify(metric=metric, comparison='m2c', **metric_kwargs)
    else:
        skill = pe.verify(metric=metric, comparison=comparison, **metric_kwargs)
    perfect_skill = Metric.perfect
    assert skill == perfect_skill


@pytest.mark.parametrize('alignment', ['same_inits', 'same_verif', 'maximize'])
@pytest.mark.parametrize('how', ['constant', 'increasing_by_lead'])
@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
@pytest.mark.parametrize('metric', HINDCAST_METRICS)
def test_HindcastEnsemble_constant_forecasts(
    hindcast_hist_obs_1d, metric, comparison, how, alignment
):
    """Test that HindcastEnsemble.verify() returns a perfect score for a perfectly
    identical forecasts."""
    he = hindcast_hist_obs_1d.isel(lead=[0, 1, 2])
    if how == 'constant':  # replaces the variable with all 1's
        he = he.apply(xr.ones_like)
    elif (
        how == 'increasing_by_lead'
    ):  # sets variable values to cftime index in days to have increase over time.
        he = he.apply(xr.ones_like)
        # set initialized values to init in cftime days
        units = 'days since 1900-01-01'
        he._datasets['initialized'] = he._datasets['initialized'] * xr.DataArray(
            cftime.date2num(he._datasets['initialized'].init, units), dims=['init']
        )
        # add initialized leads
        he._datasets['initialized'] = (
            he._datasets['initialized'] + he._datasets['initialized'].lead
        )
        # set uninitialized values to init in cftime days
        he._datasets['uninitialized'] = he._datasets['uninitialized'] * xr.DataArray(
            cftime.date2num(he._datasets['uninitialized'].time, units), dims=['time']
        )
        # set obs values to init in cftime days
        he._datasets['observations']['obs'] = he._datasets['observations'][
            'obs'
        ] * xr.DataArray(
            cftime.date2num(he._datasets['observations']['obs'].time, units),
            dims=['time'],
        )
    # get metric and comparison strings incorporating alias
    metric = METRIC_ALIASES.get(metric, metric)
    Metric = get_metric_class(metric, HINDCAST_METRICS)
    if metric == 'brier_score':

        def f(x):
            return x > 0.5

        metric_kwargs = {'logical': f}
    elif metric == 'threshold_brier_score':
        metric_kwargs = {'threshold': 0.5}
    else:
        metric_kwargs = {'useless_kwargs': 'to_ignore'}
    if Metric.probabilistic:
        skill = he.verify(
            metric=metric, comparison='m2o', alignment=alignment, **metric_kwargs
        )
    else:
        skill = he.verify(
            metric=metric, comparison=comparison, alignment=alignment, **metric_kwargs
        )
    perfect_skill = Metric.perfect
    assert skill == perfect_skill
