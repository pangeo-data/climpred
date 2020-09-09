import cftime
import numpy as np
import pytest
import xarray as xr

from climpred.comparisons import HINDCAST_COMPARISONS, PM_COMPARISONS
from climpred.metrics import HINDCAST_METRICS, METRIC_ALIASES, PM_METRICS
from climpred.utils import get_metric_class

probabilistic_metrics_requiring_logical = [
    'brier_score',
    'discrimination',
    'reliability',
]

probabilistic_metrics_requiring_more_than_member_dim = [
    'rank_histogram',
    'discrimination',
    'reliability',
]


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
    category_edges = np.array([0, 0.5, 1])
    if metric in probabilistic_metrics_requiring_logical:

        def f(x):
            return x > 0.5

        metric_kwargs = {'logical': f}
    elif metric == 'threshold_brier_score':
        metric_kwargs = {'threshold': 0.5}
    elif metric == 'contingency':
        metric_kwargs = {
            'forecast_category_edges': category_edges,
            'observation_category_edges': category_edges,
            'score': 'accuracy',
        }
    elif metric == 'rps':
        metric_kwargs = {'category_edges': category_edges}
    else:
        metric_kwargs = {}
    if Metric.probabilistic:
        dim = (
            ['member', 'init']
            if metric in probabilistic_metrics_requiring_more_than_member_dim
            else 'member'
        )
        skill = pe.verify(metric=metric, comparison='m2c', dim=dim, **metric_kwargs)
    else:
        dim = 'init' if comparison == 'e2c' else ['init', 'member']
        skill = pe.verify(
            metric=metric, comparison=comparison, dim=dim, **metric_kwargs
        )
    if metric == 'contingency':
        assert (skill == 1).all()  # checks Contingency.accuracy
    else:
        assert skill == Metric.perfect


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
    category_edges = np.array([0, 0.5, 1])
    if metric in probabilistic_metrics_requiring_logical:

        def f(x):
            return x > 0.5

        metric_kwargs = {'logical': f}
    elif metric == 'threshold_brier_score':
        metric_kwargs = {'threshold': 0.5}
    elif metric == 'contingency':
        metric_kwargs = {
            'forecast_category_edges': category_edges,
            'observation_category_edges': category_edges,
            'score': 'accuracy',
        }
    elif metric == 'rps':
        metric_kwargs = {'category_edges': category_edges}
    else:
        metric_kwargs = {}
    if Metric.probabilistic:
        skill = he.verify(
            metric=metric,
            comparison='m2o',
            dim=['member', 'init']
            if metric in probabilistic_metrics_requiring_more_than_member_dim
            else 'member',
            alignment=alignment,
            **metric_kwargs
        )
    else:
        dim = 'member' if comparison == 'm2o' else 'init'
        skill = he.verify(
            metric=metric,
            comparison=comparison,
            dim=dim,
            alignment=alignment,
            **metric_kwargs
        )
    if metric == 'contingency':
        assert (skill == 1).all()  # checks Contingency.accuracy
    else:
        assert skill == Metric.perfect
