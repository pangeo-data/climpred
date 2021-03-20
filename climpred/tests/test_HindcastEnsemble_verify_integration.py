import pytest
import xarray as xr

from climpred import HindcastEnsemble


comparison_dim = [
    ("m2o", "member"),
    ("m2o", ["init", "member"]),
    ("e2o", "init")
]
from climpred.metrics import DETERMINISTIC_HINDCAST_METRICS, PROBABILISTIC_METRICS
from climpred.comparisons import (
    PROBABILISTIC_HINDCAST_COMPARISONS,
)

references = [
    None,
    "uninitialized",
    "persistence",
    "climatology",
]

probabilistic_metrics_requiring_logical = [
    "brier_score",
    "discrimination",
    "reliability",
]

probabilistic_metrics_requiring_more_than_member_dim = [
    "rank_histogram",
    "discrimination",
    "reliability", # # TODO: RPS? use metrics.requires_member_dim?
]

import numpy as np
category_edges = np.array([-.5, 0.0, 0.5])
ITERATIONS = 2

v='SST'


@pytest.mark.parametrize("alignment", ['same_verifs','same_inits','maximize'])
@pytest.mark.parametrize("call", ['verify','bootstrap'])
@pytest.mark.parametrize("reference", references)
@pytest.mark.parametrize("comparison,dim", comparison_dim)
@pytest.mark.parametrize("metric", DETERMINISTIC_HINDCAST_METRICS)
def test_HindcastEnsemble_verify_bootstrap_deterministic_metrics(
    hindcast_hist_obs_1d, comparison, metric, dim, reference, call, alignment
):
    """
    Checks that HindcastEnsemble.verify() and HindcastEnsemble.bootstrap() for
    deterministic metrics is not all NaN.
    """
    he = hindcast_hist_obs_1d.isel(lead=[0, 1, 2], init=range(20))

    if metric == "contingency":
        metric_kwargs = {
            "forecast_category_edges": category_edges,
            "observation_category_edges": category_edges,
            "score": "accuracy",
        }
    else:
        metric_kwargs = {}
    # acc on dim member only is ill defined
    pearson_r_containing_metrics = [
        "pearson_r",
        "spearman_r",
        "pearson_r_p_value",
        "spearman_r_p_value",
        "spearman_r_eff_p_value","pearson_r_eff_p_value",
        "msess_murphy",
        "bias_slope",
        "conditional_bias",
        "std_ratio",
        "conditional_bias",
        "uacc",
        "effective_sample_size",
    ]
    if dim == "member" and metric in pearson_r_containing_metrics:
        dim = ["init", "member"]
    if 'eff' in metric:
        print('only test over init')
        dim = 'init'
    if call == 'bootstrap':
        metric_kwargs['iterations']=ITERATIONS

    print(f'HindcastEnsemble.{call}(dim={dim}, alignment="{alignment}", metric="{metric}", comparison="{comparison}", reference={reference})')
    actual = getattr(he, call)(
        comparison=comparison,
        metric=metric,
        dim=dim,
        reference=reference,
        alignment=alignment,
        **metric_kwargs
    )[v]
    print('actual',actual)

    if reference is None:
        reference=[]
    if 'init' not in dim:
        assert 'time' not in actual.dims, print('didnt expect to find time in ',actual.dims)
    if 'time' in actual.coords:
        for c in ['init','lead']:
            assert c in actual.coords['time'].coords, print(f'didnt find {c} in time.coords, found {actual.time.coords}')
    if call == 'bootstrap':
        actual = actual.sel(results='verify skill')
    # test only reference skill
    if 'skill' in actual.dims:
        actual=actual.drop_sel(skill='initialized')
    if metric in ["contingency"] or metric in pearson_r_containing_metrics:
        pass
    else:
        #alignment='maximize' ## TODO: remove
        if alignment=='same_inits':
            assert actual.notnull().all(), print('found any nan',actual)
        else:
            assert actual.notnull().any(), print('found all nan',actual)


@pytest.mark.parametrize("call", ['verify','bootstrap'])
@pytest.mark.parametrize("alignment", ['same_verifs','same_inits','maximize'])
@pytest.mark.parametrize("reference", references)
@pytest.mark.parametrize("metric", PROBABILISTIC_METRICS)
@pytest.mark.parametrize("comparison", PROBABILISTIC_HINDCAST_COMPARISONS)
def test_HindcastEnsemble_verify_bootstrap_probabilistic_metrics(
    hindcast_hist_obs_1d, metric, comparison, reference, alignment, call,
):
    """
    Checks that HindcastEnsemble.verify() and HindcastEnsemble.bootstrap() works
    without breaking for all probabilistic metrics.
    """
    he = hindcast_hist_obs_1d.isel(lead=[0, 1], init=range(10))

    if metric in probabilistic_metrics_requiring_logical:

        def f(x):
            return x > 0

        kwargs = {"logical": f}
    elif metric == "threshold_brier_score":
        kwargs = {"threshold": 0}
    elif metric == "contingency":
        kwargs = {
            "forecast_category_edges": category_edges,
            "observation_category_edges": category_edges,
            "score": "accuracy",
        }
    elif metric == "rps":
        kwargs = {"category_edges": category_edges}
    else:
        kwargs = {}
    dim = (
        ["member", "init"]
        if metric in probabilistic_metrics_requiring_more_than_member_dim
        else "member"
    )
    kwargs.update(
        {
            "comparison": comparison,
            "metric": metric,
            "dim": dim,
            "reference": reference,
            "alignment": alignment,
        }
    )
    if call == 'bootstrap':
        kwargs['iterations']=ITERATIONS

    print(f'HindcastEnsemble.{call}(dim={dim}, alignment="{alignment}", metric="{metric}", comparison="{comparison}", reference={reference})')
    actual = getattr(he, call)(**kwargs)["SST"]

    if reference is None:
        reference=[]
    if 'init' not in dim:
        assert 'time' not in actual.dims, print('didnt expect to find time in ',actual.dims)
    if 'time' in actual.coords:
        for c in ['init','lead']:
            assert c in actual.coords['time'].coords, print(f'didnt find {c} in time.coords, found {actual.time.coords}')
    if call == 'bootstrap':
        actual = actual.sel(results='verify skill')
    if 'skill' in actual.dims:
        actual=actual.drop_sel(skill='initialized')
    if metric in ['reliability','discrimination']:
        actual = actual.mean('forecast_probability')
        if metric=='discrimination':
            actual=actual.mean('event')
    elif metric in ['rank_histogram']:
        actual = actual.mean('rank')

    if metric in ['crpss_es'] and reference in ['persistence','climatology']:
        pass  # will find all 0 or NaNs
    else:
        #alignment='maximize'
        if alignment=='same_inits':
            assert actual.notnull().all(), print('found any nan',actual)
        else:
            assert actual.notnull().any(), print('found all nan',actual)
