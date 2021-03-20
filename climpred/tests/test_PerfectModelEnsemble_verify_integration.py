import numpy as np
import pytest
import xarray as xr

from climpred.metrics import DETERMINISTIC_PM_METRICS, PROBABILISTIC_METRICS
from climpred.comparisons import PROBABILISTIC_PM_COMPARISONS

xr.set_options(display_style="text")

comparison_dim_PM = [
    ("m2m", "init"),
    ("m2m", "member"),
    ("m2m", ["init", "member"]),
    ("m2e", "init"),
    ("m2e", "member"),
    ("m2e", ["init", "member"]),
    ("m2c", "init"),
    ("m2c", "member"),
    ("m2c", ["init", "member"]),
    ("e2c", "init"),
]

references = [
None,
    "uninitialized",
    "persistence",
    "climatology",

]


category_edges = np.array([9.5, 10.0, 10.5])

ITERATIONS = 2

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

v='tos'

@pytest.mark.parametrize("call", ['verify','bootstrap'])
@pytest.mark.parametrize("reference", references)
@pytest.mark.parametrize("comparison,dim", comparison_dim_PM)
@pytest.mark.parametrize("metric", DETERMINISTIC_PM_METRICS)
def test_PerfectModel_verify_bootstrap_deterministic_metrics(
    perfectModelEnsemble_initialized_control, comparison, metric, dim, reference, call
):
    """
    Checks that PerfectModel.verify() and PerfectModel.bootstrap() for
    deterministic metrics is not NaN.
    """
    pm = perfectModelEnsemble_initialized_control.isel(lead=[0, 1, 2], init=range(6))

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
        "msess_murphy",
        "bias_slope",
        "conditional_bias",
        "std_ratio",
        "conditional_bias",
        "uacc",
    ]
    if dim == "member" and metric in pearson_r_containing_metrics:
        dim = ["init", "member"]

    if call=='bootstrap':
        metric_kwargs['iterations']=ITERATIONS

    print(f'PerfectModelEnsemble.{call}(dim={dim}, metric="{metric}", comparison="{comparison}", reference={reference})')
    actual = getattr(pm,call)(
        comparison=comparison,
        metric=metric,
        dim=dim,
        reference=reference,
        **metric_kwargs
    )[v]
    print('actual',actual)

    if reference is None:
        reference = []
    # test time not in skill
    if 'init' not in dim:
        assert 'time' not in actual.dims, print('didnt expect to find time in ',actual.dims)
    # test time 2-dimensional if in skill
    if 'time' in actual.coords:
        for c in ['init','lead']:
            assert c in actual.coords['time'].coords, print(f'didnt find {c} in time.coords, found {actual.time.coords}')
    if call=='bootstrap':
        actual=actual.sel(results="verify skill")
    # test only reference skill
    if 'skill' in actual.dims:
        actual=actual.drop_sel(skill='initialized')
    if metric in ['contingency','uacc']:
        pass
    elif metric in pearson_r_containing_metrics:
        if reference != []:
            pass
        # less strict here with all NaNs, pearson_r yields NaNs for climatology
        else:
            assert actual.notnull().any(), print('found all nans', actual)
    else:
        assert actual.notnull().all(), print('found any nans', actual)


@pytest.mark.parametrize("call", ['verify','bootstrap'])
@pytest.mark.parametrize("reference", references)
@pytest.mark.parametrize("comparison", PROBABILISTIC_PM_COMPARISONS)
@pytest.mark.parametrize("metric", PROBABILISTIC_METRICS)
def test_PerfectModelEnsemble_verify_bootstrap_probabilistic_metrics(
    perfectModelEnsemble_initialized_control, metric, comparison, reference, call
):
    """
    Checks that PerfectModelEnsemble.verify() and PerfectModelEnsemble.bootstrap() works without breaking for all probabilistic metrics.
    """
    pm = perfectModelEnsemble_initialized_control.isel(lead=range(3), init=range(5))
    kwargs = {
        "comparison": comparison,
        "metric": metric,
    }
    if metric in probabilistic_metrics_requiring_logical:

        def f(x):
            return x > 10

        kwargs["logical"] = f
    elif metric == "threshold_brier_score":
        kwargs["threshold"] = 10.0
    elif metric == "contingency":
        kwargs["forecast_category_edges"] = category_edges
        kwargs["observation_category_edges"] = category_edges
        kwargs["score"] = "accuracy"
    elif metric == "rps":
        kwargs["category_edges"] = category_edges
    dim = (
        ["member", "init"]
        if metric in probabilistic_metrics_requiring_more_than_member_dim
        else "member"
    )
    kwargs["dim"] = dim
    kwargs["reference"] = reference

    if call=='bootstrap':
        kwargs["iterations"] = ITERATIONS
        kwargs["resample_dim"] = "member"

    print(f'PerfectModelEnsemble.{call}({kwargs}')
    actual = getattr(pm,call)(**kwargs)[v]
    print('actual',actual)

    if reference is None:
        reference = []
    if call=='bootstrap':
        actual=actual.sel(results="verify skill")
    if metric in ['reliability','discrimination']:
        actual = actual.mean('forecast_probability')
        if metric=='discrimination':
            actual=actual.mean('event')
    elif metric in ['rank_histogram']:
        actual = actual.mean('rank')
    if metric in ["crpss_es"] and reference in ["climatology", "persistence"]:
        pass
    else:
        assert actual.notnull().all(), print('found all nans', actual)
