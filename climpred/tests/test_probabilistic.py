import numpy as np
import pytest
import xarray as xr
from scipy.stats import norm

from climpred.bootstrap import bootstrap_hindcast, bootstrap_perfect_model
from climpred.comparisons import (
    NON_PROBABILISTIC_PM_COMPARISONS,
    PROBABILISTIC_HINDCAST_COMPARISONS,
    PROBABILISTIC_PM_COMPARISONS,
)
from climpred.metrics import METRIC_ALIASES, PROBABILISTIC_METRICS
from climpred.prediction import compute_hindcast, compute_perfect_model

ITERATIONS = 2

probabilistic_metrics_requiring_logical = [
    "brier_score",
    "discrimination",
    "reliability",
]

probabilistic_metrics_requiring_more_than_member_dim = [
    "rank_histogram",
    "discrimination",
    "reliability",
]

xr.set_options(display_style="text")


@pytest.mark.parametrize("comparison", PROBABILISTIC_PM_COMPARISONS)
@pytest.mark.parametrize("metric", PROBABILISTIC_METRICS)
def test_compute_perfect_model_da1d_not_nan_probabilistic(
    PM_da_initialized_1d, PM_da_control_1d, metric, comparison
):
    """
    Checks that there are no NaNs on perfect model probabilistic metrics of 1D
    time series.
    """
    metric_kwargs = {"comparison": comparison, "metric": metric, "dim": "member"}
    if "threshold" in metric:
        metric_kwargs["threshold"] = 10.5
    if metric == "brier_score":

        def func(x):
            return x > 0

        metric_kwargs["logical"] = func

    actual = compute_perfect_model(PM_da_initialized_1d, PM_da_control_1d)
    actual = actual.isnull().any()
    assert not actual


@pytest.mark.parametrize("metric", PROBABILISTIC_METRICS)
@pytest.mark.parametrize("comparison", PROBABILISTIC_HINDCAST_COMPARISONS)
def test_compute_hindcast_probabilistic(hindcast_recon_1d_ym, metric, comparison):
    """
    Checks that compute hindcast works without breaking.
    """
    category_edges = np.array([0, 0.5, 1])
    if metric in probabilistic_metrics_requiring_logical:

        def f(x):
            return x > 0.5

        metric_kwargs = {"logical": f}
    elif metric == "threshold_brier_score":
        metric_kwargs = {"threshold": 0.5}
    elif metric == "contingency":
        metric_kwargs = {
            "forecast_category_edges": category_edges,
            "observation_category_edges": category_edges,
            "score": "accuracy",
        }
    elif metric == "rps":
        metric_kwargs = {"category_edges": category_edges}
    else:
        metric_kwargs = {}
    dim = (
        ["member", "init"]
        if metric in probabilistic_metrics_requiring_more_than_member_dim
        else "member"
    )
    res = hindcast_recon_1d_ym.verify(
        alignment="same_verif",
        comparison=comparison,
        metric=metric,
        dim=dim,
        **metric_kwargs,
    )["SST"]
    assert not res.isnull().all()


@pytest.mark.parametrize("comparison", PROBABILISTIC_PM_COMPARISONS)
@pytest.mark.parametrize("metric", PROBABILISTIC_METRICS)
def test_bootstrap_perfect_model_da1d_not_nan_probabilistic(
    PM_da_initialized_1d, PM_da_control_1d, metric, comparison
):
    """
    Checks that there are no NaNs on perfect model probabilistic metrics of 1D
    time series.
    """
    kwargs = {
        "comparison": comparison,
        "metric": metric,
    }
    category_edges = np.array([0, 0.5, 1])
    if metric in probabilistic_metrics_requiring_logical:

        def f(x):
            return x > 0.5

        kwargs["logical"] = f
    elif metric == "threshold_brier_score":
        kwargs["threshold"] = 0.5
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

    assert (
        not compute_perfect_model(PM_da_initialized_1d, PM_da_control_1d, **kwargs)
        .isnull()
        .all()
    )

    kwargs["iterations"] = ITERATIONS
    kwargs["resample_dim"] = "member"
    actual = bootstrap_perfect_model(PM_da_initialized_1d, PM_da_control_1d, **kwargs)
    for skill in ["initialized", "uninitialized"]:
        actualk = actual.sel(skill=skill, results="verify skill")
        actualk = actualk.isnull().all()
        assert not actualk


@pytest.mark.slow
@pytest.mark.parametrize("comparison", PROBABILISTIC_HINDCAST_COMPARISONS)
@pytest.mark.parametrize("metric", PROBABILISTIC_METRICS)
def test_bootstrap_hindcast_da1d_not_nan_probabilistic(
    hind_da_initialized_1d,
    hist_da_uninitialized_1d,
    observations_da_1d,
    metric,
    comparison,
):
    """
    Checks that there are no NaNs on hindcast probabilistic metrics of 1D
    time series.
    """
    metric_kwargs = {
        "comparison": comparison,
        "metric": metric,
        "dim": "member",
        "iterations": ITERATIONS,
        "alignment": "same_verif",
    }
    if "threshold" in metric:
        metric_kwargs["threshold"] = 0.5
    if metric == "brier_score":

        def func(x):
            return x > 0

        metric_kwargs["logical"] = func

    actual = bootstrap_hindcast(
        hind_da_initialized_1d,
        hist_da_uninitialized_1d,
        observations_da_1d,
        resample_dim="member",
    )
    for skill in ["initialized", "uninitialized"]:
        actualk = actual.sel(skill=skill, results="verify skill")
        if "init" in actualk.coords:
            actualk = actualk.mean("init")
        actualk = actualk.isnull().any()
        assert not actualk


def test_compute_perfect_model_da1d_not_nan_crpss_quadratic(
    PM_da_initialized_1d, PM_da_control_1d
):
    """
    Checks that there are no NaNs on perfect model metrics of 1D time series.
    """
    actual = (
        compute_perfect_model(
            PM_da_initialized_1d.isel(lead=[0]),
            PM_da_control_1d,
            comparison="m2c",
            metric="crpss",
            gaussian=False,
            dim="member",
        )
        .isnull()
        .any()
    )
    assert not actual


@pytest.mark.slow
def test_compute_perfect_model_da1d_not_nan_crpss_quadratic_kwargs(
    PM_da_initialized_1d, PM_da_control_1d
):
    """
    Checks that there are no NaNs on perfect model metrics of 1D time series.
    """
    actual = (
        compute_perfect_model(
            PM_da_initialized_1d.isel(lead=[0]),
            PM_da_control_1d,
            comparison="m2c",
            metric="crpss",
            gaussian=False,
            dim="member",
            tol=1e-6,
            xmin=None,
            xmax=None,
            cdf_or_dist=norm,
        )
        .isnull()
        .any()
    )
    assert not actual


@pytest.mark.slow
@pytest.mark.skip(reason="takes quite long")
def test_compute_hindcast_da1d_not_nan_crpss_quadratic(
    hind_da_initialized_1d, observations_da_1d
):
    """
    Checks that there are no NaNs on hindcast metrics of 1D time series.
    """
    actual = (
        compute_hindcast(
            hind_da_initialized_1d,
            observations_da_1d,
            comparison="m2o",
            metric="crpss",
            gaussian=False,
            dim="member",
        )
        .isnull()
        .any()
    )
    assert not actual


def test_hindcast_crpss_orientation(hind_da_initialized_1d, observations_da_1d):
    """
    Checks that CRPSS hindcast as skill score > 0.
    """
    actual = compute_hindcast(
        hind_da_initialized_1d,
        observations_da_1d,
        comparison="m2o",
        metric="crpss",
        dim="member",
    )
    if "init" in actual.coords:
        actual = actual.mean("init")
    assert not (actual.isel(lead=[0, 1]) < 0).any()


def test_pm_crpss_orientation(PM_da_initialized_1d, PM_da_control_1d):
    """
    Checks that CRPSS in PM as skill score > 0.
    """
    actual = compute_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        comparison="m2m",
        metric="crpss",
        dim="member",
    )
    if "init" in actual.coords:
        actual = actual.mean("init")
    assert not (actual.isel(lead=[0, 1]) < 0).any()


# test api
# Probabilistic PM metrics dont work with non-prob PM comparison m2e and e2c
@pytest.mark.parametrize("comparison", NON_PROBABILISTIC_PM_COMPARISONS)
@pytest.mark.parametrize("metric", PROBABILISTIC_METRICS)
def test_compute_pm_probabilistic_metric_non_probabilistic_comparison_fails(
    PM_da_initialized_1d, PM_da_control_1d, metric, comparison
):
    with pytest.raises(ValueError) as excinfo:
        compute_perfect_model(
            PM_da_initialized_1d,
            PM_da_control_1d,
            comparison=comparison,
            metric=metric,
        )
    assert f"Probabilistic metric `{metric}` requires comparison" in str(excinfo.value)


@pytest.mark.parametrize("metric", ["crps"])
def test_compute_hindcast_probabilistic_metric_e2o_fails(
    hind_da_initialized_1d, observations_da_1d, metric
):
    metric = METRIC_ALIASES.get(metric, metric)
    with pytest.raises(ValueError) as excinfo:
        compute_hindcast(
            hind_da_initialized_1d,
            observations_da_1d,
            comparison="e2o",
            metric=metric,
            dim="member",
        )
    assert f"Probabilistic metric `{metric}` requires" in str(excinfo.value)


def test_hindcast_verify_brier_logical(hindcast_recon_1d_ym):
    """Test that a probabilistic score requiring a binary observations and
    probability initialized inputs gives the same results whether passing logical
    as kwarg or mapping logical before for hindcast.verify()."""
    he = hindcast_recon_1d_ym

    def logical(ds):
        return ds > 0.5

    brier_logical_passed_as_kwarg = he.verify(
        metric="brier_score",
        comparison="m2o",
        logical=logical,
        dim="member",
        alignment="same_verif",
    )
    brier_logical_mapped_before_and_member_mean = (
        he.map(logical)
        .mean("member")
        .verify(metric="brier_score", comparison="e2o", dim=[], alignment="same_verif")
    )
    brier_logical_mapped_before_no_member_mean = he.map(logical).verify(
        metric="brier_score", comparison="m2o", dim="member", alignment="same_verif"
    )
    assert (
        brier_logical_mapped_before_and_member_mean == brier_logical_passed_as_kwarg
    ).all()
    assert (
        brier_logical_mapped_before_no_member_mean == brier_logical_passed_as_kwarg
    ).all()
