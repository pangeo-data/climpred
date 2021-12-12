import numpy as np
import pytest
import xarray as xr

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

references = [
    "uninitialized",
    "persistence",
    "climatology",
    ["climatology", "uninitialized", "persistence"],
]
references_ids = [
    "uninitialized",
    "persistence",
    "climatology",
    "climatology, uninitialized, persistence",
]

xr.set_options(display_style="text")


@pytest.mark.parametrize("reference", references, ids=references_ids)
@pytest.mark.parametrize("metric", PROBABILISTIC_METRICS)
@pytest.mark.parametrize("comparison", PROBABILISTIC_HINDCAST_COMPARISONS)
def test_HindcastEnsemble_verify_bootstrap_probabilistic(
    hindcast_hist_obs_1d, metric, comparison, reference
):
    """
    Checks that HindcastEnsemble.verify() and HindcastEnsemble.bootstrap() works
    without breaking for all probabilistic metrics.
    """
    he = hindcast_hist_obs_1d.isel(lead=[0, 1], init=range(10))

    category_edges = np.array([-0.5, 0, 0.5])
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
    # verify()
    kwargs.update(
        {
            "comparison": comparison,
            "metric": metric,
            "dim": dim,
            "reference": reference,
            "alignment": "same_verifs",
        }
    )
    actual_verify = he.verify(**kwargs)["SST"]
    not actual_verify.isnull().all()

    # bootstrap()
    actual = he.bootstrap(iterations=3, **kwargs)["SST"]
    assert "dayofyear" not in actual.coords

    if isinstance(reference, str):
        reference = [reference]
    if len(reference) == 0:
        assert not actual.sel(results="verify skill").isnull().all()
    else:
        assert (
            not actual.sel(skill="initialized", results="verify skill").isnull().all()
        )
        for skill in reference:
            actual_skill = actual.sel(skill=skill, results="verify skill")
            if metric == "crpss_es" and skill in ["climatology", "persistence"]:
                pass
            else:
                assert not actual_skill.isnull().all()


@pytest.mark.parametrize("reference", references, ids=references_ids)
@pytest.mark.parametrize("comparison", PROBABILISTIC_PM_COMPARISONS)
@pytest.mark.parametrize("metric", PROBABILISTIC_METRICS)
def test_PerfectModelEnsemble_verify_bootstrap_not_nan_probabilistic(
    perfectModelEnsemble_initialized_control, metric, comparison, reference
):
    """Test PredictionEnsemble.verify/bootstrap() works for probabilistic metrics."""
    pm = perfectModelEnsemble_initialized_control.isel(lead=range(3), init=range(5))
    kwargs = {
        "comparison": comparison,
        "metric": metric,
    }
    category_edges = np.array([9.5, 10.0, 10.5])
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

    # verify()
    actual_verify = pm.verify(**kwargs)
    assert not actual_verify.tos.isnull().all()

    # bootstrap
    kwargs["iterations"] = ITERATIONS
    kwargs["resample_dim"] = "member"
    kwargs["reference"] = reference
    actual = pm.bootstrap(**kwargs).tos
    if isinstance(reference, str):
        reference = [reference]
    if len(reference) == 0:
        assert not actual.sel(results="verify skill").isnull().all()
    else:
        for skill in reference:
            actual_skill = actual.sel(skill=skill, results="verify skill")
            if metric == "crpss_es" and skill in ["climatology", "persistence"]:
                pass
            else:
                assert not actual_skill.isnull().all()


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
            hind_da_initialized_1d.isel(lead=[0, 1, 2], init=range(10)),
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
        hind_da_initialized_1d.isel(lead=range(3)),
        observations_da_1d,
        comparison="m2o",
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
    with pytest.raises(ValueError, match=f"Probabilistic metric `{metric}` requires"):
        compute_perfect_model(
            PM_da_initialized_1d,
            PM_da_control_1d,
            comparison=comparison,
            metric=metric,
        )


@pytest.mark.parametrize("metric", ["crps"])
def test_compute_hindcast_probabilistic_metric_e2o_fails(
    hind_da_initialized_1d, observations_da_1d, metric
):
    metric = METRIC_ALIASES.get(metric, metric)
    with pytest.raises(ValueError, match=f"Probabilistic metric `{metric}` requires"):
        compute_hindcast(
            hind_da_initialized_1d,
            observations_da_1d,
            comparison="e2o",
            metric=metric,
            dim="member",
        )


def test_HindcastEnsemble_rps_terciles(hindcast_hist_obs_1d):
    actual = hindcast_hist_obs_1d.isel(lead=range(3), init=range(10)).verify(
        metric="rps",
        comparison="m2o",
        dim=["member", "init"],
        alignment="same_verifs",
        category_edges=np.array([-0.5, 0.0, 0.5, 1]),
        reference="climatology",
    )  # todo really use terciles
    assert actual.notnull().all()
    rpss = 1 - actual.sel(skill="initialized") / actual.sel(skill="climatology")
    assert rpss.isel(lead=0) > 0


def test_hindcast_verify_brier_logical(hindcast_recon_1d_ym):
    """Test that a probabilistic score requiring a binary observations and
    probability initialized inputs gives the same results whether passing logical
    as kwarg or mapping logical before for hindcast.verify()."""
    he = hindcast_recon_1d_ym.isel(lead=range(3), init=range(10))

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


@pytest.mark.parametrize("aggregate", [True, False])
def test_rps_different_edges(hindcast_recon_1d_mm, aggregate):
    """Test that HindcastEnsemble.verify(metric='rps') can work with
    different category_edges for forecast and observations."""
    he = hindcast_recon_1d_mm

    if aggregate:
        he = he.smooth(dict(lead=3), how="mean")

    q = [1 / 3, 2 / 3]
    model_edges = (
        he.get_initialized()
        .groupby("init.month")
        .quantile(q=q, dim=["init", "member"])
        .rename({"quantile": "category_edge"})
    )
    obs_edges = (
        he.get_observations()
        .groupby("time.month")
        .quantile(q=q, dim="time")
        .rename({"quantile": "category_edge"})
    )
    rps = he.verify(
        metric="rps",
        dim=["member", "init"],
        alignment="maximize",
        comparison="m2o",
        category_edges=(obs_edges, model_edges),
    )
    assert "month" not in rps.dims
    assert list(rps.dims) == ["lead"]


@pytest.mark.parametrize("aggregate", [True, False])
def test_rps_one_edge(hindcast_recon_1d_mm, aggregate):
    """Test that HindcastEnsemble.verify(metric='rps') can work with same
    category_edges for forecast and observations."""
    he = hindcast_recon_1d_mm

    if aggregate:
        he = he.smooth(dict(lead=3), how="mean")

    q = [1 / 3, 2 / 3]
    obs_edges = (
        he.get_observations()
        .groupby("time.month")
        .quantile(q=q, dim="time")
        .rename({"quantile": "category_edge"})
    )
    rps = he.verify(
        metric="rps",
        dim=["member", "init"],
        alignment="maximize",
        comparison="m2o",
        category_edges=obs_edges,
    )
    assert "month" not in rps.dims
    assert list(rps.dims) == ["lead"]
