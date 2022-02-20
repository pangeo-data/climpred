import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from climpred.comparisons import PM_COMPARISONS
from climpred.metrics import __ALL_METRICS__ as all_metrics, Metric, __pearson_r
from climpred.stats import rm_poly


def my_mse_function(forecast, verif, dim=None, **metric_kwargs):
    # function
    return ((forecast - verif) ** 2).mean(dim)


my_mse = Metric(
    name="mse",
    function=my_mse_function,
    positive=True,
    probabilistic=False,
    unit_power=2,
    long_name="MSE",
    aliases=["mSe", "<<<SE"],
)

ITERATIONS = 2


@pytest.mark.parametrize("comparison", PM_COMPARISONS)
def test_custom_metric_passed_to_verify(
    perfectModelEnsemble_initialized_control, comparison
):
    """Test custom metric in PerfectModelEnsemble.verify()."""
    kwargs = dict(comparison=comparison, dim="init")
    actual = perfectModelEnsemble_initialized_control.verify(metric=my_mse, **kwargs)

    expected = perfectModelEnsemble_initialized_control.verify(metric="mse", **kwargs)
    assert_allclose(actual, expected)


@pytest.mark.slow
def test_custom_metric_passed_to_bootstrap(perfectModelEnsemble_initialized_control):
    """Test custom metric in PerfectModelEnsemble.bootstrap."""
    comparison = "e2c"
    np.random.seed(42)
    kwargs = dict(
        comparison=comparison, iterations=ITERATIONS, dim="init", resample_dim="init"
    )
    actual = perfectModelEnsemble_initialized_control.bootstrap(metric=my_mse, **kwargs)

    expected = perfectModelEnsemble_initialized_control.bootstrap(
        metric="mse", **kwargs
    )

    assert_allclose(actual, expected, rtol=0.1, atol=1)


def test_Metric_display():
    summary = __pearson_r.__repr__()
    assert "Kind: deterministic" in summary.split("\n")[4]


def test_no_repeating_metric_aliases():
    """Tests that there are no repeating aliases for metrics, which would overwrite
    the earlier defined metric."""
    METRICS = []
    for m in all_metrics:
        if m.aliases is not None:
            for a in m.aliases:
                METRICS.append(a)
    duplicates = set([x for x in METRICS if METRICS.count(x) > 1])
    print(f"Duplicate metrics: {duplicates}")
    assert len(duplicates) == 0


def test_contingency(hindcast_hist_obs_1d):
    """Test contingency table perfect results."""
    hindcast = hindcast_hist_obs_1d
    hindcast = hindcast.map(xr.ones_like)
    category_edges = np.array([-0.5, 0.0, 0.5, 0.9, 1.1])
    metric_kwargs = {
        "forecast_category_edges": category_edges,
        "observation_category_edges": category_edges,
        "score": "table",
    }
    skill = hindcast.verify(
        metric="contingency",
        comparison="m2o",
        dim=["member", "init"],
        alignment="same_verifs",
        **metric_kwargs,
    ).SST

    assert skill.isel(observations_category=-1, forecasts_category=-1).notnull().all()
    assert (
        skill.isel(
            observations_category=slice(None, -1), forecasts_category=slice(None, -1)
        )
        == 0.0
    ).all()


def test_overconfident(hindcast_hist_obs_1d):
    """Test rank_histogram and less for overconfident/underdisperive."""
    hindcast = hindcast_hist_obs_1d.copy()
    hindcast = hindcast.map(rm_poly, dim="init_or_time", deg=2)
    hindcast._datasets["initialized"] *= 0.3  # make overconfident
    less = hindcast.verify(
        metric="less",
        comparison="m2o",
        dim=["member", "init"],
        alignment="same_verifs",
    ).SST

    rh = hindcast.verify(
        metric="rank_histogram",
        comparison="m2o",
        dim=["member", "init"],
        alignment="same_verifs",
    ).SST

    assert (
        rh.isel(rank=[0, -1]) > rh.isel(rank=rh["rank"].size // 2)
    ).all()  # outer ranks larger
    assert (less < 0).all()  # underdisperive: neg less


def test_underconfident(hindcast_hist_obs_1d):
    """Test rank_histogram and less for underconfident/overdisperive."""
    hindcast = hindcast_hist_obs_1d.copy()
    hindcast = hindcast.map(rm_poly, dim="init_or_time", deg=2)
    hindcast._datasets["initialized"] *= 30  # make underconfident
    less = hindcast.verify(
        metric="less",
        comparison="m2o",
        dim=["member", "init"],
        alignment="same_verifs",
    ).SST

    rh = hindcast.verify(
        metric="rank_histogram",
        comparison="m2o",
        dim=["member", "init"],
        alignment="same_verifs",
    ).SST

    assert (
        (rh.isel(rank=[0, -1]) < rh.isel(rank=rh["rank"].size // 2))
        .isel(lead=slice(-3, None))
        .all()
    )  # outer ranks smaller
    assert (less > 0).all()  # overdisperive: pos less
