import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal

from climpred.comparisons import (
    __ALL_COMPARISONS__ as all_comparisons,
    PM_COMPARISONS,
    PROBABILISTIC_PM_COMPARISONS,
    Comparison,
    __e2c,
    __m2c,
    __m2e,
    __m2m,
)
from climpred.metrics import PM_METRICS, __mse as metric
from climpred.prediction import compute_perfect_model
from climpred.utils import get_comparison_class, get_metric_class


def test_e2c(PM_da_initialized_1d):
    """Test ensemble_mean-to-control (which can be any other one member) (e2c)
    comparison basic functionality.

    Clean comparison: Remove one control member from ensemble to use as reference.
    Take the remaining member mean as forecasts."""
    ds = PM_da_initialized_1d
    aforecast, areference = __e2c.function(ds, metric=metric)

    control_member = ds.member.values[0]
    reference = ds.sel(member=control_member, drop=True)
    # drop the member being reference
    ds = ds.drop_sel(member=control_member)
    forecast = ds.mean("member")

    eforecast, ereference = forecast, reference
    # very weak testing on shape
    assert eforecast.size == aforecast.size
    assert ereference.size == areference.size

    assert_equal(eforecast, aforecast)
    assert_equal(ereference, areference)


def test_m2c(PM_da_initialized_1d):
    """Test many-to-control (which can be any other one member) (m2c) comparison basic
    functionality.

    Clean comparison: Remove one control member from ensemble to use as reference.
    Take the remaining members as forecasts."""
    ds = PM_da_initialized_1d
    aforecast, areference = __m2c.function(ds, metric=metric)

    control_member = ds.member.values[0]
    reference = ds.sel(member=control_member, drop=True)
    # drop the member being reference
    ds_dropped = ds.drop_sel(member=control_member)
    forecast, reference = xr.broadcast(ds_dropped, reference)

    eforecast, ereference = forecast, reference
    # very weak testing on shape
    assert eforecast.size == aforecast.size
    assert ereference.size == areference.size

    assert_equal(eforecast, aforecast)
    assert_equal(ereference, areference)


def test_m2e(PM_da_initialized_1d):
    """Test many-to-ensemble-mean (m2e) comparison basic functionality.

    Clean comparison: Remove one member from ensemble to use as reference.
    Take the remaining members as forecasts."""
    ds = PM_da_initialized_1d
    aforecast, areference = __m2e.function(ds, metric=metric)

    reference_list = []
    forecast_list = []
    for m in ds.member.values:
        forecast = ds.drop_sel(member=m).mean("member")
        reference = ds.sel(member=m, drop=True)
        forecast, reference = xr.broadcast(forecast, reference)
        forecast_list.append(forecast)
        reference_list.append(reference)
    reference = xr.concat(reference_list, "member")
    forecast = xr.concat(forecast_list, "member")
    forecast["member"] = np.arange(forecast.member.size)
    reference["member"] = np.arange(reference.member.size)

    eforecast, ereference = forecast, reference
    # very weak testing on shape
    assert eforecast.size == aforecast.size
    assert ereference.size == areference.size

    assert_equal(eforecast, aforecast)
    assert_equal(ereference, areference)


def test_m2m(PM_da_initialized_1d):
    """Test many-to-many (m2m) comparison basic functionality.

    Clean comparison: Remove one member from ensemble to use as reference. Take the
    remaining members as forecasts."""
    ds = PM_da_initialized_1d
    aforecast, areference = __m2m.function(ds, metric=metric)

    reference_list = []
    forecast_list = []
    for m in ds.member.values:
        forecast = ds.drop_sel(member=m)
        forecast["member"] = np.arange(1, 1 + forecast.member.size)
        reference = ds.sel(member=m, drop=True)
        forecast, reference = xr.broadcast(forecast, reference)
        reference_list.append(reference)
        forecast_list.append(forecast)
    supervector_dim = "forecast_member"
    reference = xr.concat(reference_list, supervector_dim)
    forecast = xr.concat(forecast_list, supervector_dim)
    reference[supervector_dim] = np.arange(reference[supervector_dim].size)
    forecast[supervector_dim] = np.arange(forecast[supervector_dim].size)
    eforecast, ereference = forecast, reference
    # very weak testing here
    assert eforecast.size == aforecast.size
    assert ereference.size == areference.size


@pytest.mark.parametrize("metric", ["crps", "mse"])
@pytest.mark.parametrize("comparison", PM_COMPARISONS)
def test_all(PM_da_initialized_1d, comparison, metric):
    metric = get_metric_class(metric, PM_METRICS)
    ds = PM_da_initialized_1d
    comparison = get_comparison_class(comparison, PM_COMPARISONS)
    forecast, obs = comparison.function(ds, metric=metric)
    assert not forecast.isnull().any()
    assert not obs.isnull().any()
    if not metric.probabilistic:
        # same dimensions for deterministic metrics
        assert forecast.dims == obs.dims
    else:
        if comparison.name in PROBABILISTIC_PM_COMPARISONS:
            # same but member dim for probabilistic
            assert set(forecast.dims) - set(["member"]) == set(obs.dims)


@pytest.mark.parametrize("metric", ("rmse", "pearson_r"))
def test_new_comparison_passed_to_compute(
    PM_da_initialized_1d, PM_da_control_1d, metric
):
    def my_m2me_comparison(ds, metric=None):
        """Identical to m2e but median."""
        reference_list = []
        forecast_list = []
        supervector_dim = "member"
        for m in ds.member.values:
            forecast = ds.drop_sel(member=m).median("member")
            reference = ds.sel(member=m).squeeze()
            forecast_list.append(forecast)
            reference_list.append(reference)
        reference = xr.concat(reference_list, supervector_dim)
        forecast = xr.concat(forecast_list, supervector_dim)
        forecast[supervector_dim] = np.arange(forecast[supervector_dim].size)
        reference[supervector_dim] = np.arange(reference[supervector_dim].size)
        return forecast, reference

    my_m2me_comparison = Comparison(
        name="m2me",
        function=my_m2me_comparison,
        probabilistic=False,
        hindcast=False,
    )

    actual = compute_perfect_model(
        PM_da_initialized_1d,
        PM_da_control_1d,
        comparison=my_m2me_comparison,
        metric=metric,
    )

    expected = compute_perfect_model(
        PM_da_initialized_1d, PM_da_control_1d, comparison="m2e", metric="mse"
    )

    assert (actual - expected).mean() != 0


def test_Comparison_display():
    summary = __m2m.__repr__()
    assert "Kind: deterministic and probabilistic" in summary.split("\n")[2]


def test_no_repeating_comparison_aliases():
    """Tests that there are no repeating aliases for comparison, which would overwrite
    the earlier defined comparison."""
    COMPARISONS = []
    for c in all_comparisons:
        if c.aliases is not None:
            for a in c.aliases:
                COMPARISONS.append(a)
    duplicates = set([x for x in COMPARISONS if COMPARISONS.count(x) > 1])
    print(f"Duplicate comparisons: {duplicates}")
    assert len(duplicates) == 0
