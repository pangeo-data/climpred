import pytest
import xarray as xr
from xarray.testing import assert_allclose

from climpred.comparisons import (
    PM_COMPARISONS,
    PROBABILISTIC_HINDCAST_COMPARISONS,
    PROBABILISTIC_PM_COMPARISONS,
)
from climpred.exceptions import DimensionError
from climpred.metrics import PM_METRICS
from climpred.utils import get_comparison_class, get_metric_class

xr.set_options(display_style="text")

ITERATIONS = 2

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


@pytest.mark.parametrize("metric", ["crps", "mse"])
@pytest.mark.parametrize("comparison", PROBABILISTIC_PM_COMPARISONS)
def test_pm_comparison_stack_dims_when_deterministic(
    PM_da_initialized_1d, comparison, metric
):
    metric = get_metric_class(metric, PM_METRICS)
    comparison = get_comparison_class(comparison, PM_COMPARISONS)
    actual_f, actual_r = comparison.function(PM_da_initialized_1d, metric=metric)
    if not metric.probabilistic:
        assert "member" in actual_f.dims
        assert "member" in actual_r.dims
    else:
        assert "member" in actual_f.dims
        assert "member" not in actual_r.dims


# cannot work for e2c, m2e comparison because only 1:1 comparison
@pytest.mark.parametrize("comparison", PROBABILISTIC_PM_COMPARISONS)
def test_compute_perfect_model_dim_over_member(
    perfectModelEnsemble_initialized_control, comparison
):
    """Test deterministic metric calc skill over member dim."""
    actual = perfectModelEnsemble_initialized_control.verify(
        comparison=comparison,
        metric="rmse",
        dim="member",
    )["tos"]
    assert "init" in actual.dims
    assert not actual.isnull().any()
    # check that init is cftime object
    assert "cftime" in str(type(actual.init.values[0]))


# cannot work for e2o comparison because only 1:1 comparison
@pytest.mark.parametrize("comparison", PROBABILISTIC_HINDCAST_COMPARISONS)
def test_compute_hindcast_dim_over_member(hindcast_hist_obs_1d, comparison):
    """Test deterministic metric calc skill over member dim."""
    print(hindcast_hist_obs_1d.get_initialized().coords)
    actual = hindcast_hist_obs_1d.verify(
        comparison=comparison, metric="rmse", dim="member", alignment="same_verif"
    )["SST"]
    assert "init" in actual.dims
    # mean init because skill has still coords for init lead
    assert not actual.mean("init").isnull().any()


def test_compute_perfect_model_different_dims_quite_close(
    perfectModelEnsemble_initialized_control,
):
    """Tests nearly equal dim=['init','member'] and dim='member'."""
    stack_dims_true = perfectModelEnsemble_initialized_control.verify(
        comparison="m2c",
        metric="rmse",
        dim=["init", "member"],
    )["tos"]
    stack_dims_false = perfectModelEnsemble_initialized_control.verify(
        comparison="m2c",
        metric="rmse",
        dim="member",
    ).mean(["init"])["tos"]
    # no more than 10% difference
    assert_allclose(stack_dims_true, stack_dims_false, rtol=0.1, atol=0.03)


@pytest.mark.parametrize("metric", ["rmse", "pearson_r"])
@pytest.mark.parametrize(
    "comparison,dim",
    comparison_dim_PM,
)
def test_compute_pm_dims(
    perfectModelEnsemble_initialized_control, dim, comparison, metric
):
    """Test whether compute_pm calcs skill over many possible dims
    and comparisons and just reduces the result by dim."""
    print(dim)
    pm = perfectModelEnsemble_initialized_control
    actual = pm.verify(metric=metric, comparison=comparison, dim=dim)["tos"]
    if isinstance(dim, str):
        dim = [dim]
    # check whether only dim got reduced from coords
    if comparison == "e2c":  # dont expect member, remove manually
        assert set(pm.get_initialized().dims) - set(["member"]) - set(dim) == set(
            actual.dims
        ), print(pm.get_initialized().dims, "-", dim, "!=", actual.dims)
    else:
        assert set(pm.get_initialized().dims) - set(dim) == set(actual.dims), print(
            pm.get_initialized().dims, "-", dim, "!=", actual.dims
        )
    # check whether all nan
    if metric not in ["pearson_r"]:
        assert not actual.isnull().any()


@pytest.mark.parametrize(
    "metric,dim", [("rmse", "init"), ("rmse", "member"), ("crps", "member")]
)
def test_compute_hindcast_dims(hindcast_hist_obs_1d, dim, metric):
    """Test whether compute_hindcast calcs skill over all possible dims
    and comparisons and just reduces the result by dim."""
    actual = hindcast_hist_obs_1d.verify(
        metric=metric, dim=dim, comparison="m2o", alignment="same_verif"
    )["SST"]
    # check whether only dim got reduced from coords
    assert set(hindcast_hist_obs_1d.get_initialized().dims) - set(actual.dims) == set(
        [dim]
    )
    # check whether all nan
    if "init" in actual.dims:
        actual = actual.mean("init")
    assert not actual.isnull().any()


@pytest.mark.parametrize(
    "dim",
    ["init", "member", None, ["init", "member"], ["x", "y"], ["x", "y", "member"]],
)
def test_PM_multiple_dims(
    perfectModelEnsemble_initialized_control_3d_North_Atlantic, dim
):
    """Test that PerfectModelEnsemble accepts dims as subset from initialized dims."""
    pm = perfectModelEnsemble_initialized_control_3d_North_Atlantic
    assert pm.verify(metric="rmse", comparison="m2e", dim=dim).any()


def test_PM_multiple_dims_fail_if_not_in_initialized(
    perfectModelEnsemble_initialized_control_3d_North_Atlantic,
):
    """Test that PerfectModelEnsemble.verify() for multiple dims fails when not subset
    from initialized dims."""
    pm = perfectModelEnsemble_initialized_control_3d_North_Atlantic.isel(x=4, y=4)
    with pytest.raises(DimensionError) as excinfo:
        pm.verify(metric="rmse", comparison="m2e", dim=["init", "member", "x"])
    assert "is expected to be a subset of `initialized.dims`" in str(excinfo.value)


def test_PM_fails_probabilistic_member_not_in_dim(
    perfectModelEnsemble_initialized_control_3d_North_Atlantic,
):
    """Test that PerfectModelEnsemble.verify() raises ValueError for `member` not in
    dim if probabilistic metric."""
    pm = perfectModelEnsemble_initialized_control_3d_North_Atlantic
    with pytest.raises(ValueError) as excinfo:
        pm.verify(metric="crps", comparison="m2c", dim=["init"])
    assert (
        "requires to be computed over dimension `member`, which is not found in"
        in str(excinfo.value)
    )


@pytest.mark.parametrize("dim", [["member"], ["member", "init"]])
@pytest.mark.parametrize("metric", ["crpss", "crpss_es"])
def test_hindcast_crpss(hindcast_recon_1d_ym, metric, dim):
    """Test that CRPSS metrics reduce by dimension dim in HindcastEnsemble.verify()."""
    he = hindcast_recon_1d_ym.isel(lead=[0, 1])
    actual = he.verify(
        dim=dim, metric=metric, alignment="same_verif", comparison="m2o"
    ).squeeze()
    before_dims = he.get_initialized().dims
    debug_message = f"{before_dims} - {dim} != {actual.dims} but should be"
    for d in before_dims:
        if d not in dim:
            assert d in actual.dims, debug_message
        else:
            assert d not in actual.dims, debug_message


@pytest.mark.parametrize("comparison", ["m2m", "m2c"])
@pytest.mark.parametrize("dim", [["member"], ["member", "init"]])
@pytest.mark.parametrize("metric", ["crpss", "crpss_es"])
def test_pm_crpss(
    perfectModelEnsemble_initialized_control_1d_ym_cftime, metric, dim, comparison
):
    """Test that CRPSS metrics reduce by dimension dim in
    PerfectModelEnsemble.verify()."""
    pm = perfectModelEnsemble_initialized_control_1d_ym_cftime
    actual = pm.verify(dim=dim, metric=metric, comparison=comparison).squeeze()
    before_dims = pm.get_initialized().dims
    debug_message = f"{before_dims} - {dim} != {actual.dims} but should be"
    for d in before_dims:
        if d in dim:
            assert d not in actual.dims, debug_message
        else:
            assert d in actual.dims, debug_message


def test_pm_metric_weights(perfectModelEnsemble_initialized_control_3d_North_Atlantic):
    """Test PerfectModelEnsemble.verify() with weights yields different results."""
    pm = perfectModelEnsemble_initialized_control_3d_North_Atlantic
    skipna = True
    metric = "rmse"
    comparison = "m2e"
    dim = ["x", "y"]
    weights = pm.get_initialized()["lat"]
    s_no_weights = pm.verify(
        metric=metric, comparison=comparison, dim=dim, skipna=skipna
    )
    s_weights = pm.verify(
        metric=metric, comparison=comparison, dim=dim, skipna=skipna, weights=weights
    )
    # want to test for non equalness
    assert ((s_no_weights["tos"] - s_weights["tos"]) != 0).all()


def test_hindcast_metric_weights(hindcast_recon_3d):
    """Test HindcastEnsemble.verify() with weights yields different results."""
    he = hindcast_recon_3d
    skipna = True
    metric = "rmse"
    comparison = "e2o"
    dim = ["nlat", "nlon"]
    alignment = "same_verifs"
    weights = he.get_initialized()["TAREA"]
    s_no_weights = he.verify(
        metric=metric,
        comparison=comparison,
        dim=dim,
        skipna=skipna,
        alignment=alignment,
    )
    s_weights = he.verify(
        metric=metric,
        comparison=comparison,
        dim=dim,
        skipna=skipna,
        weights=weights,
        alignment=alignment,
    )
    # want to test for non equalness
    assert ((s_no_weights["SST"] - s_weights["SST"]) != 0).all()
