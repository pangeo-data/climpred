import numpy as np
import pytest
import xarray as xr

from climpred import PerfectModelEnsemble
from climpred.exceptions import DatasetError
from climpred.metrics import DETERMINISTIC_PM_METRICS

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

category_edges = np.array([9.5, 10.0, 10.5])

ITERATIONS = 3


def test_perfectModelEnsemble_init(PM_ds_initialized_1d):
    """Test to see if perfect model ensemble can be initialized"""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    print(PerfectModelEnsemble)
    assert pm


def test_perfectModelEnsemble_init_da(PM_da_initialized_1d):
    """Test to see if perfect model ensemble can be initialized with da"""
    pm = PerfectModelEnsemble(PM_da_initialized_1d)
    assert pm


def test_add_control(perfectModelEnsemble_initialized_control):
    """Test to see if control can be added to PerfectModelEnsemble"""
    assert perfectModelEnsemble_initialized_control.get_control()


def test_generate_uninit(perfectModelEnsemble_initialized_control):
    """Test to see if uninitialized ensemble can be bootstrapped"""
    pm = perfectModelEnsemble_initialized_control
    pm = pm.generate_uninitialized()
    assert pm.get_uninitialized()


def test_compute_persistence(perfectModelEnsemble_initialized_control):
    """Test that compute persistence can be run for perfect model ensemble"""
    perfectModelEnsemble_initialized_control._compute_persistence(metric="acc")


def test_get_initialized(PM_ds_initialized_1d):
    """Test whether get_initialized function works."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    init = pm.get_initialized()
    assert init == pm._datasets["initialized"]


def test_get_uninitialized(perfectModelEnsemble_initialized_control):
    """Test whether get_uninitialized function works."""
    pm = perfectModelEnsemble_initialized_control
    pm = pm.generate_uninitialized()
    uninit = pm.get_uninitialized()
    assert uninit == pm._datasets["uninitialized"]


def test_get_control(perfectModelEnsemble_initialized_control):
    """Test whether get_control function works."""
    ctrl = perfectModelEnsemble_initialized_control.get_control()
    assert ctrl == perfectModelEnsemble_initialized_control._datasets["control"]


def test_inplace(PM_ds_initialized_1d, PM_ds_control_1d):
    """Tests that inplace operations do not work."""
    pm = PerfectModelEnsemble(PM_ds_initialized_1d)
    # Adding a control.
    pm.add_control(PM_ds_control_1d)
    with_ctrl = pm.add_control(PM_ds_control_1d)
    assert pm != with_ctrl
    # Adding an uninitialized ensemble.
    pm = pm.add_control(PM_ds_control_1d)
    pm.generate_uninitialized()
    with_uninit = pm.generate_uninitialized()
    assert pm != with_uninit
    # Applying arbitrary func.
    pm.sum("init")
    summed = pm.sum("init")
    assert pm != summed


def test_verify(perfectModelEnsemble_initialized_control):
    """Test that verify works."""
    assert perfectModelEnsemble_initialized_control.verify(
        metric="mse", comparison="m2e", dim=["init", "member"]
    )


def test_verify_metric_kwargs(perfectModelEnsemble_initialized_control):
    """Test that verify with metric_kwargs works."""
    pm = perfectModelEnsemble_initialized_control
    pm = pm - pm.mean("time").mean("init")
    assert pm.verify(
        metric="threshold_brier_score",
        comparison="m2c",
        dim=["init", "member"],
        threshold=0.5,
    )


@pytest.mark.parametrize("reference", references, ids=references_ids)
def test_verify_reference(perfectModelEnsemble_initialized_control, reference):
    """Test that verify works with references given."""
    pm = perfectModelEnsemble_initialized_control.generate_uninitialized()
    skill = (
        pm.verify(
            metric="rmse", comparison="m2e", dim=["init", "member"], reference=reference
        )
        .expand_dims(["lon", "lat"])
        .isel(lon=[0] * 2, lat=[0] * 2)
    )  # make geospatial
    if isinstance(reference, str):
        reference = [reference]
    elif reference is None:
        reference = []
    if len(reference) == 0:
        assert "skill" not in skill.dims
    else:
        assert skill.skill.size == len(reference) + 1
    # test skills not none
    assert skill.notnull().all()
    assert "dayofyear" not in skill.coords


def test_verify_fails_expected_metric_kwargs(perfectModelEnsemble_initialized_control):
    """Test that verify without metric_kwargs fails."""
    pm = perfectModelEnsemble_initialized_control
    pm = pm - pm.mean("time").mean("init")
    with pytest.raises(ValueError) as excinfo:
        pm.verify(
            metric="threshold_brier_score", comparison="m2c", dim=["init", "member"]
        )
    assert "Please provide threshold." == str(excinfo.value)


def test_compute_uninitialized_metric_kwargs(perfectModelEnsemble_initialized_control):
    "Test that _compute_uninitialized with metric_kwargs works"
    pm = perfectModelEnsemble_initialized_control
    pm = pm - pm.mean("time").mean("init")
    pm = pm.generate_uninitialized()
    assert pm._compute_uninitialized(
        metric="threshold_brier_score",
        comparison="m2c",
        threshold=0.5,
        dim=["init", "member"],
    )


def test_bootstrap_metric_kwargs(perfectModelEnsemble_initialized_control):
    """Test that bootstrap with metric_kwargs works."""
    pm = perfectModelEnsemble_initialized_control
    pm = pm - pm.mean("time").mean("init")
    pm = pm.generate_uninitialized()
    assert pm.bootstrap(
        metric="threshold_brier_score",
        comparison="m2c",
        threshold=0.5,
        iterations=ITERATIONS,
        dim=["init", "member"],
    )


def test_calendar_matching_control(PM_da_initialized_1d, PM_ds_control_1d):
    """Tests that error is thrown if calendars mismatch when adding observations."""
    pm = PerfectModelEnsemble(PM_da_initialized_1d)
    PM_ds_control_1d["time"] = xr.cftime_range(
        start="1950", periods=PM_ds_control_1d.time.size, freq="MS", calendar="all_leap"
    )
    with pytest.raises(ValueError) as excinfo:
        pm = pm.add_control(PM_ds_control_1d)
    assert "does not match" in str(excinfo.value)


def test_persistence_dim(perfectModelEnsemble_initialized_control):
    pm = perfectModelEnsemble_initialized_control.expand_dims(
        "lon"
    ).generate_uninitialized()
    assert "lon" in pm.get_initialized().dims
    dim = ["lon"]
    metric = "rmse"
    comparison = "m2e"

    actual = pm._compute_persistence(metric=metric, dim=dim)
    assert "lon" not in actual.dims
    assert "init" in actual.dims

    actual = pm.verify(
        metric=metric,
        comparison=comparison,
        dim=dim,
        reference=["persistence", "uninitialized"],
    )
    assert "lon" not in actual.dims
    assert "init" in actual.dims

    pm = perfectModelEnsemble_initialized_control.expand_dims("lon")
    # fix _resample_iterations_idx doesnt work with singular dimension somewhere.
    pm = pm.isel(lon=[0, 0])
    actual = pm.bootstrap(metric=metric, comparison=comparison, dim=dim, iterations=2)
    assert "lon" not in actual.dims
    assert "init" in actual.dims


def test_HindcastEnsemble_as_PerfectModelEnsemble(hindcast_recon_1d_mm):
    """Test that initialized dataset for HindcastEnsemble can also be used for
    PerfectModelEnsemble."""
    v = "SST"
    alignment = "maximize"
    hindcast = hindcast_recon_1d_mm.isel(lead=[0, 1])
    assert (
        not hindcast.verify(
            metric="acc", comparison="e2o", dim="init", alignment=alignment
        )[v]
        .isnull()
        .any()
    )

    # try PerfectModelEnsemble predictability
    init = hindcast.get_initialized()
    pm = PerfectModelEnsemble(init)

    assert (
        not pm.verify(metric="acc", comparison="m2e", dim=["member", "init"])[v]
        .isnull()
        .any()
    )


def test_verify_no_need_for_control(PM_da_initialized_1d, PM_da_control_1d):
    """Tests that no error is thrown when no control present
    when calling verify(reference=['uninitialized'])."""
    v = "tos"
    comparison = "m2e"
    pm = PerfectModelEnsemble(PM_da_initialized_1d).isel(lead=[0, 1, 2])
    # verify needs to control
    skill = pm.verify(metric="mse", comparison=comparison, dim="init")
    assert not skill[v].isnull().any()
    # control not needed for normalized metrics as normalized
    # with verif which is the verification member in PM and
    # not the control simulation.
    assert (
        not pm.verify(metric="nmse", comparison=comparison, dim="init")[v]
        .isnull()
        .any()
    )

    with pytest.raises(DatasetError) as e:
        pm.verify(
            metric="mse", comparison=comparison, dim="init", reference=["persistence"]
        )
    assert "at least one control dataset" in str(e.value)

    # unlikely case that control gets deleted after generating uninitialized
    pm = pm.add_control(PM_da_control_1d).generate_uninitialized()
    pm._datasets["control"] = {}
    assert (
        not pm._compute_uninitialized(metric="mse", comparison=comparison, dim="init")[
            v
        ]
        .isnull()
        .any()
    )

    assert (
        not pm.verify(
            metric="mse", comparison=comparison, dim="init", reference=["uninitialized"]
        )[v]
        .isnull()
        .any()
    )


def test_verify_reference_same_dims(perfectModelEnsemble_initialized_control):
    """Test that verify returns the same dimensionality regardless of reference."""
    pm = perfectModelEnsemble_initialized_control.generate_uninitialized()
    pm = pm.isel(lead=[0, 1, 2], init=[0, 1, 2])
    metric = "mse"
    comparison = "m2e"
    dim = "init"
    actual_no_ref = pm.verify(
        metric=metric, comparison=comparison, dim=dim, reference=None
    )
    actual_uninit_ref = pm.verify(
        metric=metric, comparison=comparison, dim=dim, reference="uninitialized"
    )
    actual_pers_ref = pm.verify(
        metric=metric, comparison=comparison, dim=dim, reference="persistence"
    )
    actual_clim_ref = pm.verify(
        metric=metric, comparison=comparison, dim=dim, reference="climatology"
    )
    assert actual_uninit_ref.skill.size == 2
    assert actual_pers_ref.skill.size == 2
    assert actual_clim_ref.skill.size == 2
    # no additional dimension, +1 because initialized squeezed
    assert len(actual_no_ref.dims) + 1 == len(actual_pers_ref.dims)
    assert len(actual_no_ref.dims) + 1 == len(actual_uninit_ref.dims)
    assert len(actual_no_ref.dims) + 1 == len(actual_clim_ref.dims)
    assert len(actual_pers_ref.dims) == len(actual_uninit_ref.dims)
    assert len(actual_clim_ref.dims) == len(actual_uninit_ref.dims)


@pytest.mark.parametrize("reference", references, ids=references_ids)
@pytest.mark.parametrize("comparison,dim", comparison_dim_PM)
@pytest.mark.parametrize("metric", DETERMINISTIC_PM_METRICS)
def test_PerfectModel_verify_bootstrap_deterministic(
    perfectModelEnsemble_initialized_control, comparison, metric, dim, reference
):
    """
    Checks that PerfectModel.verify() and PerfectModel.bootstrap() for
    deterministic metrics is not NaN.
    """
    pm = perfectModelEnsemble_initialized_control.isel(lead=[0, 1, 2], init=range(6))
    if isinstance(reference, str):
        reference = [reference]
    if metric == "contingency":
        metric_kwargs = {
            "forecast_category_edges": category_edges,
            "observation_category_edges": category_edges,
            "score": "accuracy",
        }
    elif metric == "roc":
        metric_kwargs = {"bin_edges": category_edges}
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

    actual = pm.verify(
        comparison=comparison,
        metric=metric,
        dim=dim,
        reference=reference,
        **metric_kwargs,
    ).tos
    if metric in ["contingency"] or metric in pearson_r_containing_metrics:
        # less strict here with all NaNs, pearson_r yields NaNs for climatology
        if "climatology" in reference:
            actual = actual.drop_sel(skill="climatology")
        assert not actual.isnull().all()
    else:
        assert not actual.isnull().any()

    # bootstrap()
    actual = pm.bootstrap(
        comparison=comparison,
        metric=metric,
        dim=dim,
        iterations=ITERATIONS,
        reference=reference,
        **metric_kwargs,
    ).tos
    if len(reference) > 0:
        actual = actual.drop_sel(results="p")

    if metric in ["contingency"] or metric in pearson_r_containing_metrics:
        # less strict here with all NaNs, pearson_r yields NaNs for climatology
        if "climatology" in reference:
            actual = actual.drop_sel(skill="climatology")
        assert not actual.sel(results="verify skill").isnull().all()
    else:
        assert not actual.sel(results="verify skill").isnull().any()


@pytest.mark.parametrize("metric", ("rmse", "pearson_r"))
def test_pvalue_from_bootstrapping(perfectModelEnsemble_initialized_control, metric):
    """Test that pvalue of initialized ensemble first lead is close to 0."""
    sig = 95
    pm = perfectModelEnsemble_initialized_control.isel(lead=[0, 1, 2])
    actual = (
        pm.bootstrap(
            metric=metric,
            iterations=ITERATIONS,
            comparison="e2c",
            sig=sig,
            dim="init",
            reference="uninitialized",
        )
        .sel(skill="uninitialized", results="p")
        .isel(lead=0)
    )
    # check that significant p-value
    assert actual.tos.values < 2 * (1 - sig / 100)
    # lead units keep
    assert actual.lead.attrs["units"] == "years"


def testPerfectModelEnsemble_verify_groupby(
    perfectModelEnsemble_initialized_control,
):
    """Test groupby keyword."""
    kw = dict(
        metric="mse",
        comparison="m2e",
        dim="init",
    )
    grouped_skill = perfectModelEnsemble_initialized_control.verify(
        **kw, groupby="month"
    )
    assert "month" in grouped_skill.dims
    grouped_skill = perfectModelEnsemble_initialized_control.verify(
        **kw,
        groupby=perfectModelEnsemble_initialized_control.get_initialized().init.dt.month,
    )
    assert "month" in grouped_skill.dims
    grouped_skill = perfectModelEnsemble_initialized_control.bootstrap(
        iterations=2,
        groupby="month",
        **kw,
    )
    assert "month" in grouped_skill.dims
