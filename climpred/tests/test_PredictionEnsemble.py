import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal

from climpred import HindcastEnsemble, PerfectModelEnsemble
from climpred.classes import PredictionEnsemble
from climpred.constants import CF_LONG_NAMES, CF_STANDARD_NAMES, CLIMPRED_DIMS
from climpred.exceptions import VariableError
from climpred.testing import assert_PredictionEnsemble, check_dataset_dims_and_data_vars

xr.set_options(display_style="text")

ALLOWED_TYPES_FOR_MATH_OPERATORS = [
    int,
    float,
    np.ndarray,
    xr.DataArray,
    xr.Dataset,
    PredictionEnsemble,
]

OTHER_ALLOWED_HINDCAST = [
    2,
    2.0,
    np.array(2),
    xr.DataArray(2),
    xr.DataArray(2).chunk(),
    xr.Dataset({"SST": 2}),
    xr.Dataset({"SST": 2}).chunk(),
]
OTHER_ALLOWED_PM = [
    2,
    2.0,
    np.array(2),
    xr.DataArray(2),
    xr.DataArray(2).chunk(),
    xr.Dataset({"tos": 2}),
    xr.Dataset({"tos": 2}).chunk(),
]
OTHER_ALLOWED_IDS = [
    "int",
    "float",
    "np.array",
    "xr.DataArray",
    "chunked xr.DataArray",
    "xr.Dataset",
    "chunked xr.Dataset",
]

OTHER_NOT_ALLOWED = ["2", list(), dict(), set()]
OTHER_NOT_ALLOWED_IDS = ["str", "list", "dict", "set"]

MATH_OPERATORS = ["add", "sub", "mul", "div"]


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    return a / b


def gen_error_str(he, operator, other):
    print(type(operator))
    OPERATOR_STR = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
    }
    return f"Cannot use {type(he)} {OPERATOR_STR[operator]} {type(other)}"


# HindcastEnsemble
@pytest.mark.parametrize("operator", MATH_OPERATORS, ids=MATH_OPERATORS)
@pytest.mark.parametrize("other", OTHER_ALLOWED_HINDCAST, ids=OTHER_ALLOWED_IDS)
def test_hindcastEnsemble_plus_defined(
    hind_ds_initialized_1d,
    hist_ds_uninitialized_1d,
    observations_ds_1d,
    other,
    operator,
):
    """Test that HindcastEnsemble math operator (+-*/) other works correctly for
    allowed other types."""
    he = HindcastEnsemble(hind_ds_initialized_1d)
    he = he.add_uninitialized(hist_ds_uninitialized_1d)
    he = he.add_observations(observations_ds_1d)
    operator = eval(operator)
    he2 = operator(he, other)
    for dataset in he._datasets:
        if he._datasets[dataset]:
            print("check", dataset)
            assert_equal(he2._datasets[dataset], operator(he._datasets[dataset], other))
            # check same dims and data_vars as before
            check_dataset_dims_and_data_vars(he, he2, dataset)


@pytest.mark.parametrize("operator", MATH_OPERATORS, ids=MATH_OPERATORS)
def test_hindcastEnsemble_plus_hindcastEnsemble(
    hind_ds_initialized_1d, hist_ds_uninitialized_1d, observations_ds_1d, operator
):
    """Test that HindcastEnsemble math operator (+-*/) HindcastEnsemble works
    correctly."""
    he = HindcastEnsemble(hind_ds_initialized_1d)
    he = he.add_uninitialized(hist_ds_uninitialized_1d)
    he = he.add_observations(observations_ds_1d)
    other = he.mean("init")
    operator = eval(operator)
    he2 = operator(he, other)
    for dataset in he._datasets:
        if he._datasets[dataset]:
            print("check", dataset)
            assert_equal(
                he2._datasets[dataset],
                operator(he._datasets[dataset], other._datasets[dataset]),
            )
            # check same dims and data_vars as before
            check_dataset_dims_and_data_vars(he, he2, dataset)


@pytest.mark.parametrize("operator", MATH_OPERATORS, ids=MATH_OPERATORS)
@pytest.mark.parametrize("other", OTHER_NOT_ALLOWED, ids=OTHER_NOT_ALLOWED_IDS)
def test_hindcastEnsemble_plus_not_defined(hind_ds_initialized_1d, other, operator):
    """Test that HindcastEnsemble math operator (+-*/) other raises error for
    non-defined others."""
    he = HindcastEnsemble(hind_ds_initialized_1d)
    error_str = gen_error_str(he, operator, other)
    operator = eval(operator)
    with pytest.raises(TypeError) as excinfo:
        operator(he, other)
    assert f"{error_str} because type {type(other)} not supported" in str(excinfo.value)


@pytest.mark.parametrize("operator", MATH_OPERATORS, ids=MATH_OPERATORS)
@pytest.mark.parametrize("other", OTHER_ALLOWED_PM[-2:], ids=OTHER_ALLOWED_IDS[-2:])
def test_hindcastEnsemble_plus_Dataset_different_name(
    hind_ds_initialized_1d, other, operator
):
    """Test that HindcastEnsemble math operator (+-*/) other raises error for
    non-defined others."""

    he = HindcastEnsemble(hind_ds_initialized_1d)
    error_str = gen_error_str(he, operator, other)
    operator = eval(operator)
    with pytest.raises(VariableError) as excinfo:
        operator(he, other)
    assert f"{error_str} with new `data_vars`" in str(excinfo.value)


@pytest.mark.parametrize("operator", MATH_OPERATORS, ids=MATH_OPERATORS)
def test_hindcastEnsemble_plus_broadcast(hind_ds_initialized_3d, operator):
    """Test that HindcastEnsemble math operator (+-*/) other also broadcasts
    correctly."""
    he = HindcastEnsemble(hind_ds_initialized_3d)
    operator = eval(operator)
    # minimal adding an offset or like multiplying area
    he2 = operator(
        he, xr.ones_like(hind_ds_initialized_3d.isel(init=1, lead=1, drop=True))
    )
    he3 = operator(he, 1)
    assert_PredictionEnsemble(he2, he3)


def test_hindcastEnsemble_operator_different_datasets(
    hind_ds_initialized_1d, observations_ds_1d
):
    """Test that HindcastEnsemble math operator (+-*/) on HindcastEnsemble."""
    he = HindcastEnsemble(hind_ds_initialized_1d)
    he = he.add_observations(observations_ds_1d)
    he2 = HindcastEnsemble(hind_ds_initialized_1d)
    assert not (he2 - he).equals(he2)
    assert not (he - he2).equals(he)


def test_HindcastEnsemble_area_weighted_mean(hind_ds_initialized_3d):
    """Test area weighted mean HindcastEnsemble."""
    he = HindcastEnsemble(hind_ds_initialized_3d)
    # fake area
    area = hind_ds_initialized_3d["TAREA"]
    spatial_dims = [d for d in hind_ds_initialized_3d.dims if d not in CLIMPRED_DIMS]
    # PredictionEnsemble doesnt like other data_vars
    he_self_spatial_mean = (he * area).sum(spatial_dims) / area.sum()
    # weighted requires Dataset
    area = area.to_dataset(name="area")
    he_xr_spatial_mean = he.weighted(area).mean(spatial_dims)
    assert_PredictionEnsemble(
        he_self_spatial_mean, he_xr_spatial_mean, how="allclose", rtol=0.03, atol=0.05
    )


# basically all copied
# PerfectModelEnsemble
@pytest.mark.parametrize("operator", MATH_OPERATORS, ids=MATH_OPERATORS)
@pytest.mark.parametrize("other", OTHER_ALLOWED_PM, ids=OTHER_ALLOWED_IDS)
def test_PerfectModelEnsemble_plus_defined(
    PM_ds_initialized_1d, PM_ds_control_1d, other, operator
):
    """Test that PerfectModelEnsemble math operator (+-*/) other works correctly for
    allowed other types."""
    he = PerfectModelEnsemble(PM_ds_initialized_1d)
    he = he.add_control(PM_ds_control_1d)
    operator = eval(operator)
    he2 = operator(he, other)
    for dataset in he._datasets:
        if he._datasets[dataset]:
            print("check", dataset)
            assert_equal(he2._datasets[dataset], operator(he._datasets[dataset], other))
            # check same dims and data_vars as before
            check_dataset_dims_and_data_vars(he, he2, dataset)


@pytest.mark.parametrize("operator", MATH_OPERATORS, ids=MATH_OPERATORS)
def test_PerfectModelEnsemble_plus_PerfectModelEnsemble(
    PM_ds_initialized_1d, PM_ds_control_1d, operator
):
    """Test that PerfectModelEnsemble math operator (+-*/) PerfectModelEnsemble works
    correctly."""
    he = PerfectModelEnsemble(PM_ds_initialized_1d)
    he = he.add_control(PM_ds_control_1d)
    other = he.mean("init")
    operator = eval(operator)
    he2 = operator(he, other)
    for dataset in he._datasets:
        if he._datasets[dataset]:
            print("check", dataset)
            assert_equal(
                he2._datasets[dataset],
                operator(he._datasets[dataset], other._datasets[dataset]),
            )
            # check same dims and data_vars as before
            check_dataset_dims_and_data_vars(he, he2, dataset)


@pytest.mark.parametrize("operator", MATH_OPERATORS, ids=MATH_OPERATORS)
@pytest.mark.parametrize("other", OTHER_NOT_ALLOWED, ids=OTHER_NOT_ALLOWED_IDS)
def test_PerfectModelEnsemble_plus_not_defined(PM_ds_initialized_1d, other, operator):
    """Test that PerfectModelEnsemble math operator (+-*/) other raises error for
    non-defined others."""
    he = PerfectModelEnsemble(PM_ds_initialized_1d)
    error_str = gen_error_str(he, operator, other)
    operator = eval(operator)
    with pytest.raises(TypeError) as excinfo:
        operator(he, other)
    assert f"{error_str} because type {type(other)} not supported" in str(excinfo.value)


@pytest.mark.parametrize("operator", MATH_OPERATORS, ids=MATH_OPERATORS)
@pytest.mark.parametrize(
    "other", OTHER_ALLOWED_HINDCAST[-2:], ids=OTHER_ALLOWED_IDS[-2:]
)
def test_PerfectModelEnsemble_plus_Dataset_different_name(
    PM_ds_initialized_1d, other, operator
):
    """Test that PerfectModelEnsemble math operator (+-*/) other raises error for
    Dataset with other dims and/or variables."""
    he = PerfectModelEnsemble(PM_ds_initialized_1d)
    error_str = gen_error_str(he, operator, other)
    operator = eval(operator)
    with pytest.raises(VariableError) as excinfo:
        operator(he, other)
    assert f"{error_str} with new `data_vars`" in str(excinfo.value)


@pytest.mark.parametrize("operator", MATH_OPERATORS, ids=MATH_OPERATORS)
def test_PerfectModelEnsemble_plus_broadcast(PM_ds_initialized_3d, operator):
    """Test that PerfectModelEnsemble math operator (+-*/) other also broadcasts
    correctly."""
    he = PerfectModelEnsemble(PM_ds_initialized_3d)
    operator = eval(operator)
    # minimal adding an offset or like multiplying area
    he2 = operator(
        he, xr.ones_like(PM_ds_initialized_3d.isel(init=1, lead=1, drop=True))
    )
    he3 = operator(he, 1)
    assert_PredictionEnsemble(he2, he3)


def test_PerfectModelEnsemble_area_weighted_mean(PM_ds_initialized_3d):
    """Test area weighted mean PerfectModelEnsemble."""
    he = PerfectModelEnsemble(PM_ds_initialized_3d)
    # fake area
    area = np.cos(PM_ds_initialized_3d.lat) + 1
    spatial_dims = [d for d in PM_ds_initialized_3d.dims if d not in CLIMPRED_DIMS]
    # PredictionEnsemble doesnt like other data_vars
    he_self_spatial_mean = (he * area).sum(spatial_dims) / area.sum()
    # weighted requires Dataset
    area = area.to_dataset(name="area")
    he_xr_spatial_mean = he.weighted(area).mean(spatial_dims)
    assert_PredictionEnsemble(
        he_self_spatial_mean, he_xr_spatial_mean, how="allclose", rtol=0.03, atol=0.05
    )


@pytest.mark.parametrize("varlist", [["tos", "sos"], ["AMO"], "AMO"])
def test_subset_getitem_datavariables(
    perfectModelEnsemble_3v_initialized_control_1d, varlist
):
    """Test variable subselection from __getitem__."""
    pm = perfectModelEnsemble_3v_initialized_control_1d
    all_datavars = list(pm.get_initialized().data_vars)
    pm_subset = pm[varlist]
    if isinstance(varlist, str):
        varlist = [varlist]
    # test that varlist is present
    for var in varlist:
        assert var in pm_subset.get_initialized().data_vars
    # test that others are not present anymore
    for var in all_datavars:
        if var not in varlist:
            assert var not in pm_subset.get_initialized().data_vars


@pytest.mark.parametrize("equal", [True, False])
def test_eq_ne(perfectModelEnsemble_3v_initialized_control_1d, equal):
    xr.set_options(display_style="text")
    pm = perfectModelEnsemble_3v_initialized_control_1d
    if equal:
        pm2 = pm
        assert isinstance(pm2, PredictionEnsemble)
        print(pm, pm2)
        print("expect: True, False")
        print(pm == pm2)
        print(pm != pm2)
        assert pm == pm2
        assert isinstance(pm == pm2, bool)
        assert not (pm != pm2)
        assert isinstance(pm != pm2, bool)
        # assert False
    else:
        pm2 = pm * 1.6
        assert isinstance(pm2, PredictionEnsemble)
        print(pm, pm2)
        print("expect: True, False")
        print(pm != pm2)
        print(pm == pm2)
        assert not (pm == pm2)
        assert isinstance(pm == pm2, bool)
        assert pm != pm2
    assert isinstance(pm != pm2, bool)


pe = [
    pytest.lazy_fixture("hindcast_hist_obs_1d"),
    pytest.lazy_fixture("perfectModelEnsemble_initialized_control"),
]
pe_ids = ["HindcastEnsemble", "PerfectModelEnsemble"]


@pytest.mark.parametrize("pe", pe, ids=pe_ids)
def test_PredictionEnsemble_data_vars(pe):
    assert isinstance(pe.data_vars, xr.core.dataset.DataVariables)
    assert list(pe.data_vars) == list(pe.get_initialized().data_vars)
    assert len(pe) == len(pe.data_vars)


@pytest.mark.parametrize("pe", pe, ids=pe_ids)
def test_PredictionEnsemble_delitem(pe):
    v = list(pe.data_vars)[0]
    del pe[v]
    assert list(pe.data_vars) == []


@pytest.mark.parametrize("pe", pe, ids=pe_ids)
def test_PredictionEnsemble_nbytes(pe):
    assert pe.nbytes > pe.get_initialized().nbytes


@pytest.mark.parametrize("pe", pe, ids=pe_ids)
def test_PredictionEnsemble_coords(pe):
    assert isinstance(pe.coords, xr.core.coordinates.DatasetCoordinates)
    assert "time" in pe.coords
    assert "time" not in pe.get_initialized().coords


@pytest.mark.parametrize("pe", pe, ids=pe_ids)
def test_PredictionEnsemble_sizes(pe):
    assert isinstance(pe.sizes, dict)
    assert len(pe.sizes) > len(pe.get_initialized().sizes)


@pytest.mark.parametrize("pe", pe, ids=pe_ids)
def test_PredictionEnsemble_dims(pe):
    assert isinstance(pe.dims, xr.core.utils.Frozen)
    assert len(pe.dims) > len(pe.get_initialized().dims)


@pytest.mark.parametrize("pe", pe, ids=pe_ids)
def test_PredictionEnsemble_contains(pe):
    v = list(pe.data_vars)[0]
    not_v = v + "notVar"
    assert isinstance(v in pe, bool)
    assert v in pe
    assert not_v not in pe


@pytest.mark.parametrize("pe", pe, ids=pe_ids)
def test_PredictionEnsemble_equals(pe):
    assert isinstance(pe.equals(pe), bool)
    assert pe.equals(pe)
    pe2 = pe.copy(deep=False)
    pe2._datasets["initialized"].init.attrs["comment"] = "should not fail"
    assert pe.equals(pe2)

    pe2 = pe2 + 1
    assert not pe.equals(pe2)


@pytest.mark.parametrize("pe", pe, ids=pe_ids)
def test_PredictionEnsemble_identical(pe):
    assert isinstance(pe.identical(pe), bool)
    assert pe.identical(pe)

    v = list(pe.data_vars)[0]
    pe2 = pe.copy(deep=False)
    pe2._datasets["initialized"][v].attrs["comment"] = "should fail"
    assert not pe.identical(pe2)


@pytest.mark.parametrize("pe", pe, ids=pe_ids)
def test_PredictionEnsemble_cf(pe):
    """Test that cf_xarray added metadata."""

    # standard_names
    for k, v in CF_STANDARD_NAMES.items():
        if k in pe.coords:
            assert v == pe.get_initialized().coords[k].attrs["standard_name"]

    # long_names
    for k, v in CF_LONG_NAMES.items():
        if k in pe.coords:
            assert v == pe.get_initialized().coords[k].attrs["long_name"]

    # description
    for k, v in CF_LONG_NAMES.items():
        if k in pe.coords:
            assert len(pe.get_initialized().coords[k].attrs["description"]) > 5


def test_warn_if_chunked_along_init_member_time(
    hindcast_hist_obs_1d, perfectModelEnsemble_initialized_control
):
    """Test that _warn_if_chunked_along_init_member_time warns."""
    he = hindcast_hist_obs_1d
    with pytest.warns(UserWarning, match="is chunked along dimensions"):
        he_chunked = HindcastEnsemble(
            he.get_initialized().chunk({"init": 10})
        ).add_observations(he.get_observations())
        with pytest.raises(
            ValueError, match="pass ``allow_rechunk=True`` in ``dask_gufunc_kwargs``"
        ):
            he_chunked.verify(
                metric="rmse", dim="init", comparison="e2o", alignment="same_inits"
            )

    with pytest.warns(UserWarning, match="is chunked along dimensions"):
        he_chunked = HindcastEnsemble(he.get_initialized()).add_observations(
            he.get_observations().chunk({"time": 10})
        )
        with pytest.raises(
            ValueError, match="pass ``allow_rechunk=True`` in ``dask_gufunc_kwargs``"
        ):
            he_chunked.verify(
                metric="rmse", dim="init", comparison="e2o", alignment="same_inits"
            )

    pm = perfectModelEnsemble_initialized_control
    with pytest.warns(UserWarning, match="is chunked along dimensions"):
        PerfectModelEnsemble(pm.get_initialized().chunk({"init": 10}))
