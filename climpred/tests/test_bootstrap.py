"""Test bootstrap.py."""

import dask
import numpy as np
import pytest
import xarray as xr
from xskillscore.core.resampling import (
    resample_iterations as _resample_iterations,
    resample_iterations_idx as _resample_iterations_idx,
)

from climpred.bootstrap import (
    _bootstrap_by_stacking,
    _chunk_before_resample_iterations_idx,
    _resample,
    bootstrap_hindcast,
    bootstrap_uninit_pm_ensemble_from_control_cftime,
)
from climpred.constants import CONCAT_KWARGS
from climpred.exceptions import KeywordError
from climpred.utils import _transpose_and_rechunk_to

# TODO: move to conftest.py
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

xr.set_options(display_style="text")


def test_bootstrap_PM_keep_lead_attrs(perfectModelEnsemble_initialized_control):
    """Test bootstrap_perfect_model works lazily."""
    pm = perfectModelEnsemble_initialized_control
    pm.get_initialized().lead.attrs["units"] = "years"
    s = pm.bootstrap(
        iterations=ITERATIONS,
        comparison="m2c",
        metric="mse",
    )
    assert "units" in s.lead.attrs
    assert s.lead.attrs["units"] == pm.get_initialized().lead.attrs["units"]


@pytest.mark.parametrize("comparison,dim", comparison_dim_PM)
@pytest.mark.parametrize("chunk", [True, False])
def test_bootstrap_PM_lazy_results(
    perfectModelEnsemble_initialized_control, chunk, comparison, dim
):
    """Test bootstrap_perfect_model works lazily."""
    pm = perfectModelEnsemble_initialized_control.isel(lead=range(3))
    if chunk:
        pm = pm.chunk({"lead": 2}).chunk({"time": -1})
    else:
        pm = pm.compute()
    s = pm.bootstrap(
        iterations=ITERATIONS,
        comparison=comparison,
        metric="mse",
        dim=dim,
    )
    assert dask.is_dask_collection(s) == chunk


@pytest.mark.slow
@pytest.mark.parametrize("chunk", [True, False])
def test_bootstrap_hindcast_lazy(
    hindcast_hist_obs_1d,
    chunk,
):
    """Test bootstrap_hindcast works lazily."""
    he = hindcast_hist_obs_1d.isel(lead=range(3), init=range(10))
    if chunk:
        he = he.chunk({"lead": 2})
    else:
        he = he.compute()

    s = he.bootstrap(
        iterations=ITERATIONS,
        comparison="e2o",
        metric="mse",
        alignment="same_verifs",
        dim="init",
    )
    assert dask.is_dask_collection(s) == chunk


@pytest.mark.slow
@pytest.mark.parametrize("resample_dim", ["member", "init"])
def test_bootstrap_hindcast_resample_dim(
    hindcast_hist_obs_1d,
    resample_dim,
):
    """Test bootstrap_hindcast when resampling member or init and alignment
    same_inits."""
    hindcast_hist_obs_1d.isel(lead=range(3), init=range(10)).bootstrap(
        iterations=ITERATIONS,
        comparison="e2o",
        metric="mse",
        resample_dim=resample_dim,
        alignment="same_inits",
        dim="init",
    )


def bootstrap_uninit_pm_ensemble_from_control(init_pm, control):
    """
    Create a pseudo-ensemble from control run. Deprecated in favor of
    `bootstrap_uninit_pm_ensemble_from_control_cftime`.

    Note:
        Needed for block bootstrapping confidence intervals of a metric in perfect
        model framework. Takes randomly segments of length of ensemble dataset from
        control and rearranges them into ensemble and member dimensions.

    Args:
        init_pm (xarray object): ensemble simulation.
        control (xarray object): control simulation.

    Returns:
        uninit (xarray object): pseudo-ensemble generated from control run.
    """
    nens = init_pm.init.size
    nmember = init_pm.member.size
    length = init_pm.lead.size
    c_start = 0
    c_end = control["time"].size
    lead_time = init_pm["lead"]

    def set_coords(uninit, init, dim):
        uninit[dim] = init[dim].values
        return uninit

    def isel_years(control, year_s, length):
        new = control.isel(time=slice(year_s, year_s + length))
        new = new.rename({"time": "lead"})
        new["lead"] = lead_time
        return new

    def create_pseudo_members(control):
        startlist = np.random.randint(c_start, c_end - length - 1, nmember)
        uninit_ens = xr.concat(
            (isel_years(control, start, length) for start in startlist),
            dim="member",
            **CONCAT_KWARGS,
        )
        return uninit_ens

    uninit = xr.concat(
        (
            set_coords(create_pseudo_members(control), init_pm, "member")
            for _ in range(nens)
        ),
        dim="init",
        **CONCAT_KWARGS,
    )
    # chunk to same dims
    return (
        _transpose_and_rechunk_to(uninit, init_pm)
        if dask.is_dask_collection(uninit)
        else uninit
    )


def test_bootstrap_uninit_pm_ensemble_from_control_cftime_annual_identical(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
):
    """Test ``bootstrap_uninit_pm_ensemble_from_control_cftime`` cftime identical to
    ``bootstrap_uninit_pm_ensemble_from_control`` for annual data."""
    cftime_res = bootstrap_uninit_pm_ensemble_from_control_cftime(
        PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
    )

    noncftime_res = bootstrap_uninit_pm_ensemble_from_control(
        PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
    )
    # lead and member identical
    for d in ["lead", "member"]:
        assert (cftime_res[d] == noncftime_res[d]).all()
    # init same size
    assert cftime_res["init"].size == noncftime_res["init"].size
    assert cftime_res.dims == noncftime_res.dims
    assert list(cftime_res.data_vars) == list(noncftime_res.data_vars)


def test_bootstrap_uninit_pm_ensemble_from_control_cftime_annual_identical_da(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
):
    """Test ``bootstrap_uninit_pm_ensemble_from_control_cftime`` cftime identical to
    ``bootstrap_uninit_pm_ensemble_from_control`` for annual data."""
    PM_ds_initialized_1d_ym_cftime = PM_ds_initialized_1d_ym_cftime["tos"]
    PM_ds_control_1d_ym_cftime = PM_ds_control_1d_ym_cftime["tos"]
    cftime_res = bootstrap_uninit_pm_ensemble_from_control_cftime(
        PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
    )
    noncftime_res = bootstrap_uninit_pm_ensemble_from_control(
        PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
    )
    # lead and member identical
    for d in ["lead", "member"]:
        assert (cftime_res[d] == noncftime_res[d]).all()
    # init same size
    assert cftime_res["init"].size == noncftime_res["init"].size
    assert cftime_res.name == noncftime_res.name
    # assert cftime_res.shape == noncftime_res.shape
    # assert cftime_res.dims == noncftime_res.dims


@pytest.mark.parametrize(
    "init, control",
    [
        (
            pytest.lazy_fixture("PM_ds_initialized_1d_ym_cftime"),
            pytest.lazy_fixture("PM_ds_control_1d_ym_cftime"),
        ),
        (
            pytest.lazy_fixture("PM_ds_initialized_1d_mm_cftime"),
            pytest.lazy_fixture("PM_ds_control_1d_mm_cftime"),
        ),
        (
            pytest.lazy_fixture("PM_ds_initialized_1d_dm_cftime"),
            pytest.lazy_fixture("PM_ds_control_1d_dm_cftime"),
        ),
    ],
)
def test_bootstrap_uninit_pm_ensemble_from_control_cftime_all_freq(init, control):
    """Test bootstrap_uninit_pm_ensemble_from_control_cftime for all freq data."""
    init = init.isel(lead=range(3), init=range(5))
    uninit = bootstrap_uninit_pm_ensemble_from_control_cftime(init, control)
    # lead and member identical
    for d in ["lead", "member"]:
        assert (uninit[d] == init[d]).all()
    # init same size
    assert uninit["init"].size == init["init"].size


def test_bootstrap_by_stacking_dataset(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
):
    res = _bootstrap_by_stacking(
        PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
    )
    assert res.lead.attrs["units"] == "years"
    assert isinstance(res, xr.Dataset)
    assert res.tos.dims == PM_ds_initialized_1d_ym_cftime.tos.dims


def test_bootstrap_by_stacking_dataarray(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
):
    v = list(PM_ds_initialized_1d_ym_cftime.data_vars)[0]
    res = _bootstrap_by_stacking(
        PM_ds_initialized_1d_ym_cftime[v], PM_ds_control_1d_ym_cftime[v]
    )
    assert res.lead.attrs["units"] == "years"
    assert isinstance(res, xr.DataArray)
    assert res.dims == PM_ds_initialized_1d_ym_cftime[v].dims


def test_bootstrap_by_stacking_chunked(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
):
    res_chunked = _bootstrap_by_stacking(
        PM_ds_initialized_1d_ym_cftime.chunk(),
        PM_ds_control_1d_ym_cftime.chunk(),
    )
    assert dask.is_dask_collection(res_chunked)
    res_chunked = res_chunked.compute()
    res = _bootstrap_by_stacking(
        PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
    )
    for d in ["lead", "member"]:
        assert (res_chunked[d] == res[d]).all()
    # init same size
    assert res_chunked["init"].size == res["init"].size


def test_bootstrap_by_stacking_two_var_dataset(
    PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
):
    """Test _bootstrap_by_stacking when init_pm and control two variable dataset."""
    PM_ds_initialized_1d_ym_cftime["sos"] = PM_ds_initialized_1d_ym_cftime["tos"]
    PM_ds_control_1d_ym_cftime["sos"] = PM_ds_control_1d_ym_cftime["tos"]
    res = _bootstrap_by_stacking(
        PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
    )
    res_cf = bootstrap_uninit_pm_ensemble_from_control_cftime(
        PM_ds_initialized_1d_ym_cftime, PM_ds_control_1d_ym_cftime
    )
    assert len(list(res.data_vars)) == len(list(res_cf.data_vars))
    # lead and member identical
    for d in ["lead", "member"]:
        assert (res[d] == res_cf[d]).all()
    # init same size
    assert res["init"].size == res_cf["init"].size


def test_bootstrap_hindcast_raises_error(
    hind_da_initialized_1d, hist_da_uninitialized_1d, observations_da_1d
):
    """Test that error is raised when user tries to resample over init and align over
    same_verifs."""
    with pytest.raises(KeywordError):
        bootstrap_hindcast(
            hind_da_initialized_1d,
            hist_da_uninitialized_1d,
            observations_da_1d,
            iterations=ITERATIONS,
            comparison="e2o",
            metric="mse",
            resample_dim="init",
            alignment="same_verifs",
        )


def test_resample_1_size(PM_da_initialized_1d):
    """Tests that the resampled dimensions are appropriate for a single iteration."""
    dim = "member"
    expected = _resample(PM_da_initialized_1d, resample_dim=dim)
    # 1 somehow fails
    actual = _resample_iterations_idx(PM_da_initialized_1d, 2, dim=dim).isel(
        iteration=0
    )
    assert expected.size == actual.size
    assert expected[dim].size == actual[dim].size


def test_resample_size(PM_da_initialized_1d):
    """Tests that the resampled dimensions are appropriate for many iterations."""
    dim = "member"
    expected = xr.concat(
        [_resample(PM_da_initialized_1d, resample_dim=dim) for i in range(ITERATIONS)],
        "iteration",
    )
    actual = _resample_iterations_idx(PM_da_initialized_1d, ITERATIONS, dim=dim)
    assert expected.size == actual.size
    assert expected[dim].size == actual[dim].size


@pytest.mark.parametrize("chunk", [True, False])
@pytest.mark.parametrize("replace", [True, False])
def test_resample_iterations_same(PM_da_initialized_1d, chunk, replace):
    """Test that both `resample_iterations` functions yield same result shape."""
    ds = PM_da_initialized_1d.isel(lead=range(3), init=range(5))
    if chunk:
        ds = ds.chunk()
    ds_r_idx = _resample_iterations_idx(ds, ITERATIONS, "member", replace=replace)
    ds_r = _resample_iterations(ds, ITERATIONS, "member", replace=replace)
    for d in ds.dims:
        xr.testing.assert_identical(ds_r[d], ds_r_idx[d])
        assert ds_r.size == ds_r_idx.size


def test_chunk_before_resample_iterations_idx(PM_da_initialized_3d_full):
    """Test that chunksize after `_resample_iteration_idx` is lower than
    `optimal_blocksize`."""
    chunking_dims = ["x", "y"]
    iterations = 50
    optimal_blocksize = 100000000
    ds_chunked = _chunk_before_resample_iterations_idx(
        PM_da_initialized_3d_full.chunk(),
        iterations,
        chunking_dims,
        optimal_blocksize=optimal_blocksize,
    )
    ds_chunked_chunksize = ds_chunked.data.nbytes / ds_chunked.data.npartitions
    print(
        dask.utils.format_bytes(ds_chunked_chunksize * iterations),
        "<",
        dask.utils.format_bytes(1.5 * optimal_blocksize),
    )
    assert ds_chunked_chunksize * iterations < 1.5 * optimal_blocksize


@pytest.mark.parametrize("chunk", [True, False])
@pytest.mark.parametrize("replace", [True, False])
def test_resample_iterations_dim_max(PM_da_initialized_1d, chunk, replace):
    """Test that both `resample_iterations(dim_max=n)` gives n members."""
    ds = PM_da_initialized_1d.isel(lead=range(3), init=range(5))
    ds = ds.sel(member=list(ds.member.values) * 2)
    ds["member"] = np.arange(1, 1 + ds.member.size)
    if chunk:
        ds = ds.chunk()
    ds_r = _resample_iterations(
        ds,
        ITERATIONS,
        "member",
        replace=replace,
        dim_max=PM_da_initialized_1d.member.size,
    )
    assert (ds_r["member"] == PM_da_initialized_1d.member).all()


@pytest.mark.skip(reason="this is a bug, test fails and should be resolved.")
def test_resample_iterations_dix_no_squeeze(PM_da_initialized_1d):
    """Test _resample_iteration_idx with singular dimension.

    Currently this fails for dimensions with just a single index as we use `squeeze` in
    the code and not using squeeze doesnt maintain functionality. This means that
    _resample_iteration_idx should not be called on singleton dimension inputs (which
    is not critical and can be circumvented when using squeeze before climpred.).
    """
    da = PM_da_initialized_1d.expand_dims("test_dim")
    print(da)
    actual = _resample_iterations_idx(da, iterations=ITERATIONS)
    assert "test_dim" in actual.dims


@pytest.mark.parametrize("metric", ["acc", "mae"])
def test_bootstrap_p_climatology(hindcast_hist_obs_1d, metric):
    """Test that p from bootstrap is close to 0 if skillful."""
    reference = "climatology"
    bskill = hindcast_hist_obs_1d.bootstrap(
        metric=metric,
        comparison="e2o",
        dim="init",
        iterations=21,
        alignment="same_inits",
        reference=reference,
    )
    v = "SST"
    lead = 1
    # first lead skill full
    if metric in ["acc"]:
        assert (
            (
                bskill.sel(skill="initialized", results="verify skill")
                > bskill.sel(skill=reference, results="verify skill")
            )
            .sel(lead=lead)[v]
            .all()
        )
    else:
        assert (
            (
                bskill.sel(skill="initialized", results="verify skill")
                < bskill.sel(skill=reference, results="verify skill")
            )
            .sel(lead=lead)[v]
            .all()
        )
    assert bskill.sel(skill=reference, results="p").sel(lead=lead)[v] < 0.1


def test_generate_uninitialized(hindcast_hist_obs_1d):
    """Test HindcastEnsemble.generate_uninitialized()"""
    from climpred.stats import rm_poly

    hindcast_hist_obs_1d_new = hindcast_hist_obs_1d.map(
        rm_poly, dim="init_or_time", deg=2
    ).generate_uninitialized()
    # new created
    assert not hindcast_hist_obs_1d_new.get_initialized().equals(
        hindcast_hist_obs_1d.get_initialized()
    )
    # skill different
    kw = dict(
        metric="mse",
        comparison="e2o",
        dim="init",
        alignment="same_verifs",
        reference="uninitialized",
    )
    assert not hindcast_hist_obs_1d_new.verify(**kw).equals(
        hindcast_hist_obs_1d.verify(**kw)
    )
