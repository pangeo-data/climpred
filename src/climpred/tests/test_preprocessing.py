import os

import numpy as np
import pytest
import xarray as xr

from climpred.constants import CLIMPRED_ENSEMBLE_DIMS
from climpred.preprocessing.shared import (
    load_hindcast,
    rename_SLM_to_climpred_dims,
    rename_to_climpred_dims,
    set_integer_time_axis,
)

on_mistral = False
try:
    host = os.environ["HOSTNAME"]
    for node in ["mlogin", "mistralpp"]:
        if node in host:
            on_mistral = True
            from climpred.preprocessing.mpi import get_path
except KeyError:
    pass

# check for intake_esm to be installed
try:
    import intake
    import intake_esm

    print(intake_esm.__version__)
    intake_esm_loaded = True
except ImportError:
    intake_esm_loaded = False


def preprocess_1var(ds, v="global_primary_production"):
    return ds[v].to_dataset(name=v).squeeze()


@pytest.mark.mistral
@pytest.mark.skipif(not on_mistral, reason="requires to be on mistral.dkrz.de")
@pytest.mark.parametrize(
    "inits,members",
    [(range(1961, 1964), range(3, 6)), (range(1970, 1972), range(1, 3))],
)
def test_load_hindcast(inits, members):
    """Test that `load_hindcast` loads the appropriate files."""
    actual = load_hindcast(
        inits=inits,
        members=members,
        preprocess=preprocess_1var,
        get_path=get_path,
    )
    assert isinstance(actual, xr.Dataset)
    assert (actual["init"].values == inits).all()
    assert (actual["member"].values == members).all()
    assert "global_primary_production" in actual.data_vars
    assert len(actual.data_vars) == 1


@pytest.mark.mistral
@pytest.mark.skipif(not on_mistral, reason="requires to be on mistral.dkrz.de")
@pytest.mark.skipif(not intake_esm_loaded, reason="requires intake_esm to be installed")
def test_climpred_pre_with_intake_esm():
    """Test that `preprocess` including `set_integer_time_axis` enables concatination
    of all hindcast into one xr.object."""
    col_url = "/home/mpim/m300524/intake-esm-datastore/catalogs/mistral-cmip6.json"
    col = intake.open_esm_datastore(col_url)
    # load 2 members for 2 inits from one model
    query = dict(
        experiment_id=["dcppA-hindcast"],
        table_id="Amon",
        member_id=["r1i1p1f1", "r2i1p1f1"],
        dcpp_init_year=[1970, 1971],
        variable_id="tas",
        source_id="MPI-ESM1-2-HR",
    )
    cat = col.search(**query)
    cdf_kwargs = {"chunks": {"time": 12}, "decode_times": False}

    def preprocess(ds):
        # extract tiny spatial and temporal subset
        ds = ds.isel(lon=[50, 51, 52], lat=[50, 51, 52], time=np.arange(12 * 2))
        # make time dim identical
        ds = set_integer_time_axis(ds)
        return ds

    dset_dict = cat.to_dataset_dict(cdf_kwargs=cdf_kwargs, preprocess=preprocess)
    # get first dict value
    ds = dset_dict[list(dset_dict.keys())[0]]
    assert isinstance(ds, xr.Dataset)
    ds = rename_to_climpred_dims(ds)
    # check for all CLIMPRED_DIMS
    for c in CLIMPRED_ENSEMBLE_DIMS:
        assert c in ds.coords
    # check for requested dimsizes
    assert ds.member.size == 2
    assert ds.init.size == 2


def test_rename_SLM(da_SLM):
    """Check that dimensions in input are renamed by rename_SLM_to_climpred_dims."""
    da_renamed = rename_SLM_to_climpred_dims(da_SLM)
    for dim in CLIMPRED_ENSEMBLE_DIMS:
        assert dim in da_renamed.dims
    for dim in da_SLM.dims:
        assert dim not in da_renamed.dims


def test_rename_climpred_dims(da_dcpp):
    """Check that dimensions in input are renamed by rename_to_climpred_dims."""
    da_renamed = rename_to_climpred_dims(da_dcpp)
    for dim in CLIMPRED_ENSEMBLE_DIMS:
        assert dim in da_renamed.dims
    for dim in da_dcpp.dims:
        assert dim not in da_renamed.dims
