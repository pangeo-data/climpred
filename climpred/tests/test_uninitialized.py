"""Test compute_uninitialized."""
import pytest

from climpred.constants import VALID_ALIGNMENTS
from climpred.reference import compute_uninitialized


@pytest.mark.parametrize("alignment", VALID_ALIGNMENTS)
def test_compute_uninitialized_alignment(
    hind_ds_initialized_1d, reconstruction_ds_1d, hist_ds_uninitialized_1d, alignment
):
    """Tests that compute_uninitialized works for various alignments."""
    res = (
        compute_uninitialized(
            hind_ds_initialized_1d,
            hist_ds_uninitialized_1d,
            reconstruction_ds_1d,
            metric="pr",
            comparison="e2o",
            alignment=alignment,
        )
        .isnull()
        .any()
    )
    for var in res.data_vars:
        assert not res[var]


def test_compute_uninitialized_same_verifs(
    hind_ds_initialized_1d, reconstruction_ds_1d, hist_ds_uninitialized_1d
):
    """Tests that uninitialized skill is same at all leads for `same_verifs`
    alignment."""
    res = compute_uninitialized(
        hind_ds_initialized_1d,
        hist_ds_uninitialized_1d,
        reconstruction_ds_1d,
        metric="pr",
        comparison="e2o",
        alignment="same_verifs",
    ).to_array()
    assert ((res - res[0]) == 0).all()


def test_uninitialized_keeps_member_dim(hindcast_hist_obs_1d):
    """https://github.com/pangeo-data/climpred/issues/735"""
    skill = hindcast_hist_obs_1d.verify(
        dim="init",
        metric="mse",
        comparison="m2o",
        reference="uninitialized",
        alignment="maximize",
    ).SST
    assert "member" in skill.dims
    assert (skill.std("member") > 0).all()
