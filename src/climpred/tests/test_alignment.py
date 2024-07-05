import logging

import numpy as np
import pytest
import xskillscore as xs

from climpred import HindcastEnsemble
from climpred.alignment import _isin
from climpred.exceptions import CoordinateError


def test_same_inits_initializations(hindcast_hist_obs_1d, caplog):
    """Tests that inits are identical at all leads for `same_inits` alignment."""
    with caplog.at_level(logging.INFO):
        hindcast_hist_obs_1d.verify(
            metric="rmse",
            comparison="e2o",
            dim="init",
            alignment="same_inits",
        )
        for i, record in enumerate(caplog.record_tuples):
            if i >= 2:
                print(record)
                assert "inits: 1954-01-01 00:00:00-2005-01-01 00:00:00" in record[2]


def test_same_inits_verification_dates(hindcast_hist_obs_1d, caplog):
    """Tests that appropriate verifs are being used at each lead for `same_inits`
    alignment."""
    with caplog.at_level(logging.INFO):
        FIRST_INIT, LAST_INIT = 1955, 2006
        hindcast_hist_obs_1d.verify(
            metric="rmse",
            comparison="e2o",
            dim="init",
            alignment="same_inits",
        )
        nleads = hindcast_hist_obs_1d.coords["lead"].size
        for i, record in zip(
            np.arange(nleads + 2),
            caplog.record_tuples,
        ):
            if i >= 2:
                print(record)
                assert (
                    f"verifs: {FIRST_INIT + i}-01-01 00:00:00-{LAST_INIT + i}-01-01"
                    in record[2]
                )


@pytest.mark.parametrize("alignment", ["same_inits", "same_verifs"])
def test_disjoint_verif_time(small_initialized_da, small_verif_da, alignment):
    """Tests that alignment works with disjoint time in the verification
    data, i.e., non-continuous time sampling to verify against."""
    hind = small_initialized_da
    verif = small_verif_da.drop_sel(time=1992)
    actual = (
        HindcastEnsemble(hind)
        .add_observations(verif)
        .verify(comparison="e2o", dim="init", alignment=alignment, metric="mse")
    )
    assert actual.notnull().all()
    # hindcast inits: [1990, 1991, 1992, 1993]
    # verif times: [1990, 1991, 1993, 1994]
    a = hind.sel(init=[1990, 1992, 1993]).rename({"init": "time"})
    b = verif.sel(time=[1991, 1993, 1994])
    a["time"] = b["time"]
    expected = xs.mse(a, b, "time")
    assert actual == expected


@pytest.mark.parametrize("alignment", ["same_inits", "same_verifs"])
def test_disjoint_inits(small_initialized_da, small_verif_da, alignment):
    """Tests that alignment works with disjoint inits in the verification
    data, i.e., non-continuous initializing to verify with."""
    hind = small_initialized_da.drop_sel(init=1991)
    verif = small_verif_da
    actual = (
        HindcastEnsemble(hind)
        .add_observations(verif)
        .verify(comparison="e2o", dim="init", alignment=alignment, metric="mse")
    )
    assert actual.notnull().all()
    # hindcast inits: [1990, 1992, 1993]
    # verif times: [1990, 1991, 1992, 1993, 1994]
    a = hind.rename({"init": "time"})
    b = verif.sel(time=[1991, 1993, 1994])
    a["time"] = b["time"]
    expected = xs.mse(a, b, "time")
    assert actual == expected


def test_same_verifs_verification_dates(hindcast_hist_obs_1d, caplog):
    """Tests that verifs are identical at all leads for `same_verifs` alignment."""
    with caplog.at_level(logging.INFO):
        hindcast_hist_obs_1d.verify(
            metric="rmse",
            comparison="e2o",
            dim="init",
            alignment="same_verifs",
        )
        for i, record in enumerate(caplog.record_tuples):
            if i >= 2:
                print(record)
                assert "verifs: 1964-01-01 00:00:00-2015-01-01 00:00:00" in record[2]


def test_same_verifs_initializations(hindcast_hist_obs_1d, caplog):
    """Tests that appropriate verifs are being used at each lead for `same_inits`
    alignment."""
    with caplog.at_level(logging.INFO):
        FIRST_INIT, LAST_INIT = 1963, 2014
        hindcast_hist_obs_1d.verify(
            metric="rmse",
            comparison="e2o",
            dim="init",
            alignment="same_verifs",
        )
        nleads = hindcast_hist_obs_1d.coords["lead"].size
        for i, record in zip(
            np.arange(nleads + 2),
            caplog.record_tuples,
        ):
            if i >= 2:
                print(record)
                assert (
                    f"inits: {FIRST_INIT - i}-01-01 00:00:00-{LAST_INIT - i}-01-01 00:00:00"
                    in record[2]
                )


def test_same_verifs_raises_error_when_not_possible(hindcast_hist_obs_1d):
    """Tests that appropriate error is raised when a common set of verification dates
    cannot be found with the supplied initializations."""
    hind = hindcast_hist_obs_1d.isel(lead=slice(0, 3), init=[1, 3, 5, 7, 9])
    with pytest.raises(CoordinateError):
        hind.verify(
            metric="rmse", comparison="e2o", dim="init", alignment="same_verifs"
        )


def test_maximize_alignment_inits(hindcast_hist_obs_1d, caplog):
    """Tests that appropriate inits are selected for `maximize` alignment."""
    with caplog.at_level(logging.INFO):
        hindcast_hist_obs_1d.verify(
            metric="rmse",
            comparison="e2o",
            dim="init",
            alignment="maximize",
        )
        # Add dummy values for the first two lines since they are just metadata.
        for i, record in zip(
            np.concatenate(([0, 0], hindcast_hist_obs_1d.coords["lead"].values)),
            caplog.record_tuples,
        ):
            if i >= 1:
                print(record)
                assert (
                    f"inits: 1954-01-01 00:00:00-{2013 - i}-01-01 00:00:00" in record[2]
                )


def test_maximize_alignment_verifs(hindcast_hist_obs_1d, caplog):
    """Tests that appropriate verifs are selected for `maximize` alignment."""
    with caplog.at_level(logging.INFO):
        hindcast_hist_obs_1d.verify(
            metric="rmse",
            comparison="e2o",
            dim="init",
            alignment="maximize",
        )
        # Add dummy values for the first two lines since they are just metadata.
        for i, record in zip(
            np.concatenate(([0, 0], hindcast_hist_obs_1d.coords["lead"].values)),
            caplog.record_tuples,
        ):
            if i >= 1:
                print(record)
                assert (
                    f"verifs: {1956 + i}-01-01 00:00:00-2015-01-01 00:00:00"
                    in record[2]
                )


def test_my_isin(hindcast_recon_1d_ym):
    """Test _isin function calc on asi8 instead of xr.CFTimeIndex is equvi to isin."""
    he = hindcast_recon_1d_ym
    all_verifs = he.get_observations()["time"]
    init_lead_matrix = (
        he.coords["valid_time"]
        .drop_vars("valid_time")
        .rename(None)
        .rename({"init": "time"})
    )
    previous = init_lead_matrix.isin(all_verifs)
    faster = _isin(init_lead_matrix, all_verifs)
    assert previous.equals(faster)


def test_same_verifs_valid_time_no_nan(hindcast_hist_obs_1d):
    """Test that no NaNs are in valid_time coordinate for same_verifs."""
    skill = hindcast_hist_obs_1d.verify(
        metric="rmse",
        comparison="e2o",
        dim=[],  # important
        alignment="same_verifs",
    )
    assert not skill.coords["valid_time"].isnull().any()
