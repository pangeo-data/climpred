import logging

import numpy as np
import pytest
import xskillscore as xs

from climpred.prediction import compute_hindcast


def test_same_inits_initializations(
    hind_ds_initialized_1d_cftime, reconstruction_ds_1d_cftime, caplog
):
    """Tests that inits are identical at all leads for `same_inits` alignment."""
    with caplog.at_level(logging.INFO):
        compute_hindcast(
            hind_ds_initialized_1d_cftime,
            reconstruction_ds_1d_cftime,
            alignment='same_inits',
        )
        for i, record in enumerate(caplog.record_tuples):
            if i >= 2:
                print(record)
                # Hard-coded for now, since we know what the inits should be for the
                # demo data.
                assert 'inits=1954-01-01 00:00:00-2007-01-01 00:00:00' in record[2]


def test_same_inits_verification_dates(
    hind_ds_initialized_1d_cftime, reconstruction_ds_1d_cftime, caplog
):
    """Tests that appropriate verifs are being used at each lead for `same_inits`
    alignment."""
    with caplog.at_level(logging.INFO):
        # Hard-coded for now, since we know what the inits should be for the demo data.
        FIRST_INIT, LAST_INIT = 1954, 2007
        compute_hindcast(
            hind_ds_initialized_1d_cftime,
            reconstruction_ds_1d_cftime,
            alignment='same_inits',
        )
        nleads = hind_ds_initialized_1d_cftime['lead'].size
        for i, record in zip(np.arange(nleads + 2), caplog.record_tuples,):
            if i >= 2:
                print(record)
                assert (
                    f'verif={FIRST_INIT+i}-01-01 00:00:00-{LAST_INIT+i}-01-01 00:00:00'
                    in record[2]
                )


@pytest.mark.parametrize('alignment', ['same_inits', 'same_verifs'])
def test_same_inits_disjoint_verif_time(
    small_initialized_da, small_verif_da, alignment
):
    """Tests that alignment works with disjoint time in the verification
    data, i.e., non-continuous time sampling to verify against."""
    hind = small_initialized_da
    verif = small_verif_da.drop_sel(time=1992)
    actual = compute_hindcast(hind, verif, alignment=alignment, metric='mse')
    assert actual.notnull().all()
    # hindcast inits: [1990, 1991, 1992, 1993]
    # verif times: [1990, 1991, 1993, 1994]
    a = hind.sel(init=[1990, 1993]).rename({'init': 'time'})
    b = verif.sel(time=[1991, 1994])
    a['time'] = b['time']
    expected = xs.mse(a, b, 'time')
    assert actual == expected


@pytest.mark.parametrize('alignment', ['same_inits', 'same_verifs'])
def test_same_inits_disjoint_inits(small_initialized_da, small_verif_da, alignment):
    """Tests that alignment works with disjoint inits in the verification
    data, i.e., non-continuous initializing to verify with."""
    hind = small_initialized_da.drop_sel(init=1991)
    verif = small_verif_da
    actual = compute_hindcast(hind, verif, alignment=alignment, metric='mse')
    assert actual.notnull().all()
    # hindcast inits: [1990, 1992, 1993]
    # verif times: [1990, 1991, 1992, 1993, 1994]
    a = hind.rename({'init': 'time'})
    b = verif.sel(time=[1991, 1993, 1994])
    a['time'] = b['time']
    expected = xs.mse(a, b, 'time')
    assert actual == expected


def test_same_verifs_verification_dates(
    hind_ds_initialized_1d_cftime, reconstruction_ds_1d_cftime, caplog
):
    """Tests that verifs are identical at all leads for `same_verifs` alignment."""
    with caplog.at_level(logging.INFO):
        compute_hindcast(
            hind_ds_initialized_1d_cftime,
            reconstruction_ds_1d_cftime,
            alignment='same_verifs',
        )
        for i, record in enumerate(caplog.record_tuples):
            if i >= 2:
                print(record)
                # Hard-coded for now, since we know what the verifs should be for the
                # demo data.
                assert 'verif=1964-01-01 00:00:00-2017-01-01 00:00:00' in record[2]


def test_same_verifs_initializations(
    hind_ds_initialized_1d_cftime, reconstruction_ds_1d_cftime, caplog
):
    """Tests that appropriate verifs are being used at each lead for `same_inits`
    alignment."""
    with caplog.at_level(logging.INFO):
        # Hard-coded for now, since we know what the inits should be for the demo data.
        FIRST_INIT, LAST_INIT = 1964, 2017
        compute_hindcast(
            hind_ds_initialized_1d_cftime,
            reconstruction_ds_1d_cftime,
            alignment='same_verifs',
        )
        nleads = hind_ds_initialized_1d_cftime['lead'].size
        for i, record in zip(np.arange(nleads + 2), caplog.record_tuples,):
            if i >= 2:
                print(record)
                assert (
                    f'inits={FIRST_INIT-i}-01-01 00:00:00-{LAST_INIT-i}-01-01 00:00:00'
                    in record[2]
                )
