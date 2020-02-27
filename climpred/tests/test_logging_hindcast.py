import logging

import pytest
import xarray as xr

from climpred.prediction import compute_hindcast


def test_logg_compute_hindcast(hind_ds_initialized_1d, reconstruction_ds_1d, caplog):
    """
    Checks that logging in compute hindcast works.
    """
    # setting up data for cftime init/time
    hind_ds_initialized_1d['init'] = xr.cftime_range(
        start=str(hind_ds_initialized_1d.init.min().values),
        freq='YS',
        periods=hind_ds_initialized_1d.init.size,
    )
    hind_ds_initialized_1d.lead.attrs['units'] = 'years'
    reconstruction_ds_1d['time'] = xr.cftime_range(
        start=str(reconstruction_ds_1d.time.min().values),
        freq='YS',
        periods=reconstruction_ds_1d.time.size,
    )

    with caplog.at_level(logging.INFO):
        compute_hindcast(hind_ds_initialized_1d, reconstruction_ds_1d)
        # check for each record
        for i, record in enumerate(caplog.record_tuples):
            print(record)
            lead = hind_ds_initialized_1d.isel(lead=i).lead.values.astype('int')
            assert f'at lead={str(lead).zfill(2)}' in record[2]
            # for now just checking years
            first_year_per_lead = (
                hind_ds_initialized_1d.init.min().dt.year.values + lead
            )
            assert f'{first_year_per_lead}' in record[2]


@pytest.mark.parametrize('alignment', ['init', 'verif', 'maximize', None])
def test_logg_compute_hindcast_alignment(
    hind_ds_initialized_1d, reconstruction_ds_1d, caplog, alignment
):
    """
    Checks that logging in compute hindcast captures the correct verification-time
    matching based on `alignment`.
    """
    # setting up data for cftime init/time
    hind_ds_initialized_1d['init'] = xr.cftime_range(
        start=str(hind_ds_initialized_1d.init.min().values),
        freq='YS',
        periods=hind_ds_initialized_1d.init.size,
    )
    hind_ds_initialized_1d.lead.attrs['units'] = 'years'
    reconstruction_ds_1d['time'] = xr.cftime_range(
        start=str(reconstruction_ds_1d.time.min().values),
        freq='YS',
        periods=reconstruction_ds_1d.time.size,
    )

    with caplog.at_level(logging.INFO):
        compute_hindcast(hind_ds_initialized_1d, reconstruction_ds_1d)
        # check for each record
        for i, record in enumerate(caplog.record_tuples):
            print(record)
            lead = hind_ds_initialized_1d.isel(lead=i).lead.values.astype('int')
            assert f'at lead={str(lead).zfill(2)}' in record[2]
            if alignment == 'maximize':
                # check that
                first_year_per_lead = (
                    hind_ds_initialized_1d.init.min().dt.year.values + lead
                )
                assert f'{first_year_per_lead}' in record[2]
            elif alignment == 'verif':
                # check that verification time is always the same
                # const_verif_length = xx
                pass  # skip test for now
                # get verification times
                # first = 1980  # made up
                # last = 2000  # made up
                # assert f'{first}' in record[2] and f'{last}' in record[2]
                # check always same time length verified
                # assert const_verif_length == last_year_per_lead - first_year_per_lead
            elif alignment == 'init':
                # check that verification length is always the same and init stay the
                # same, therefore verification time should monotonically decrease for
                # monotonic lead increases
                # const_verif_length = xx
                first_year_per_lead = (
                    hind_ds_initialized_1d.init.min().dt.year.values + lead
                )
                assert f'{first_year_per_lead}' in record[2]
                last_year_per_lead = (
                    hind_ds_initialized_1d.init.max().dt.year.values
                    + lead
                    - hind_ds_initialized_1d.lead.size
                )
                assert f'{last_year_per_lead}' in record[2]
                # check always same time length verified
                # assert const_verif_length == last_year_per_lead - first_year_per_lead


@pytest.mark.parametrize('alignment', ['init', 'verif', 'maximize'])
def test_logg_compute_hindcast_alignment_checking_actual_years(
    hind_ds_initialized_1d, reconstruction_ds_1d, caplog, alignment
):
    """
    Checks that logging in compute hindcast works where we check on actual hard coded
    numbers based on pre-defined datasets from `load_dataset`:

    data:
    - hind: hind_ds_initialized_1d: inits: 1955-2017, leads: 1-10
    - verif: reconstruction_ds_1d: time: 1954-2015

    expected verification time logged for keyword `alignment`: and pattern:
    - inits: 1956-2006, 1957-2007, ... , 1965-2015 : first_common_init_verif+lead -
    - verif: 1965-2006 : max(first_init+max_lead, verif_time)-min(last_init-max_lead)
    - maximize: 1956-2015, 1957-2015, ... , 1965-2015: maximizing
    """
    # setting up data for cftime init/time
    hind_ds_initialized_1d['init'] = xr.cftime_range(
        start=str(hind_ds_initialized_1d.init.min().values),
        freq='YS',
        periods=hind_ds_initialized_1d.init.size,
    )
    hind_ds_initialized_1d.lead.attrs['units'] = 'years'
    reconstruction_ds_1d['time'] = xr.cftime_range(
        start=str(reconstruction_ds_1d.time.min().values),
        freq='YS',
        periods=reconstruction_ds_1d.time.size,
    )

    with caplog.at_level(logging.INFO):
        compute_hindcast(hind_ds_initialized_1d, reconstruction_ds_1d)
        # check for each record
        for i, record in enumerate(caplog.record_tuples):
            print(record)
            lead = hind_ds_initialized_1d.isel(lead=i).lead.values.astype('int')
            assert f'at lead={str(lead).zfill(2)}' in record[2]
            if alignment == 'maximize':
                # check that
                first_year_per_lead = (
                    hind_ds_initialized_1d.init.min().dt.year.values + lead
                )
                assert f'{first_year_per_lead}' in record[2]
            elif alignment == 'verif':
                # check that verification time is always the same
                # const_verif_length = xx
                pass  # skip test for now
                # get verification times
                # first = 1980  # made up
                # last = 2000  # made up
                # assert f'{first}' in record[2] and f'{last}' in record[2]
                # check always same time length verified
                # assert const_verif_length == last_year_per_lead - first_year_per_lead
            elif alignment == 'init':
                # check that verification length is always the same and init stay the
                # same, therefore verification time should monotonically decrease for
                # monotonic lead increases
                # const_verif_length = xx
                first_year_per_lead = (
                    hind_ds_initialized_1d.init.min().dt.year.values + lead
                )
                assert f'{first_year_per_lead}' in record[2]
                last_year_per_lead = (
                    hind_ds_initialized_1d.init.max().dt.year.values
                    + lead
                    - hind_ds_initialized_1d.lead.size
                )
                assert f'{last_year_per_lead}' in record[2]
                # check always same time length verified
                # assert const_verif_length == last_year_per_lead - first_year_per_lead
