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


@pytest.mark.parametrize('common', ['init', 'verif', 'max_dof', None])
def test_logg_compute_hindcast_common(
    hind_ds_initialized_1d, reconstruction_ds_1d, caplog, common
):
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
            if common in [None, 'max_dof']:
                # check that
                first_year_per_lead = (
                    hind_ds_initialized_1d.init.min().dt.year.values + lead
                )
                assert f'{first_year_per_lead}' in record[2]
            elif common == 'verif':
                # check that verification time is always the same
                # const_verif_length = xx
                pass  # skip test for now
                # get verification times
                # first = 1980  # made up
                # last = 2000  # made up
                # assert f'{first}' in record[2] and f'{last}' in record[2]
                # check always same time length verified
                # assert const_verif_length == last_year_per_lead - first_year_per_lead
            elif common == 'init':
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
