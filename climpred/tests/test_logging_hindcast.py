import logging

import xarray as xr

from climpred.prediction import compute_hindcast


def test_logg_compute_hindcast(
    hind_ds_initialized_1d, reconstruction_ds_1d, caplog,
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
            # for now just checking years
            first_real_time_year = (
                hind_ds_initialized_1d.init.min().dt.year.values + lead
            )
            assert f'{first_real_time_year}' in record[2]
