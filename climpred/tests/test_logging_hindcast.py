import logging

from climpred.prediction import compute_hindcast


def test_logg_compute_hindcast(
    hind_ds_initialized_1d, reconstruction_ds_1d, caplog,
):
    """
    Checks that logging in compute hindcast works.
    """
    with caplog.at_level(logging.INFO):
        compute_hindcast(hind_ds_initialized_1d, reconstruction_ds_1d)
        # check for each record
        for i, record in enumerate(caplog.record_tuples):
            print(record)
            lead = hind_ds_initialized_1d.isel(lead=i).lead.values.astype('int')
            assert f'at lead={lead}' in record[2]
            # should change to cftime
            # for now just checking years
            first_real_time = hind_ds_initialized_1d.init.min().values + lead
            assert f'{first_real_time}' in record[2]
