import logging

from climpred.prediction import compute_hindcast


def test_log_compute_hindcast(
    hind_ds_initialized_1d_cftime, reconstruction_ds_1d_cftime, caplog
):
    """Tests that logging works for compute_hindcast."""
    LOG_STRINGS = ['lead', 'dim', 'inits', 'verif']
    with caplog.at_level(logging.INFO):
        compute_hindcast(hind_ds_initialized_1d_cftime, reconstruction_ds_1d_cftime)
        for i, record in enumerate(caplog.record_tuples):
            print(record)
            assert all(x in record[2] for x in LOG_STRINGS)


def test_log_compute_hindcast_alignment_same_init(
    hind_ds_initialized_1d_cftime, reconstruction_ds_1d_cftime, caplog
):
    """Tests that inits are identical in the logger at all leads for same_inits
    alignment."""
    with caplog.at_level(logging.INFO):
        compute_hindcast(
            hind_ds_initialized_1d_cftime,
            reconstruction_ds_1d_cftime,
            alignment='same_inits',
        )
        for record in caplog.record_tuples:
            print(record)
            # Hard-coded for now, since we know what the inits should be for the demo
            # data.
            assert 'inits=1954-01-01 00:00:00-2007-01-01 00:00:00' in record[2]


def test_log_compute_hindcast_same_init_verifs(
    hind_ds_initialized_1d_cftime, reconstruction_ds_1d_cftime, caplog
):
    """Tests that appropriate verifs are being used at each lead for same_inits
    alignment."""
    with caplog.at_level(logging.INFO):
        # Hard-coded for now, since we know what the inits should be for the demo data.
        FIRST_INIT, LAST_INIT = 1954, 2007
        compute_hindcast(
            hind_ds_initialized_1d_cftime,
            reconstruction_ds_1d_cftime,
            alignment='same_inits',
        )
        for i, record in zip(
            hind_ds_initialized_1d_cftime['lead'].values, caplog.record_tuples
        ):
            assert (
                f'verif={FIRST_INIT+i}-01-01 00:00:00-{LAST_INIT+i}-01-01 00:00:00'
                in record[2]
            )
