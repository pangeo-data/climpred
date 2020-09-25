import logging

from climpred.prediction import compute_hindcast


def test_log_compute_hindcast(
    hind_ds_initialized_1d_cftime, reconstruction_ds_1d_cftime, caplog
):
    """Tests that logging works for compute_hindcast."""
    LOG_STRINGS = ["lead", "inits", "verifs"]
    with caplog.at_level(logging.INFO):
        compute_hindcast(hind_ds_initialized_1d_cftime, reconstruction_ds_1d_cftime)
        for i, record in enumerate(caplog.record_tuples):
            # Skip header information.
            if i >= 2:
                print(record)
                assert all(x in record[2] for x in LOG_STRINGS)
