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


def test_log_HindcastEnsemble_verify(hindcast_hist_obs_1d, caplog):
    """Test that verify logs."""
    LOG_STRINGS = ["lead", "inits", "verifs"]
    with caplog.at_level(logging.INFO):
        hindcast_hist_obs_1d.verify(
            metric="mse", comparison="e2o", dim="init", alignment="same_verif"
        )
        for i, record in enumerate(caplog.record_tuples):
            # Skip header information.
            if i >= 2:
                print(record)
                assert all(x in record[2] for x in LOG_STRINGS)
                assert "initialized" in record[2]
