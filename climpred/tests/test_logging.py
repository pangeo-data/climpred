import logging


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


def test_log_HindcastEnsemble_verify_datetime(hindcast_hist_obs_1d):
    hindcast = hindcast_hist_obs_1d.copy()
    hindcast.verify(metric="mse", comparison="e2o", dim="init", alignment="maximize")
    # convert to datetime
    hindcast = hindcast._convert_calendar("standard")
    hindcast.verify(metric="mse", comparison="e2o", dim="init", alignment="maximize")
