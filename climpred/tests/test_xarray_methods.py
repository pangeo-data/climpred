def test_drop_vars(hind_ds_initialized_1d):
    """Tests that ds.drop_vars() from version 0.14.1 of xarray works."""
    assert hind_ds_initialized_1d.drop_vars("lead")
