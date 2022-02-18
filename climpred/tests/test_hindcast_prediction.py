"""Test compute_persistence."""

from climpred.reference import compute_persistence


def test_persistence_lead0_lead1(
    hind_ds_initialized_1d, hind_ds_initialized_1d_lead0, reconstruction_ds_1d
):
    """
    Checks that compute persistence returns the same results with a lead-0 and lead-1
    framework.
    """
    res1 = compute_persistence(
        hind_ds_initialized_1d, reconstruction_ds_1d, metric="rmse"
    )
    res2 = compute_persistence(
        hind_ds_initialized_1d_lead0, reconstruction_ds_1d, metric="rmse"
    )
    assert (res1.SST.values == res2.SST.values).all()
