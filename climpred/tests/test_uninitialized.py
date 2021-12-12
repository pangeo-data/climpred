"""Test compute_uninitialized."""
import pytest

from climpred.constants import VALID_ALIGNMENTS
from climpred.reference import compute_uninitialized


@pytest.mark.parametrize("alignment", VALID_ALIGNMENTS)
def test_compute_uninitialized_alignment(
    hind_ds_initialized_1d, reconstruction_ds_1d, hist_ds_uninitialized_1d, alignment
):
    """Tests that compute_uninitialized works for various alignments."""
    res = (
        compute_uninitialized(
            hind_ds_initialized_1d,
            hist_ds_uninitialized_1d,
            reconstruction_ds_1d,
            metric="pr",
            comparison="e2o",
            alignment=alignment,
        )
        .isnull()
        .any()
    )
    for var in res.data_vars:
        assert not res[var]


def test_compute_uninitialized_same_verifs(
    hind_da_initialized_1d, reconstruction_da_1d, hist_da_uninitialized_1d
):
    """Tests that uninitialized skill is same at all leads for `same_verifs`
    alignment."""
    res = compute_uninitialized(
        hind_da_initialized_1d,
        hist_da_uninitialized_1d,
        reconstruction_da_1d,
        metric="pr",
        comparison="e2o",
        alignment="same_verifs",
    )
    assert ((res - res[0]) == 0).all()
