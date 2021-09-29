import numpy as np
import pytest
import xarray as xr

from climpred import HindcastEnsemble


@pytest.mark.parametrize(
    "init_freq,lead_unit",
    [
        ("AS-JUL", "years"),
        ("AS-JUL", "months"),
        ("AS-JUL", "seasons"),
        ("MS", "months"),
        ("3M", "days"),
        ("7D", "days"),
        ("1D", "hours"),
        ("1H", "seconds"),
    ],
)
@pytest.mark.parametrize("calendar", ["ProlepticGregorian", "standard", "360_day"])
def test_hindcastEnsemble_init_time(init_freq, lead_unit, calendar):
    """Test to see HindcastEnsemble can be initialized and creates valid_time
    coordinate depending on init and lead for different calendars and lead units."""
    p = 3
    nlead = 2
    lead = [0, 1]

    init = xr.cftime_range(start="2000", freq=init_freq, periods=p)
    data = np.random.rand(p, nlead)
    init = xr.DataArray(
        data,
        dims=["init", "lead"],
        coords={"init": init, "lead": lead},
        name="initialized",
    )
    init.lead.attrs["units"] = lead_unit
    coords = HindcastEnsemble(init).coords
    assert "valid_time" in coords
    assert (coords["valid_time"].isel(lead=0, drop=True) == coords["init"]).all()
    assert (coords["valid_time"].isel(lead=1, drop=True) != coords["init"]).all()


@pytest.mark.parametrize(
    "reference", [None, "climatology", "uninitialized", "persistence"]
)
@pytest.mark.parametrize("alignment", ["same_verif", "same_inits", "maximize"])
def test_verify_valid_time(hindcast_hist_obs_1d, alignment, reference):
    """Test that verify has 2d valid_time coordinate."""
    result = hindcast_hist_obs_1d.verify(
        metric="rmse",
        comparison="e2o",
        dim=[],
        alignment=alignment,
        reference=reference,
    )
    assert "time" not in result.dims
    assert "valid_time" in result.coords
    assert len(result.coords["valid_time"].dims) == 2
    if reference:
        assert set(result.dims) == set(["init", "lead", "skill"])
    else:
        assert set(result.dims) == set(["init", "lead"])


@pytest.mark.parametrize(
    "reference", [None, "climatology", "uninitialized", "persistence"]
)
@pytest.mark.parametrize("alignment", ["same_verif", "same_inits", "maximize"])
def test_bootstrap_valid_time(hindcast_hist_obs_1d, alignment, reference):
    """Test that bootstrap has 2d valid_time coordinate."""
    result = hindcast_hist_obs_1d.bootstrap(
        iterations=2,
        metric="rmse",
        comparison="e2o",
        dim=[],
        alignment=alignment,
        reference=reference,
    )
    assert "time" not in result.dims
    assert "valid_time" in result.coords
    assert len(result.coords["valid_time"].dims) == 2
    if reference:
        assert set(result.dims) == set(["init", "lead", "results", "skill"])
    else:
        assert set(result.dims) == set(["init", "lead", "results"])
