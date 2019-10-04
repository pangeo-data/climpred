import numpy as np
import xarray as xr
from climpred.prediction import compute_perfect_model
from climpred.toy import create_initialized, run_skill_for_ensemble
from climpred.tutorial import load_dataset
from xarray.testing import assert_allclose

# standard parameters; can be changed
lead = xr.DataArray(np.arange(0, 20, 1), dims='lead')
lead['lead'] = lead.values
control = load_dataset('MPI-control-1D')['tos'].isel(period=-1, area=1)
# amplitudes of the signal and noise
noise_amplitude = control.std().values * 2.5
signal_amplitude = control.std().values
# period of potentially predictable variable
P = 8


def test_initialized_has_skill():
    """
    Checks plots from bootstrap_perfect_model works.
    """
    ds = create_initialized(
        lead=lead,
        ninit=10,
        nmember=10,
        signal_amplitude=signal_amplitude,
        noise_amplitude=noise_amplitude,
    )
    skill = compute_perfect_model(ds, ds, metric='pearson_r')
    one = skill.isel(lead=0) / skill.isel(lead=0)
    assert_allclose(skill.isel(lead=0), one)
    assert (skill.isel(lead=[0, 1, 2]) > 0.5).all()


def test_larger_ensemble_less_skill_spread_than_smaller():
    ss_small = run_skill_for_ensemble(
        ninit=3,
        nmember=3,
        signal_amplitude=signal_amplitude,
        noise_amplitude=noise_amplitude,
        plot=False,
    )
    ss_large = run_skill_for_ensemble(
        ninit=10,
        nmember=10,
        signal_amplitude=signal_amplitude,
        noise_amplitude=noise_amplitude,
        plot=False,
    )
    assert (
        ss_small.isel(lead=slice(3, None)).std('bootstrap')
        > ss_large.isel(lead=slice(3, None)).std('bootstrap')
    ).all()
