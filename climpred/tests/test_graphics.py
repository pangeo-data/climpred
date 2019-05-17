# import matplotlib.pyplot as plt

from climpred.bootstrap import bootstrap_hindcast, bootstrap_perfect_model
# from climpred.graphics import plot_bootstrapped_skill_over_leadyear
from climpred.loadutils import open_dataset


def test_mpi_hindcast_plot_bootstrapped_skill_over_leadyear():
    """
    Checks plots from bootstrap MPI hindcast works.
    """
    v = 'SST'
    base = 'MPIESM_miklip_baseline1-'
    hind = open_dataset(base + 'hind-' + v + '-global')[v]
    hist = open_dataset(base + 'hist-' + v + '-global')[v]
    assim = open_dataset(base + 'assim-' + v + '-global')[v]
    # sig = 95
    bootstrap = 5
    res = bootstrap_hindcast(hind,
                             hist,
                             assim,
                             metric='pearson_r',
                             bootstrap=bootstrap)
    # plot_bootstrapped_skill_over_leadyear(res, sig)
    assert res is not None


def test_mpi_pm_plot_bootstrapped_skill_over_leadyear():
    """
    Checks plots from bootstrap MPI hindcast works.
    """
    da = open_dataset('MPI-PM-DP-1D').isel(area=1, period=-1)
    PM_da_ds1d = da['tos']

    da = open_dataset('MPI-control-1D').isel(area=1, period=-1)
    PM_da_control1d = da['tos']

    # sig = 95
    bootstrap = 5
    res = bootstrap_perfect_model(PM_da_ds1d,
                                  PM_da_control1d,
                                  metric='pearson_r',
                                  bootstrap=bootstrap)

    # plot_bootstrapped_skill_over_leadyear(res, sig)
    assert res is not None
