# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


import xarray as xr
import numpy as np
from . import randn, parameterized

from climpred.prediction import compute_perfect_model
from climpred.bootstrap import bootstrap_perfect_model
from climpred.constants import PM_COMPARISONS, PM_METRICS

# faster
PM_METRICS = ['rmse', 'pearson_r', 'crpss']
# PM_COMPARISONS = ['m2e', 'e2c']


class Generate:
    """
    Generate random ds, control to be benckmarked.
    """

    timeout = 600
    repeat = (2, 5, 20)

    def make_ds(self):

        # ds
        self.ds = xr.Dataset()
        self.nmember = 3
        self.ninit = 4
        self.nlead = 3
        self.nx = 90  # 4 deg
        self.ny = 45  # 4 deg
        self.control_start = 3000
        self.control_end = 3300
        self.ntime = 300

        frac_nan = 0.0

        # control
        self.control = xr.Dataset()

        times = np.arange(self.control_start, self.control_end)
        leads = np.arange(1, 1 + self.nlead)
        members = np.arange(1, 1 + self.nmember)
        inits = list(
            np.random.randint(self.control_start, self.control_end + 1, self.ninit)
        )

        lons = xr.DataArray(
            np.linspace(0, 360, self.nx),
            dims=('lon',),
            attrs={'units': 'degrees east', 'long_name': 'longitude'},
        )
        lats = xr.DataArray(
            np.linspace(-90, 90, self.ny),
            dims=('lat',),
            attrs={'units': 'degrees north', 'long_name': 'latitude'},
        )
        self.ds['tos'] = xr.DataArray(
            randn(
                (self.nmember, self.ninit, self.nlead, self.nx, self.ny),
                frac_nan=frac_nan,
            ),
            coords={
                'member': members,
                'init': inits,
                'lon': lons,
                'lat': lats,
                'lead': leads,
            },
            dims=('member', 'init', 'lead', 'lon', 'lat'),
            name='tos',
            encoding=None,
            attrs={'units': 'foo units', 'description': 'a description'},
        )
        self.control['tos'] = xr.DataArray(
            randn((self.ntime, self.nx, self.ny), frac_nan=frac_nan),
            coords={'lon': lons, 'lat': lats, 'time': times},
            dims=('time', 'lon', 'lat'),
            name='tos',
            encoding=None,
            attrs={'units': 'foo units', 'description': 'a description'},
        )

        self.ds.attrs = {'history': 'created for xarray benchmarking'}


class Compute(Generate):
    """
    A few examples that benchmark climpred compute_perfect_model.
    """

    def setup(self, *args, **kwargs):
        self.make_ds()

    @parameterized(['metric', 'comparison'], (PM_METRICS, PM_COMPARISONS))
    def time_compute_perfect_model(self, metric, comparison):
        """Take time for compute_perfect_model."""
        compute_perfect_model(
            self.ds, self.control, metric=metric, comparison=comparison
        )

    @parameterized(['metric', 'comparison'], (['pearson_r', 'crpss'], PM_COMPARISONS))
    def peakmem_compute_perfect_model(self, metric, comparison):
        """Take memory peak for compute_perfect_model for all comparisons."""
        compute_perfect_model(
            self.ds, self.control, metric=metric, comparison=comparison
        )

    def time_bootstrap_perfect_model(self):
        """Take time for bootstrap_perfect_model for one metric."""
        bootstrap_perfect_model(
            self.ds, self.control, metric='mae', comparison='e2c', bootstrap=5
        )

    @parameterized(['metric', 'comparison'], (['pearson_r', 'crpss'], PM_COMPARISONS))
    def peakmem_bootstrap_perfect_model(self, metric, comparison):
        """Take memory peak for bootstrap_perfect_model."""
        bootstrap_perfect_model(
            self.ds, self.control, metric=metric, comparison=comparison, bootstrap=5
        )
