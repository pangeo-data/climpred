# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


import dask
import numpy as np
import xarray as xr

from climpred.bootstrap import bootstrap_hindcast
# from climpred.constants import HINDCAST_COMPARISONS
from climpred.prediction import compute_hindcast

from . import parameterized, randn, requires_dask

HINDCAST_COMPARISONS = ['m2r']  # e2r and probabilistic dont match
METRICS = ['rmse', 'pearson_r', 'crpss']

bootstrap = 8


def _ensure_loaded(res):
    """Compute no lazy results."""
    if dask.is_dask_collection(res):
        res = res.compute()
    return res


class Generate:
    """
    Generate random hind, control to be benckmarked.
    """

    timeout = 600
    repeat = (2, 5, 20)

    def make_hind_ref(self):
        """hind and ref mimik hindcast experiment"""
        self.hind = xr.Dataset()
        self.reference = xr.Dataset()
        self.hist = xr.Dataset()

        self.nmember = 3
        self.nlead = 3
        self.nx = 64
        self.ny = 64
        self.init_start = 1960
        self.init_end = 2000
        self.ninit = self.init_end - self.init_start

        frac_nan = 0.0

        inits = np.arange(self.init_start, self.init_end)
        leads = np.arange(1, 1 + self.nlead)
        members = np.arange(1, 1 + self.nmember)

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
        self.hind['tos'] = xr.DataArray(
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
        self.reference['tos'] = xr.DataArray(
            randn((self.ninit, self.nx, self.ny), frac_nan=frac_nan),
            coords={'lon': lons, 'lat': lats, 'time': inits},
            dims=('time', 'lon', 'lat'),
            name='tos',
            encoding=None,
            attrs={'units': 'foo units', 'description': 'a description'},
        )

        self.hist['tos'] = xr.DataArray(
            randn((self.ninit, self.nx, self.ny, self.nmember), frac_nan=frac_nan),
            coords={'lon': lons, 'lat': lats, 'time': inits, 'member': members},
            dims=('time', 'lon', 'lat', 'member'),
            name='tos',
            encoding=None,
            attrs={'units': 'foo units', 'description': 'a description'},
        )

        self.hind.attrs = {'history': 'created for xarray benchmarking'}


class Compute(Generate):
    """
    A few examples that benchmark climpred compute_hindcast.
    """

    def setup(self, *args, **kwargs):
        self.make_hind_ref()

    @parameterized(['metric', 'comparison'], (METRICS, HINDCAST_COMPARISONS))
    def time_compute_hindcast(self, metric, comparison):
        """Take time for compute_hindcast."""
        _ensure_loaded(
            compute_hindcast(
                self.hind, self.reference, metric=metric, comparison=comparison
            )
        )

    @parameterized(['metric', 'comparison'], (METRICS, HINDCAST_COMPARISONS))
    def peakmem_compute_hindcast(self, metric, comparison):
        """Take memory peak for compute_hindcast for all comparisons."""
        _ensure_loaded(
            compute_hindcast(
                self.hind, self.reference, metric=metric, comparison=comparison
            )
        )

    @parameterized(['metric', 'comparison'], (METRICS, HINDCAST_COMPARISONS))
    def time_bootstrap_hindcast(self, metric, comparison):
        """Take time for bootstrap_hindcast for one metric."""
        _ensure_loaded(
            bootstrap_hindcast(
                self.hind,
                self.reference,
                self.hist,
                metric=metric,
                comparison=comparison,
                bootstrap=bootstrap,
            )
        )

    @parameterized(['metric', 'comparison'], (METRICS, HINDCAST_COMPARISONS))
    def peakmem_bootstrap_hindcast(self, metric, comparison):
        """Take memory peak for bootstrap_hindcast."""
        _ensure_loaded(
            bootstrap_hindcast(
                self.hind,
                self.reference,
                self.hist,
                metric=metric,
                comparison=comparison,
                bootstrap=bootstrap,
            )
        )


class ComputeDask(Compute):
    def setup(self, *args, **kwargs):
        requires_dask()
        super().setup(**kwargs)
        # chunk along a spatial dimension to enable embarrasingly parallel computation
        self.hind = self.hind['tos'].chunk({'lon': self.nx // bootstrap})
        self.reference = self.reference['tos'].chunk({'lon': self.nx // bootstrap})
        self.hist = self.hist['tos'].chunk({'lon': self.nx // bootstrap})
