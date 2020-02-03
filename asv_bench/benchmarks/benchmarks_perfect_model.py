import numpy as np
import xarray as xr

from climpred.bootstrap import bootstrap_perfect_model
from climpred.prediction import compute_perfect_model

from . import ensure_loaded, parameterized, randn, requires_dask

# only take subselection of all possible metrics
METRICS = ['rmse', 'pearson_r', 'crpss']
# only take comparisons compatible with probabilistic metrics
PM_COMPARISONS = ['m2m', 'm2c']

BOOTSTRAP = 8


class Generate:
    """
    Generate input data for benchmark."""

    timeout = 600
    repeat = (2, 5, 20)

    def make_initialized_control(self):
        """Generates initialized ensembles and a control simulation, mimicking a
        perfect-model experiment."""
        self.ds = xr.Dataset()
        self.control = xr.Dataset()
        self.nmember = 3
        self.ninit = 4
        self.nlead = 3
        self.nx = 64
        self.ny = 64
        self.control_start = 3000
        self.control_end = 3300
        self.ntime = 300

        FRAC_NAN = 0.0

        times = np.arange(self.control_start, self.control_end)
        leads = np.arange(1, 1 + self.nlead)
        members = np.arange(1, 1 + self.nmember)
        inits = (
            np.random.choice(self.control_end - self.control_start, self.ninit)
            + self.control_start
        )

        lons = xr.DataArray(
            np.linspace(0.5, 359.5, self.nx),
            dims=('lon',),
            attrs={'units': 'degrees east', 'long_name': 'longitude'},
        )
        lats = xr.DataArray(
            np.linspace(-89.5, 89.5, self.ny),
            dims=('lat',),
            attrs={'units': 'degrees north', 'long_name': 'latitude'},
        )
        self.ds['var'] = xr.DataArray(
            randn(
                (self.nmember, self.ninit, self.nlead, self.nx, self.ny),
                frac_nan=FRAC_NAN,
            ),
            coords={
                'member': members,
                'init': inits,
                'lon': lons,
                'lat': lats,
                'lead': leads,
            },
            dims=('member', 'init', 'lead', 'lon', 'lat'),
            name='var',
            attrs={'units': 'var units', 'description': 'a description'},
        )
        self.control['var'] = xr.DataArray(
            randn((self.ntime, self.nx, self.ny), frac_nan=FRAC_NAN),
            coords={'lon': lons, 'lat': lats, 'time': times},
            dims=('time', 'lon', 'lat'),
            name='var',
            attrs={'units': 'var units', 'description': 'a description'},
        )

        self.ds.attrs = {'history': 'created for xarray benchmarking'}


class Compute(Generate):
    """
    Benchmark time and peak memory of `compute_perfect_model` and
    `bootstrap_perfect_model`.
    """

    def setup(self, *args, **kwargs):
        self.make_initialized_control()

    @parameterized(['metric', 'comparison'], (METRICS, PM_COMPARISONS))
    def time_compute_perfect_model(self, metric, comparison):
        """Take time for `compute_perfect_model`."""
        ensure_loaded(
            compute_perfect_model(
                self.ds, self.control, metric=metric, comparison=comparison
            )
        )

    @parameterized(['metric', 'comparison'], (METRICS, PM_COMPARISONS))
    def peakmem_compute_perfect_model(self, metric, comparison):
        """Take memory peak for `compute_perfect_model`."""
        ensure_loaded(
            compute_perfect_model(
                self.ds, self.control, metric=metric, comparison=comparison
            )
        )

    @parameterized(['metric', 'comparison'], (METRICS, PM_COMPARISONS))
    def time_bootstrap_perfect_model(self, metric, comparison):
        """Take time for `bootstrap_perfect_model`."""
        ensure_loaded(
            bootstrap_perfect_model(
                self.ds,
                self.control,
                metric=metric,
                comparison=comparison,
                bootstrap=BOOTSTRAP,
            )
        )

    @parameterized(['metric', 'comparison'], (METRICS, PM_COMPARISONS))
    def peakmem_bootstrap_perfect_model(self, metric, comparison):
        """Take memory peak for `bootstrap_perfect_model`."""
        ensure_loaded(
            bootstrap_perfect_model(
                self.ds,
                self.control,
                metric=metric,
                comparison=comparison,
                bootstrap=BOOTSTRAP,
            )
        )


class ComputeDask(Compute):
    def setup(self, *args, **kwargs):
        """Benchmark time and peak memory of `compute_hindcast` and
        `bootstrap_hindcast`. This executes the same tests as `Compute` but on chunked
        data."""
        requires_dask()
        # magic taken from
        # https://github.com/pydata/xarray/blob/stable/asv_bench/benchmarks/rolling.py
        super().setup(**kwargs)
        # chunk along a spatial dimension to enable embarrasingly parallel computation
        self.ds = self.ds['var'].chunk({'lon': self.nx // BOOTSTRAP})
        self.control = self.control['var'].chunk({'lon': self.nx // BOOTSTRAP})
