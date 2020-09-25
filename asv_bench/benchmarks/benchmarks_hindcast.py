import numpy as np
import xarray as xr
from dask.distributed import Client

from climpred.bootstrap import bootstrap_hindcast
from climpred.metrics import PROBABILISTIC_METRICS
from climpred.prediction import compute_hindcast

from . import ensure_loaded, parameterized, randn, requires_dask

# only take subselection of all possible metrics
METRICS = ["rmse", "pearson_r", "crpss"]
# only take comparisons compatible with probabilistic metrics
HINDCAST_COMPARISONS = ["m2o"]

ITERATIONS = 16


class Generate:
    """
    Generate input data for benchmark.
    """

    timeout = 600
    repeat = (2, 5, 20)

    def make_hind_obs(self):
        """Generates initialized hindcast, uninitialized historical and observational
        data, mimicking a hindcast experiment."""
        self.hind = xr.Dataset()
        self.observations = xr.Dataset()
        self.uninit = xr.Dataset()

        self.nmember = 3
        self.nlead = 5
        self.nx = 72
        self.ny = 36
        self.iterations = ITERATIONS
        self.init_start = 1960
        self.init_end = 2000
        self.ninit = self.init_end - self.init_start
        self.client = None

        FRAC_NAN = 0.0

        inits = xr.cftime_range(
            start=str(self.init_start), end=str(self.init_end - 1), freq="YS"
        )
        leads = np.arange(1, 1 + self.nlead)
        members = np.arange(1, 1 + self.nmember)

        lons = xr.DataArray(
            np.linspace(0.5, 359.5, self.nx),
            dims=("lon",),
            attrs={"units": "degrees east", "long_name": "longitude"},
        )
        lats = xr.DataArray(
            np.linspace(-89.5, 89.5, self.ny),
            dims=("lat",),
            attrs={"units": "degrees north", "long_name": "latitude"},
        )
        self.hind["var"] = xr.DataArray(
            randn(
                (self.nmember, self.ninit, self.nlead, self.nx, self.ny),
                frac_nan=FRAC_NAN,
            ),
            coords={
                "member": members,
                "init": inits,
                "lon": lons,
                "lat": lats,
                "lead": leads,
            },
            dims=("member", "init", "lead", "lon", "lat"),
            name="var",
            attrs={"units": "var units", "description": "a description"},
        )
        self.observations["var"] = xr.DataArray(
            randn((self.ninit, self.nx, self.ny), frac_nan=FRAC_NAN),
            coords={"lon": lons, "lat": lats, "time": inits},
            dims=("time", "lon", "lat"),
            name="var",
            attrs={"units": "var units", "description": "a description"},
        )

        self.uninit["var"] = xr.DataArray(
            randn((self.ninit, self.nx, self.ny, self.nmember), frac_nan=FRAC_NAN),
            coords={"lon": lons, "lat": lats, "time": inits, "member": members},
            dims=("time", "lon", "lat", "member"),
            name="var",
            attrs={"units": "var units", "description": "a description"},
        )

        self.hind.attrs = {"history": "created for xarray benchmarking"}
        self.hind.lead.attrs["units"] = "years"
        self.uninit.time.attrs["units"] = "years"
        self.observations.time.attrs["units"] = "years"


class Compute(Generate):
    """
    Benchmark time and peak memory of `compute_hindcast` and `bootstrap_hindcast`.
    """

    def setup(self, *args, **kwargs):
        self.make_hind_obs()

    @parameterized(["metric", "comparison"], (METRICS, HINDCAST_COMPARISONS))
    def time_compute_hindcast(self, metric, comparison):
        """Take time for `compute_hindcast`."""
        dim = "member" if metric in PROBABILISTIC_METRICS else "init"
        ensure_loaded(
            compute_hindcast(
                self.hind,
                self.observations,
                metric=metric,
                comparison=comparison,
                dim=dim,
            )
        )

    @parameterized(["metric", "comparison"], (METRICS, HINDCAST_COMPARISONS))
    def peakmem_compute_hindcast(self, metric, comparison):
        """Take memory peak for `compute_hindcast`."""
        dim = "member" if metric in PROBABILISTIC_METRICS else "init"
        ensure_loaded(
            compute_hindcast(
                self.hind,
                self.observations,
                metric=metric,
                comparison=comparison,
                dim=dim,
            )
        )

    @parameterized(["metric", "comparison"], (METRICS, HINDCAST_COMPARISONS))
    def time_bootstrap_hindcast(self, metric, comparison):
        """Take time for `bootstrap_hindcast`."""
        dim = "member" if metric in PROBABILISTIC_METRICS else "init"
        ensure_loaded(
            bootstrap_hindcast(
                self.hind,
                self.uninit,
                self.observations,
                metric=metric,
                comparison=comparison,
                iterations=self.iterations,
                dim=dim,
            )
        )

    @parameterized(["metric", "comparison"], (METRICS, HINDCAST_COMPARISONS))
    def peakmem_bootstrap_hindcast(self, metric, comparison):
        """Take memory peak for `bootstrap_hindcast`."""
        dim = "member" if metric in PROBABILISTIC_METRICS else "init"
        ensure_loaded(
            bootstrap_hindcast(
                self.hind,
                self.uninit,
                self.observations,
                metric=metric,
                comparison=comparison,
                iterations=self.iterations,
                dim=dim,
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
        self.hind = self.hind["var"].chunk()
        self.observations = self.observations["var"].chunk()
        self.uninit = self.uninit["var"].chunk()


class ComputeDaskDistributed(ComputeDask):
    def setup(self, *args, **kwargs):
        """Benchmark time and peak memory of `compute_hindcast` and
        `bootstrap_hindcast`. This executes the same tests as `Compute` but
        on chunked data with dask.distributed.Client."""
        requires_dask()
        # magic taken from
        # https://github.com/pydata/xarray/blob/stable/asv_bench/benchmarks/rolling.py
        super().setup(**kwargs)
        self.client = Client()

    def cleanup(self):
        self.client.shutdown()


class ComputeSmall(Compute):
    def setup(self, *args, **kwargs):
        """Benchmark time and peak memory of `compute_hindcast` and
        `bootstrap_hindcast`. This executes the same tests as `Compute` but on 1D
        data."""
        requires_dask()
        # magic taken from
        # https://github.com/pydata/xarray/blob/stable/asv_bench/benchmarks/rolling.py
        super().setup(**kwargs)
        # chunk along a spatial dimension to enable embarrasingly parallel computation
        spatial_dims = ["lon", "lat"]
        self.hind = self.hind.mean(spatial_dims)
        self.observations = self.observations.mean(spatial_dims)
        self.uninit = self.uninit.mean(spatial_dims)
        self.iterations = 500
