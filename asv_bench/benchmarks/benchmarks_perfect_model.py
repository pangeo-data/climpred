import numpy as np
import xarray as xr
from dask.distributed import Client

from climpred.bootstrap import bootstrap_perfect_model
from climpred.metrics import PROBABILISTIC_METRICS
from climpred.prediction import compute_perfect_model

from . import ensure_loaded, parameterized, randn, requires_dask

# only take subselection of all possible metrics
METRICS = ["rmse", "pearson_r", "crpss"]
# only take comparisons compatible with probabilistic metrics
PM_COMPARISONS = ["m2m", "m2c"]

ITERATIONS = 16


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
        self.nmember = 5
        self.ninit = 6
        self.nlead = 10
        self.iterations = ITERATIONS
        self.nx = 72
        self.ny = 36
        self.control_start = 3000
        self.control_end = 3300
        self.ntime = self.control_end - self.control_start
        self.client = None

        FRAC_NAN = 0.0

        times = xr.cftime_range(
            start=str(self.control_start),
            periods=self.ntime,
            freq="YS",
            calendar="noleap",
        )
        leads = np.arange(1, 1 + self.nlead)
        members = np.arange(1, 1 + self.nmember)
        inits = xr.cftime_range(
            start=str(self.control_start),
            periods=self.ninit,
            freq="10YS",
            calendar="noleap",
        )

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
        self.ds["var"] = xr.DataArray(
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
        self.control["var"] = xr.DataArray(
            randn((self.ntime, self.nx, self.ny), frac_nan=FRAC_NAN),
            coords={"lon": lons, "lat": lats, "time": times},
            dims=("time", "lon", "lat"),
            name="var",
            attrs={"units": "var units", "description": "a description"},
        )

        self.ds.attrs = {"history": "created for xarray benchmarking"}
        self.ds.lead.attrs["units"] = "years"
        self.control.time.attrs["units"] = "years"


class Compute(Generate):
    """
    Benchmark time and peak memory of `compute_perfect_model` and
    `bootstrap_perfect_model`.
    """

    def setup(self, *args, **kwargs):
        self.make_initialized_control()

    @parameterized(["metric", "comparison"], (METRICS, PM_COMPARISONS))
    def time_compute_perfect_model(self, metric, comparison):
        """Take time for `compute_perfect_model`."""
        dim = "member" if metric in PROBABILISTIC_METRICS else None
        ensure_loaded(
            compute_perfect_model(
                self.ds, self.control, metric=metric, comparison=comparison, dim=dim
            )
        )

    @parameterized(["metric", "comparison"], (METRICS, PM_COMPARISONS))
    def peakmem_compute_perfect_model(self, metric, comparison):
        """Take memory peak for `compute_perfect_model`."""
        dim = "member" if metric in PROBABILISTIC_METRICS else None
        ensure_loaded(
            compute_perfect_model(
                self.ds, self.control, metric=metric, comparison=comparison, dim=dim
            )
        )

    @parameterized(["metric", "comparison"], (METRICS, PM_COMPARISONS))
    def time_bootstrap_perfect_model(self, metric, comparison):
        """Take time for `bootstrap_perfect_model`."""
        dim = "member" if metric in PROBABILISTIC_METRICS else None
        ensure_loaded(
            bootstrap_perfect_model(
                self.ds,
                self.control,
                metric=metric,
                comparison=comparison,
                iterations=self.iterations,
                dim=dim,
            )
        )

    @parameterized(["metric", "comparison"], (METRICS, PM_COMPARISONS))
    def peakmem_bootstrap_perfect_model(self, metric, comparison):
        """Take memory peak for `bootstrap_perfect_model`."""
        dim = "member" if metric in PROBABILISTIC_METRICS else None
        ensure_loaded(
            bootstrap_perfect_model(
                self.ds,
                self.control,
                metric=metric,
                comparison=comparison,
                iterations=self.iterations,
                dim=dim,
            )
        )


class ComputeDask(Compute):
    def setup(self, *args, **kwargs):
        """Benchmark time and peak memory of `compute_perfect_model` and
        `bootstrap_perfect_model`. This executes the same tests as `Compute` but
        on chunked data."""
        requires_dask()
        # magic taken from
        # https://github.com/pydata/xarray/blob/stable/asv_bench/benchmarks/rolling.py
        super().setup(**kwargs)
        # chunk along a spatial dimension to enable embarrasingly parallel computation
        self.ds = self.ds["var"].chunk()
        self.control = self.control["var"].chunk()


class ComputeDaskDistributed(ComputeDask):
    def setup(self, *args, **kwargs):
        """Benchmark time and peak memory of `compute_perfect_model` and
        `bootstrap_perfect_model`. This executes the same tests as `Compute` but
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
        """Benchmark time and peak memory of `compute_perfect_model` and
        `bootstrap_perfect_model`. This executes the same tests as `Compute`
        but on 1D data."""
        # magic taken from
        # https://github.com/pydata/xarray/blob/stable/asv_bench/benchmarks/rolling.py
        super().setup(**kwargs)
        spatial_dims = ["lon", "lat"]
        self.ds = self.ds.mean(spatial_dims)
        self.control = self.control.mean(spatial_dims)
        self.iterations = 500
