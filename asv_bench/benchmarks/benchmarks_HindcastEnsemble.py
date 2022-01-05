import numpy as np
import xarray as xr
from dask.distributed import Client

from climpred import HindcastEnsemble
from climpred.metrics import PROBABILISTIC_METRICS

from . import ensure_loaded, parameterized, randn, requires_dask

# only take subselection of all possible metrics
METRICS = ["rmse", "crps"]
# only take comparisons compatible with probabilistic metrics
HINDCAST_COMPARISONS = ["m2o"]

ITERATIONS = 16


class Generate:
    """
    Generate input data for benchmark.
    """

    timeout = 600
    repeat = (2, 5, 20)

    def make_hindcast(self):
        """Generates initialized hindcast, uninitialized historical and observational
        data, mimicking a hindcast experiment."""
        self.initialized = xr.Dataset()
        self.observations = xr.Dataset()
        self.uninitialized = xr.Dataset()

        spatial_res = 2  # degrees
        self.nmember = 10
        self.nlead = 10
        self.nx = 360 // spatial_res
        self.ny = 360 // spatial_res
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
        self.initialized["var"] = xr.DataArray(
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

        self.uninitialized["var"] = xr.DataArray(
            randn((self.ninit, self.nx, self.ny, self.nmember), frac_nan=FRAC_NAN),
            coords={"lon": lons, "lat": lats, "time": inits, "member": members},
            dims=("time", "lon", "lat", "member"),
            name="var",
            attrs={"units": "var units", "description": "a description"},
        )

        self.initialized.attrs = {"history": "created for xarray benchmarking"}
        self.initialized.lead.attrs["units"] = "years"

        self.PredictionEnsemble = (
            HindcastEnsemble(self.initialized)
            .add_uninitialized(self.uninitialized)
            .add_observations(self.observations)
        )


class Compute(Generate):
    """
    Benchmark time and peak memory of `PredictionEnsemble.verify` and
    `PredictionEnsemble.bootstrap`.
    """

    def setup(self, *args, **kwargs):
        self.make_hindcast()
        self.alignment = "same_verif"
        self.reference = None  # ['uninitialized','climatology','persistence']

    @parameterized(["metric", "comparison"], (METRICS, HINDCAST_COMPARISONS))
    def time_PredictionEnsemble_verify(self, metric, comparison):
        """Take time for `PredictionEnsemble.verify`."""
        dim = "member" if metric in PROBABILISTIC_METRICS else "init"
        ensure_loaded(
            self.PredictionEnsemble.verify(
                metric=metric,
                comparison=comparison,
                dim=dim,
                alignment=self.alignment,
                reference=self.reference,
            )
        )

    @parameterized(["metric", "comparison"], (METRICS, HINDCAST_COMPARISONS))
    def peakmem_PredictionEnsemble_verify(self, metric, comparison):
        """Take memory peak for `PredictionEnsemble.verify`."""
        dim = "member" if metric in PROBABILISTIC_METRICS else "init"
        ensure_loaded(
            self.PredictionEnsemble.verify(
                metric=metric,
                comparison=comparison,
                dim=dim,
                alignment=self.alignment,
                reference=self.reference,
            )
        )

    @parameterized(["metric", "comparison"], (METRICS, HINDCAST_COMPARISONS))
    def time_PredictionEnsemble_bootstrap(self, metric, comparison):
        """Take time for `PredictionEnsemble.bootstrap`."""
        dim = "member" if metric in PROBABILISTIC_METRICS else "init"
        ensure_loaded(
            self.PredictionEnsemble.bootstrap(
                metric=metric,
                comparison=comparison,
                iterations=self.iterations,
                dim=dim,
                alignment=self.alignment,
                reference=self.reference,
            )
        )

    @parameterized(["metric", "comparison"], (METRICS, HINDCAST_COMPARISONS))
    def peakmem_PredictionEnsemble_bootstrap(self, metric, comparison):
        """Take memory peak for `PredictionEnsemble.bootstrap`."""
        dim = "member" if metric in PROBABILISTIC_METRICS else "init"
        ensure_loaded(
            self.PredictionEnsemble.bootstrap(
                metric=metric,
                comparison=comparison,
                iterations=self.iterations,
                dim=dim,
                alignment=self.alignment,
                reference=self.reference,
            )
        )


class ComputeDask(Compute):
    def setup(self, *args, **kwargs):
        """Benchmark time and peak memory of `PredictionEnsemble.verify` and
        `PredictionEnsemble.bootstrap`. This executes the same tests as `Compute` but
        on chunked data."""
        requires_dask()
        super().setup(**kwargs)
        # chunk along a spatial dimension to enable embarrasingly parallel computation
        self.PredictionEnsemble = self.PredictionEnsemble.chunk({"lead": 1}).chunk(
            {"lon": "auto"}
        )


class ComputeDaskDistributed(ComputeDask):
    def setup(self, *args, **kwargs):
        """Benchmark time and peak memory of `PredictionEnsemble.verify` and
        `PredictionEnsemble.bootstrap`. This executes the same tests as `Compute` but
        on chunked data with dask.distributed.Client."""
        requires_dask()
        super().setup(**kwargs)
        self.client = Client()

    def cleanup(self):
        self.client.shutdown()


class ComputeSmall(Compute):
    def setup(self, *args, **kwargs):
        """Benchmark time and peak memory of `PredictionEnsemble.verify` and
        `PredictionEnsemble.bootstrap`. This executes the same tests as `Compute` but on 1D
        data."""
        requires_dask()
        super().setup(**kwargs)
        self.PredictionEnsemble = self.PredictionEnsemble.isel(lon=0, lat=0)
        self.iterations = 500
