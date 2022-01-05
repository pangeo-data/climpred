import numpy as np
import xarray as xr
from dask.distributed import Client

from climpred import HindcastEnsemble, PerfectModelEnsemble
from climpred.metrics import PROBABILISTIC_METRICS
from climpred.tutorial import load_dataset

from . import _skip_slow, ensure_loaded, parameterized, randn, requires_dask

# only take subselection of all possible metrics
METRICS = ["rmse", "crps"]
REFERENCES = ["uninitialized", "climatology", "persistence"]
ITERATIONS = 16


class Compute:
    """
    Benchmark time and peak memory of `PredictionEnsemble.verify` and
    `PredictionEnsemble.bootstrap`.
    """

    timeout = 600
    repeat = (2, 5, 10)  # https://asv.readthedocs.io/en/stable/benchmarks.html

    def setup(self, *args, **kwargs):
        # self.get_data()
        self.alignment = None
        self.reference = None  # ['uninitialized','climatology','persistence']

    def get_kwargs(self, metric=None, bootstrap=False):
        """Adjust kwargs for verify/bootstrap matching with metric."""
        dim = "member" if metric in PROBABILISTIC_METRICS else "init"
        if self.PredictionEnsemble.kind == "hindcast":
            comparison = "m2o" if metric in PROBABILISTIC_METRICS else "e2o"
        elif self.PredictionEnsemble.kind == "perfect":
            comparison = "m2c" if metric in PROBABILISTIC_METRICS else "m2e"
        metric_kwargs = dict(
            metric=metric,
            comparison=comparison,
            dim=dim,
            reference=self.reference,
        )
        if bootstrap:
            metric_kwargs["iterations"] = self.iterations
        if self.PredictionEnsemble.kind == "hindcast":
            metric_kwargs["alignment"] = self.alignment
        return metric_kwargs

    @parameterized(["metric"], (METRICS))
    def time_PredictionEnsemble_verify(self, metric):
        """Take time for `PredictionEnsemble.verify`."""
        ensure_loaded(self.PredictionEnsemble.verify(**self.get_kwargs(metric=metric)))

    @parameterized(["metric"], (METRICS))
    def peakmem_PredictionEnsemble_verify(self, metric):
        """Take memory peak for `PredictionEnsemble.verify`."""
        ensure_loaded(self.PredictionEnsemble.verify(**self.get_kwargs(metric=metric)))

    @parameterized(["metric"], (METRICS))
    def time_PredictionEnsemble_bootstrap(self, metric):
        """Take time for `PredictionEnsemble.bootstrap`."""
        _skip_slow()
        ensure_loaded(
            self.PredictionEnsemble.bootstrap(
                **self.get_kwargs(metric=metric, bootstrap=True)
            )
        )

    @parameterized(["metric"], (METRICS))
    def peakmem_PredictionEnsemble_bootstrap(self, metric):
        """Take memory peak for `PredictionEnsemble.bootstrap`."""
        _skip_slow()
        ensure_loaded(
            self.PredictionEnsemble.bootstrap(
                **self.get_kwargs(metric=metric, bootstrap=True)
            )
        )


class GenerateHindcastEnsemble(Compute):
    """
    Generate random input data.
    """

    def get_data(self):
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

    def setup(self, *args, **kwargs):
        _skip_slow()
        self.get_data()
        self.alignment = "same_verif"
        self.reference = None


class GeneratePerfectModelEnsemble(GenerateHindcastEnsemble):
    """Generate `PerfectModelEnsemble` out of `HindcastEnsemble`."""

    def setup(self, *args, **kwargs):
        _skip_slow()
        self.get_data()
        self.PredictionEnsemble = PerfectModelEnsemble(self.initialized).add_control(
            self.observations
        )
        self.PredictionEnsemble = self.PredictionEnsemble.generate_uninitialized()
        self.alignment = None
        self.reference = None


class GenerateSmallHindcastEnsemble(GenerateHindcastEnsemble):
    """Generate single grid point `HindcastEnsemble`."""

    def setup(self, *args, **kwargs):
        super().setup(**kwargs)
        self.PredictionEnsemble = self.PredictionEnsemble.isel(lon=0, lat=0)


class GenerateSmallReferencesHindcastEnsemble(GenerateSmallHindcastEnsemble):
    """Generate single grid point `HindcastEnsemble` with all references."""

    def setup(self, *args, **kwargs):
        _skip_slow()
        super().setup(**kwargs)
        self.reference = REFERENCES


class GenerateSmallPerfectModelEnsemble(GeneratePerfectModelEnsemble):
    """Generate single grid point `PerfectModelEnsemble`."""

    def setup(self, *args, **kwargs):
        _skip_slow()
        super().setup(**kwargs)
        self.PredictionEnsemble = self.PredictionEnsemble.isel(lon=0, lat=0)


class GenerateSmallReferencesPerfectModelEnsemble(GenerateSmallPerfectModelEnsemble):
    """Generate single grid point `PerfectModelEnsemble`."""

    def setup(self, *args, **kwargs):
        _skip_slow()
        super().setup(**kwargs)
        self.reference = REFERENCES


class GenerateHindcastEnsembleDask(GenerateHindcastEnsemble):
    def setup(self, *args, **kwargs):
        """The same tests but on spatially chunked data."""
        _skip_slow()
        requires_dask()
        super().setup(**kwargs)
        # chunk along a spatial dimension to enable embarrasingly parallel computation
        self.PredictionEnsemble = self.PredictionEnsemble.chunk({"lead": 1}).chunk(
            {"lon": "auto"}
        )


class GenerateHindcastEnsembleDaskDistributed(GenerateHindcastEnsembleDask):
    def setup(self, *args, **kwargs):
        """The same tests but on spatially chunked data with dask.distributed.Client."""
        _skip_slow()
        requires_dask()
        super().setup(**kwargs)
        self.client = Client()

    def cleanup(self):
        self.client.shutdown()


class GenerateHindcastEnsembleSmall(GenerateHindcastEnsemble):
    def setup(self, *args, **kwargs):
        """The same tests but on 1D data."""
        requires_dask()
        super().setup(**kwargs)
        self.PredictionEnsemble = self.PredictionEnsemble.isel(lon=0, lat=0)
        self.iterations = 500


class S2S(Compute):
    """Tutorial data from S2S project."""

    def get_data(self):
        init = load_dataset("ECMWF_S2S_Germany").t2m
        obs = load_dataset("Observations_Germany").t2m
        self.PredictionEnsemble = HindcastEnsemble(init).add_observations(obs)

    def setup(self, *args, **kwargs):
        _skip_slow()
        self.get_data()
        self.alignment = "same_inits"
        self.reference = None  # ['uninitialized','climatology','persistence']
        self.iterations = 16


class NMME(Compute):
    """Tutorial data from NMME project."""

    def get_data(self):
        init = load_dataset("NMME_hindcast_Nino34_sst")
        obs = load_dataset("NMME_OIv2_Nino34_sst")
        self.PredictionEnsemble = HindcastEnsemble(init).add_observations(obs)

    def setup(self, *args, **kwargs):
        _skip_slow()
        self.get_data()
        self.alignment = "same_inits"
        self.reference = None  # ['uninitialized','climatology','persistence']
        self.iterations = 16
