import warnings

import numpy as np
import xarray as xr
from dask.distributed import Client

from climpred import HindcastEnsemble, PerfectModelEnsemble, set_options
from climpred.metrics import PROBABILISTIC_METRICS
from climpred.tutorial import load_dataset

from . import _skip_slow, ensure_loaded, parameterized, randn, requires_dask

# only take subselection of all possible metrics
METRICS = ["mse", "crps"]
REFERENCES = ["uninitialized", "climatology", "persistence"]
ITERATIONS = 8

set_options(climpred_warnings=False)


warnings.filterwarnings("ignore", message="Index.ravel returning ndarray is deprecated")


class Compute:
    """
    Benchmark time and peak memory of `PredictionEnsemble.verify` and
    `PredictionEnsemble.bootstrap`.
    """

    # https://asv.readthedocs.io/en/stable/benchmarks.html
    timeout = 300.0
    repeat = 1
    number = 5

    def setup(self, *args, **kwargs):
        raise NotImplementedError()

    def get_kwargs(self, metric=None, bootstrap=False):
        """Adjust kwargs for verify/bootstrap matching with metric."""
        if not isinstance(
            self.PredictionEnsemble, (PerfectModelEnsemble, HindcastEnsemble)
        ):
            raise NotImplementedError()
        dim = ["init", "member"] if metric in PROBABILISTIC_METRICS else "init"
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
            metric_kwargs["resample_dim"] = self.resample_dim
        if self.PredictionEnsemble.kind == "hindcast":
            metric_kwargs["alignment"] = self.alignment
        return metric_kwargs

    @parameterized(["metric"], (METRICS))
    def time_verify(self, metric):
        """Take time for `PredictionEnsemble.verify`."""
        ensure_loaded(
            self.PredictionEnsemble.verify(
                **self.get_kwargs(metric=metric, bootstrap=False)
            )
        )

    @parameterized(["metric"], (METRICS))
    def peakmem_verify(self, metric):
        """Take memory peak for `PredictionEnsemble.verify`."""
        ensure_loaded(
            self.PredictionEnsemble.verify(
                **self.get_kwargs(metric=metric, bootstrap=False)
            )
        )

    @parameterized(["metric"], (METRICS))
    def time_bootstrap(self, metric):
        """Take time for `PredictionEnsemble.bootstrap`."""
        ensure_loaded(
            self.PredictionEnsemble.bootstrap(
                **self.get_kwargs(metric=metric, bootstrap=True)
            )
        )

    @parameterized(["metric"], (METRICS))
    def peakmem_bootstrap(self, metric):
        """Take memory peak for `PredictionEnsemble.bootstrap`."""
        ensure_loaded(
            self.PredictionEnsemble.bootstrap(
                **self.get_kwargs(metric=metric, bootstrap=True)
            )
        )


class GenerateHindcastEnsemble(Compute):
    """
    Generate random input data.
    """

    def get_data(self, spatial_res=5):
        """Generates initialized hindcast, uninitialized historical and observational
        data, mimicking a hindcast experiment."""
        self.initialized = xr.Dataset()
        self.observations = xr.Dataset()
        self.uninitialized = xr.Dataset()

        self.nmember = 10
        self.nlead = 5
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
        ).squeeze()
        self.observations["var"] = xr.DataArray(
            randn((self.ninit, self.nx, self.ny), frac_nan=FRAC_NAN),
            coords={"lon": lons, "lat": lats, "time": inits},
            dims=("time", "lon", "lat"),
            name="var",
            attrs={"units": "var units", "description": "a description"},
        ).squeeze()

        self.uninitialized["var"] = xr.DataArray(
            randn((self.ninit, self.nx, self.ny, self.nmember), frac_nan=FRAC_NAN),
            coords={"lon": lons, "lat": lats, "time": inits, "member": members},
            dims=("time", "lon", "lat", "member"),
            name="var",
            attrs={"units": "var units", "description": "a description"},
        ).squeeze()

        self.initialized.attrs = {"history": "created for xarray benchmarking"}
        self.initialized.lead.attrs["units"] = "years"

        self.PredictionEnsemble = (
            HindcastEnsemble(self.initialized)
            .add_uninitialized(self.uninitialized)
            .add_observations(self.observations)
        )

    def setup(self, *args, **kwargs):
        self.get_data()
        self.alignment = "same_inits"
        self.reference = None
        self.resample_dim = "member"
        self.iterations = ITERATIONS


class GeneratePerfectModelEnsemble(GenerateHindcastEnsemble):
    """Generate `PerfectModelEnsemble` out of `HindcastEnsemble`."""

    def setup(self, *args, **kwargs):
        self.get_data()
        self.PredictionEnsemble = PerfectModelEnsemble(self.initialized).add_control(
            self.observations
        )
        self.PredictionEnsemble = self.PredictionEnsemble.generate_uninitialized()
        self.reference = None
        self.resample_dim = "member"
        self.iterations = ITERATIONS


class GenerateHindcastEnsembleSmall(GenerateHindcastEnsemble):
    """Generate single grid point `HindcastEnsemble`."""

    def setup(self, *args, **kwargs):
        self.get_data(spatial_res=360)
        self.PredictionEnsemble = (
            HindcastEnsemble(self.initialized)
            .add_uninitialized(self.uninitialized)
            .add_observations(self.observations)
        )
        self.alignment = "same_inits"
        self.resample_dim = "member"
        self.reference = None
        self.iterations = ITERATIONS


class GenerateHindcastEnsembleSmallReferences(GenerateHindcastEnsembleSmall):
    """Generate single grid point `HindcastEnsemble` with all references."""

    def setup(self, *args, **kwargs):
        _skip_slow()
        super().setup(**kwargs)
        self.reference = REFERENCES
        self.alignment = "maximize"
        self.reference = None
        self.resample_dim = "member"


class GeneratePerfectModelEnsembleSmall(GeneratePerfectModelEnsemble):
    """Generate single grid point `PerfectModelEnsemble`."""

    def setup(self, *args, **kwargs):
        self.get_data(spatial_res=360)
        self.PredictionEnsemble = PerfectModelEnsemble(self.initialized).add_control(
            self.observations
        )
        self.PredictionEnsemble = self.PredictionEnsemble.generate_uninitialized()
        self.alignment = None
        self.reference = None
        self.resample_dim = "member"
        self.iterations = ITERATIONS


class GeneratePerfectModelEnsembleSmallReferences(GeneratePerfectModelEnsembleSmall):
    """Generate single grid point `PerfectModelEnsemble` with all references."""

    def setup(self, *args, **kwargs):
        _skip_slow()
        super().setup(**kwargs)
        self.reference = REFERENCES
        self.alignment = None
        self.reference = None
        self.resample_dim = "member"
        self.iterations = ITERATIONS


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


class S2S(Compute):
    """Tutorial data from S2S project."""

    number = 3

    def get_data(self):
        _skip_slow()
        init = load_dataset("ECMWF_S2S_Germany").t2m.isel(lead=slice(None, None, 7))
        obs = load_dataset("Observations_Germany").t2m
        self.PredictionEnsemble = (
            HindcastEnsemble(init).add_observations(obs).generate_uninitialized()
        )

    def setup(self, *args, **kwargs):
        self.get_data()
        self.alignment = "maximize"
        self.resample_dim = "init"
        self.reference = None
        self.iterations = ITERATIONS


class NMME(Compute):
    """Tutorial data from NMME project."""

    def get_data(self):
        init = (
            load_dataset("NMME_hindcast_Nino34_sst")
            .isel(model=0)
            .sel(S=slice("1985", "2005"))
        )
        obs = load_dataset("NMME_OIv2_Nino34_sst")
        self.PredictionEnsemble = (
            HindcastEnsemble(init).add_observations(obs).generate_uninitialized()
        )

    def setup(self, *args, **kwargs):
        self.get_data()
        self.alignment = "maximize"
        self.resample_dim = "init"
        self.reference = None
        self.iterations = ITERATIONS
