"""Bootstrap or resampling operators for functional compute_ functions."""

import logging
import warnings
from copy import copy

import dask
import numpy as np
import xarray as xr
import xskillscore as xs

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None
from xskillscore.core.resampling import (
    resample_iterations as _resample_iterations,
    resample_iterations_idx as _resample_iterations_idx,
)

from .checks import (
    has_dims,
    has_valid_lead_units,
    warn_if_chunking_would_increase_performance,
)
from .constants import CLIMPRED_DIMS, CONCAT_KWARGS
from .options import OPTIONS
from .stats import dpp

try:
    from .stats import varweighted_mean_period
except ImportError:
    varweighted_mean_period = None  # type: ignore
from .utils import (
    _transpose_and_rechunk_to,
    find_start_dates_for_given_init,
    get_lead_cftime_shift_args,
    lead_units_equal_control_time_stride,
    rechunk_to_single_chunk_if_more_than_one_chunk_along_dim,
    shift_cftime_singular,
)


def _p_ci_from_sig(sig):
    """Convert significance level sig:float=95 to p-values:float=0.025-0.975."""
    p = (100 - sig) / 100
    ci_low = p / 2
    ci_high = 1 - p / 2
    return p, ci_low, ci_high


def _resample(initialized, resample_dim):
    """Resample with replacement in dimension ``resample_dim``.

    Args:
        initialized (xr.Dataset): input xr.Dataset to be resampled.
        resample_dim (str): dimension to resample along.

    Returns:
        xr.Dataset: resampled along ``resample_dim``.

    """
    to_be_resampled = initialized[resample_dim].values
    smp = np.random.choice(to_be_resampled, len(to_be_resampled))
    smp_initialized = initialized.sel({resample_dim: smp})
    # ignore because then inits should keep their labels
    if resample_dim != "init":
        smp_initialized[resample_dim] = initialized[resample_dim].values
    return smp_initialized


def _distribution_to_ci(ds, ci_low, ci_high, dim="iteration"):
    """Get confidence intervals from bootstrapped distribution.

    Needed for bootstrapping confidence intervals and p_values of a metric.

    Args:
        ds (xr.Dataset): distribution.
        ci_low (float): low confidence interval.
        ci_high (float): high confidence interval.
        dim (str): dimension to apply xr.quantile to. Defaults to: "iteration"

    Returns:
        uninit_initialized (xr.Dataset): uninitialize initializedcast with
            initialized.coords.
    """
    ds = rechunk_to_single_chunk_if_more_than_one_chunk_along_dim(ds, dim)
    if isinstance(ds, xr.Dataset):
        for v in ds.data_vars:
            if np.issubdtype(ds[v].dtype, np.bool_):
                ds[v] = ds[v].astype(np.float_)  # fails on py>36 if boolean dtype
    else:
        if np.issubdtype(ds.dtype, np.bool_):
            ds = ds.astype(np.float_)  # fails on py>36 if boolean dtype
    return ds.quantile(q=[ci_low, ci_high], dim=dim, skipna=False)


def _pvalue_from_distributions(ref_skill, init_skill, metric=None):
    """Get probability that reference forecast skill is larger than initialized skill.

    Needed for bootstrapping confidence intervals and p_values of a metric in
    the hindcast framework. Checks whether a simple forecast like persistence,
    climatology or uninitialized performs better than initialized forecast. Need to
    keep in mind the orientation of metric (whether larger values are better or worse
    than smaller ones.)

    Args:
        ref_skill (xr.Dataset): persistence or uninitialized skill.
        init_skill (xr.Dataset): initialized skill.
        metric (Metric): metric class Metric

    Returns:
        pv (xr.Dataset): probability that simple forecast performs better
            than initialized forecast.
    """
    pv = ((ref_skill - init_skill) > 0).mean("iteration")
    if not metric.positive:
        pv = 1 - pv
    return pv


def bootstrap_uninitialized_ensemble(initialized, hist):
    """Resample uninitialized hindcast from historical members.

    Note:
        Needed for bootstrapping confidence intervals and p_values of a metric in
        the hindcast framework. Takes initialized.lead.size timesteps from historical at
        same forcing and rearranges them into ensemble and member dimensions.

    Args:
        initialized (xr.Dataset): hindcast.
        hist (xr.Dataset): historical uninitialized.

    Returns:
        uninit_initialized (xr.Dataset): uninitialize hindcast with initialized.coords.
    """
    has_dims(hist, "member", "historical ensemble")
    has_dims(initialized, "member", "initialized hindcast ensemble")
    # Put this after `convert_time_index` since it assigns "years" attribute if the
    # `init` dimension is a `float` or `int`.
    has_valid_lead_units(initialized)

    # find range for bootstrapping
    first_init = max(hist.time.min(), initialized["init"].min())

    n, freq = get_lead_cftime_shift_args(
        initialized.lead.attrs["units"], initialized.lead.size
    )
    hist_last = shift_cftime_singular(hist.time.max(), -1 * n, freq)
    last_init = min(hist_last, initialized["init"].max())

    initialized = initialized.sel(init=slice(first_init, last_init))

    uninit_initialized = []
    for init in initialized.init.values:
        # take uninitialized members from hist at init forcing
        # (:cite:t:`Goddard2013` allows 5 year forcing range here)
        uninit_at_one_init_year = hist.sel(
            time=slice(
                shift_cftime_singular(init, 1, freq),
                shift_cftime_singular(init, n, freq),
            ),
        ).rename({"time": "lead"})
        uninit_at_one_init_year["lead"] = np.arange(
            1, 1 + uninit_at_one_init_year["lead"].size
        )
        uninit_initialized.append(uninit_at_one_init_year)
    uninit_initialized = xr.concat(uninit_initialized, "init")
    uninit_initialized["init"] = initialized["init"].values
    uninit_initialized.lead.attrs["units"] = initialized.lead.attrs["units"]
    uninit_initialized["member"] = hist["member"].values
    return (
        _transpose_and_rechunk_to(
            uninit_initialized,
            initialized.isel(member=[0] * uninit_initialized.member.size),
        )
        if dask.is_dask_collection(uninit_initialized)
        else uninit_initialized
    )


def bootstrap_uninit_pm_ensemble_from_control_cftime(init_pm, control):
    """Create a pseudo-ensemble from control run.

    Bootstrap random numbers for years to construct an uninitialized ensemble from.
    This assumes a continous control simulation without gaps.

    Note:
        Needed for block bootstrapping  a metric in perfect-model framework. Takes
        random segments of length ``block_length`` from control based on ``dayofyear``
        (and therefore assumes a constant climate control simulation) and rearranges
        them into ensemble and member dimensions.

    Args:
        init_pm (xr.Dataset): initialized ensemble simulation.
        control (xr.Dataset): control simulation.

    Returns:
        uninit_pm (xr.Dataset): uninitialized ensemble generated from control run.
    """
    lead_units_equal_control_time_stride(init_pm, control)
    # short cut if annual leads
    if init_pm.lead.attrs["units"] == "years":
        return _bootstrap_by_stacking(init_pm, control)

    block_length = init_pm.lead.size
    freq = get_lead_cftime_shift_args(init_pm.lead.attrs["units"], block_length)[1]
    nmember = init_pm.member.size
    # start and end years possible to resample the actual uninitialized ensembles from
    c_start_year = control.time.min().dt.year.astype("int")
    # dont resample from years that control wont have timesteps for all leads
    c_end_year = (
        shift_cftime_singular(control.time.max(), -block_length, freq).dt.year.astype(
            "int"
        )
        - 1
    )

    def sel_time(start_year_int, suitable_start_dates):
        """Select time of control from suitable_start_dates based on start_year_int."""
        start_time = suitable_start_dates.time.sel(time=str(start_year_int))
        end_time = shift_cftime_singular(start_time, block_length - 1, freq)
        new = control.sel(time=slice(*start_time, *end_time))
        new["time"] = init_pm.lead.values
        return new

    def create_pseudo_members(init):
        """For every initialization take a different set of start years."""
        startlist = np.random.randint(c_start_year, c_end_year, nmember)
        suitable_start_dates = find_start_dates_for_given_init(control, init)
        return xr.concat(
            (sel_time(start, suitable_start_dates) for start in startlist),
            dim="member",
            **CONCAT_KWARGS,
        )

    uninit = xr.concat(
        (create_pseudo_members(init) for init in init_pm.init),
        dim="init",
        **CONCAT_KWARGS,
    ).rename({"time": "lead"})
    uninit["member"] = init_pm.member.values
    uninit["lead"] = init_pm.lead
    # chunk to same dims
    transpose_kwargs = (
        {"transpose_coords": False} if isinstance(init_pm, xr.DataArray) else {}
    )
    uninit = uninit.transpose(*init_pm.dims, **transpose_kwargs)
    return (
        _transpose_and_rechunk_to(uninit, init_pm)
        if dask.is_dask_collection(uninit)
        else uninit
    )


def resample_uninitialized_from_initialized(init, resample_dim=["init", "member"]):
    """Generate ``uninitialized`` by resampling from ``initialized``.

    Generate an uninitialized ensemble by resampling without replacement from the
    initialized prediction ensemble. Full years of the first lead present from the
    initialized are relabeled to a different year.
    """
    if (init.init.dt.year.groupby("init.year").count().diff("year") != 0).any():
        raise ValueError(
            "`resample_uninitialized_from_initialized` only works if the same number "
            " of initializations is present each year, found "
            f'{init.init.dt.year.groupby("init.year").count()}.'
        )
    if "init" not in resample_dim:
        raise ValueError(
            f"Only resampling on `init` makes forecasts uninitialzed."
            f"Found resample_dim={resample_dim}."
        )
    init = init.isel(lead=0, drop=True)
    # resample init
    init_notnull = init.where(init.notnull(), drop=True)
    full_years = list(set(init_notnull.init.dt.year.values))

    years_same = True
    while years_same:
        m = full_years.copy()
        np.random.shuffle(m)
        years_same = (np.array(m) - np.array(full_years) == 0).any()

    resampled_inits = xr.concat([init.sel(init=str(i)).init for i in m], "init")
    resampled_uninit = init.sel(init=resampled_inits)
    resampled_uninit["init"] = init_notnull.sel(
        init=slice(str(full_years[0]), str(full_years[-1]))
    ).init
    # take time dim and overwrite with sorted
    resampled_uninit = (
        resampled_uninit.swap_dims({"init": "valid_time"})
        .drop_vars("init")
        .rename({"valid_time": "time"})
    )
    resampled_uninit = resampled_uninit.assign_coords(
        time=resampled_uninit.time.sortby("time").values
    )

    # resample members
    if "member" in resample_dim:
        resampled_members = np.random.randint(0, init.member.size, init.member.size)
        resampled_uninit = resampled_uninit.isel(member=resampled_members)
        resampled_uninit["member"] = init.member

    from . import __version__ as version

    resampled_uninit.attrs.update(
        {
            "description": (
                "created by `HindcastEnsemble.generate_uninitialized()` "
                " resampling years without replacement from initialized"
            ),
            "documentation": f"https://climpred.readthedocs.io/en/v{version}/api/climpred.classes.HindcastEnsemble.generate_uninitialized.html#climpred.classes.HindcastEnsemble.generate_uninitialized",  # noqa: E501
        }
    )
    return resampled_uninit


def _bootstrap_by_stacking(init_pm, control):
    """Bootstrap member, lead, init from control by reshaping.

    Fast track of function
    `bootstrap_uninit_pm_ensemble_from_control_cftime` when lead units is 'years'.
    """
    assert isinstance(init_pm, type(control))
    lead_unit = init_pm.lead.attrs["units"]
    if isinstance(init_pm, xr.Dataset):
        init_pm = init_pm.to_array()
        init_was_dataset = True
    else:
        init_was_dataset = False
    if isinstance(control, xr.Dataset):
        control = control.to_array()

    init_size = init_pm.init.size * init_pm.member.size * init_pm.lead.size
    # select random start points
    new_time = np.random.randint(
        0, control.time.size - init_pm.lead.size, init_size // (init_pm.lead.size)
    )
    new_time = np.array(
        [np.arange(s, s + init_pm.lead.size) for s in new_time]
    ).flatten()[:init_size]
    larger = control.isel(time=new_time)
    fake_init = init_pm.stack(time=tuple(d for d in init_pm.dims if d in CLIMPRED_DIMS))
    # exchange values
    transpose_kwargs = (
        {"transpose_coords": False} if isinstance(init_pm, xr.DataArray) else {}
    )
    larger = larger.transpose(*fake_init.dims, **transpose_kwargs)
    fake_init.data = larger.data
    fake_uninit = fake_init.unstack()
    if init_was_dataset:
        fake_uninit = fake_uninit.to_dataset(dim="variable")
    fake_uninit["lead"] = init_pm["lead"]
    fake_uninit.lead.attrs["units"] = lead_unit
    return fake_uninit


def _get_resample_func(ds):
    """Decide for resample function based on input `ds`.

    Returns:
      callable: `_resample_iterations`: if big and chunked `ds`
                `_resample_iterations_idx`: else (if small and eager `ds`)
    """
    if OPTIONS["resample_iterations_func"] == "default":
        resample_func = (
            _resample_iterations
            if (
                dask.is_dask_collection(ds)
                and len(ds.dims) > 3
                # > 2MB
                and ds.nbytes > 2000000
            )
            else _resample_iterations_idx
        )
        for d in ds.dims:
            if ds.sizes[d] == 1:
                resample_func = _resample_iterations
    else:
        resample_func = getattr(xs, OPTIONS["resample_iterations_func"])
    return resample_func


def _maybe_auto_chunk(ds, dims):
    """Auto-chunk on dimension `dims`.

    Args:
        ds (xr.Dataset): input data.
        dims (list of str or str): Dimensions to auto-chunk in.

    Returns:
        xr.Dataset: auto-chunked along `dims`

    """
    if dask.is_dask_collection(ds) and dims != []:
        if isinstance(dims, str):
            dims = [dims]
        chunks = [d for d in dims if d in ds.dims]
        chunks = {key: "auto" for key in chunks}
        ds = ds.chunk(chunks)
    return ds


def _chunk_before_resample_iterations_idx(
    ds, iterations, chunking_dims, optimal_blocksize=100000000
):
    """Chunk that after _resample_iteration_idx chunks have optimal `optimal_blocksize`.

    Args:
        ds (xr.obejct): input data`.
        iterations (int): number of bootstrap iterations in `_resample_iterations_idx`.
        chunking_dims (list of str or str): Dimension(s) to chunking in.
        optimal_blocksize (int): dask blocksize to aim at in bytes.
            Defaults to 100000000.

    Returns:
        xr.Dataset: chunked to have blocksize: optimal_blocksize/iterations.

    """
    if isinstance(chunking_dims, str):
        chunking_dims = [chunking_dims]
    # size of CLIMPRED_DIMS
    climpred_dim_chunksize = 8 * np.prod(
        np.array([ds[d].size for d in CLIMPRED_DIMS if d in ds.dims])
    )
    # remaining blocksize for remaining dims considering iteration
    spatial_dim_blocksize = optimal_blocksize / (climpred_dim_chunksize * iterations)
    # size of remaining dims
    chunking_dims_size = np.prod(
        np.array([ds[d].size for d in ds.dims if d not in CLIMPRED_DIMS])
    )  # ds.lat.size*ds.lon.size
    # chunks needed to get to optimal blocksize
    chunks_needed = chunking_dims_size / spatial_dim_blocksize
    # get size clon, clat for spatial chunks
    cdim = [1 for i in chunking_dims]
    nchunks = np.prod(cdim)
    stepsize = 1
    counter = 0
    while nchunks < chunks_needed:
        for i, d in enumerate(chunking_dims):
            c = cdim[i]
            if c <= ds[d].size:
                c = c + stepsize
                cdim[i] = c
            nchunks = np.prod(cdim)
        counter += 1
        if counter == 100:
            break
    # convert number of chunks to chunksize
    chunks = dict()
    for i, d in enumerate(chunking_dims):
        chunksize = ds[d].size // cdim[i]
        if chunksize < 1:
            chunksize = 1
        chunks[d] = chunksize
    ds = ds.chunk(chunks)
    return ds


def resample_skill_loop(self, iterations, resample_dim, verify_kwargs):
    # slow: loop and verify each time
    # used for HindcastEnsemble.bootstrap(metric='acc') and if
    # other resample_skill funcs dont work
    logging.info("use resample_skill_loop")

    resampled_skills = []
    if not self.get_initialized():
        # warn not found and therefore generate
        warnings.warn(
            "uninitialized not found and are therefore generated by generate_uninitialized()"
        )
    self_for_loop = self.copy()
    loop = range(iterations)
    if tqdm:
        loop = tqdm(loop)
    for i in loop:
        # resample initialized
        self_for_loop._datasets["initialized"] = _resample(
            self.get_initialized(), resample_dim
        )
        if "uninitialized" in verify_kwargs["reference"]:
            # resample uninitialized
            if not self.get_uninitialized():
                # warn not found and therefore generate
                self_for_loop._datasets["uninitialized"] = (
                    self.generate_uninitialized().get_uninitialized()
                )
            else:
                self_for_loop._datasets["uninitialized"] = _resample(
                    self.get_uninitialized(), "member"
                )
        resampled_skills.append(self_for_loop.verify(**verify_kwargs))
    resampled_skills = xr.concat(resampled_skills, "iteration")
    return resampled_skills


def resample_skill_exclude_resample_dim_from_dim(
    self, iterations, resample_dim, verify_kwargs
):
    # fast way by verify(dim=dim_no_resample_dim) and then resampling init
    # used for HindcastEnsemble.bootstrap(resample_dim='init')
    logging.info("use resample_skill_exclude_resample_dim_from_dim")
    if OPTIONS["resample_iterations_func"] == "default":
        if "groupby" in verify_kwargs:
            resample_func = _resample_iterations
        else:
            resample_func = _get_resample_func(self.get_initialized())
    else:
        resample_func = getattr(xs, OPTIONS["resample_iterations_func"])

    verify_kwargs_no_dim = verify_kwargs.copy()
    del verify_kwargs_no_dim["dim"]
    dim = verify_kwargs.get("dim", [])
    if resample_dim in dim:
        remaining_dim = copy(dim)
        remaining_dim.remove(resample_dim)
        post_dim = resample_dim
    else:
        remaining_dim = []  # dim
        post_dim = dim
    verify_skill = self.verify(dim=remaining_dim, **verify_kwargs_no_dim)
    resampled_skills = resample_func(verify_skill, iterations, dim=resample_dim).mean(
        post_dim
    )
    resampled_skills.lead.attrs = self.get_initialized().lead.attrs
    return resampled_skills


def resample_skill_resample_before(self, iterations, resample_dim, verify_kwargs):
    # fast way by resampling member and do vectorized verify
    # used for PerfectModelEnsemble.bootstrap()
    # used for HindcastEnsemble.bootstrap(resample_dim='member')
    logging.info("use resample_skill_resample_before")
    if OPTIONS["resample_iterations_func"] == "default":
        if "groupby" in verify_kwargs:
            resample_func = _resample_iterations
        else:
            resample_func = _get_resample_func(self.get_initialized())
    else:
        resample_func = getattr(xs, OPTIONS["resample_iterations_func"])
    logging.info("using resample_func:", resample_func.__name__)

    chunking_dims = [d for d in self.get_initialized().dims if d not in CLIMPRED_DIMS]
    copy_self = self.copy()
    copy_self._datasets["initialized"] = resample_func(
        self.get_initialized(), iterations, resample_dim
    )
    copy_self._datasets["initialized"] = _maybe_auto_chunk(
        copy_self._datasets["initialized"], ["iteration"] + chunking_dims
    )

    copy_self._datasets["initialized"].lead.attrs = self.get_initialized().lead.attrs

    if "uninitialized" in verify_kwargs["reference"]:
        if not self.get_uninitialized():
            # warn not found and therefore generate
            warnings.warn(
                "uninitialized not found therefore generated by"
                " generate_uninitialized()"
            )
            members_to_sample_from = 50
            repeat = (
                members_to_sample_from // copy_self.get_initialized().member.size + 1
            )
            uninit_initialized = xr.concat(
                [
                    self.generate_uninitialized().get_uninitialized()
                    for i in range(repeat)
                ],
                dim="member",
                **CONCAT_KWARGS,
            )
            uninit_initialized["member"] = np.arange(
                1, 1 + uninit_initialized.member.size
            )
            if dask.is_dask_collection(uninit_initialized):
                # too minimize tasks: ensure uninit_initialized get pre-computed
                # alternativly .chunk({'member':-1})
                uninit_initialized = uninit_initialized.compute().chunk()
            # resample uninit always over member and select only initialized.member.size
            bootstrapped_uninit = resample_func(
                uninit_initialized,
                iterations,
                "member",
                replace=False,
                dim_max=self.get_initialized()["member"].size,
            )
            # bootstrapped_uninit["lead"] = self.get_initialized()["lead"]
            # effectively only when _resample_iteration_idx which doesnt use dim_max
            bootstrapped_uninit = bootstrapped_uninit.isel(
                member=slice(None, self.get_initialized().member.size)
            )
            bootstrapped_uninit["member"] = np.arange(
                1, 1 + bootstrapped_uninit.member.size
            )
            copy_self._datasets["uninitialized"] = bootstrapped_uninit
        else:
            copy_self._datasets["uninitialized"] = resample_func(
                copy_self._datasets["uninitialized"], iterations, "member"
            )
        if dask.is_dask_collection(copy_self._datasets["uninitialized"]):
            copy_self._datasets["uninitialized"] = copy_self._datasets[
                "uninitialized"
            ].chunk({"member": -1})
            copy_self._datasets["uninitialized"] = _maybe_auto_chunk(
                copy_self._datasets["uninitialized"], ["iteration"] + chunking_dims
            )
    resampled_skills = copy_self.verify(**verify_kwargs)
    return resampled_skills


def _bootstrap_func(
    func,
    ds,
    resample_dim,
    sig=95,
    iterations=500,
    *func_args,
    **func_kwargs,
):
    """Calc sig % threshold of function based on iterations resampling with replacement.

    Reference:
        * Mason, S. J., and G. M. Mimmack. “The Use of Bootstrap Confidence
          Intervals for the Correlation Coefficient in Climatology.” Theoretical and
          Applied Climatology 45, no. 4 (December 1, 1992): 229–33.
          https://doi.org/10/b6fnsv.

    Args:
        func (function): function to be bootstrapped.
        ds (xr.Dataset): first input argument of func. `chunk` ds on `dim` other
            than `resample_dim` for potential performance increase when multiple
            CPUs available.
        resample_dim (str): dimension to resample from.
        sig (int,float,list): significance levels to return. Defaults to 95.
        iterations (int): number of resample iterations. Defaults to 500.
        *func_args (type): `*func_args`.
        **func_kwargs (type): `**func_kwargs`.

    Returns:
        sig_level: bootstrapped significance levels with
            dimensions of init_pm and len(sig) if sig is list
    """
    if not callable(func):
        raise ValueError(f"Please provide func as a function, found {type(func)}")
    warn_if_chunking_would_increase_performance(ds)
    if isinstance(sig, list):
        psig = [i / 100 for i in sig]
    else:
        psig = sig / 100

    resample_func = _get_resample_func(ds)
    bootstraped_ds = resample_func(ds, iterations, dim=resample_dim, replace=False)
    bootstraped_results = func(bootstraped_ds, *func_args, **func_kwargs)
    bootstraped_results = rechunk_to_single_chunk_if_more_than_one_chunk_along_dim(
        bootstraped_results, dim="iteration"
    )
    sig_level = bootstraped_results.quantile(dim="iteration", q=psig, skipna=False)
    return sig_level


def dpp_threshold(control, sig=95, iterations=500, dim="time", **dpp_kwargs):
    """Calc DPP significance levels from re-sampled dataset.

    Reference:
        :cite:t:`Feng2011`

    See also:
        * :py:func:`.climpred.bootstrap._bootstrap_func`
        * :py:func:`.climpred.stats.dpp`
    """
    return _bootstrap_func(
        dpp, control, dim, sig=sig, iterations=iterations, **dpp_kwargs
    )


def varweighted_mean_period_threshold(control, sig=95, iterations=500, time_dim="time"):
    """Calc variance-weighted mean period significance levels from resampled dataset.

    See also:
        * :py:func:`.climpred.bootstrap._bootstrap_func`
        * :py:func:`.climpred.stats.varweighted_mean_period`
    """
    if varweighted_mean_period is None:
        raise ImportError(
            "xrft is not installed; see "
            "https://xrft.readthedocs.io/en/latest/installation.html"
        )

    return _bootstrap_func(
        varweighted_mean_period,
        control,
        time_dim,
        sig=sig,
        iterations=iterations,
    )
