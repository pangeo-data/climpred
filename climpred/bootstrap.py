"""Bootstrap or resampling operators for functional compute_ functions."""

import warnings

import dask
import numpy as np
import xarray as xr
from xskillscore.core.resampling import (
    resample_iterations as _resample_iterations,
    resample_iterations_idx as _resample_iterations_idx,
)

from climpred.constants import CLIMPRED_DIMS, CONCAT_KWARGS, PM_CALENDAR_STR

from .checks import (
    has_dims,
    has_valid_lead_units,
    warn_if_chunking_would_increase_performance,
)
from .comparisons import (
    ALL_COMPARISONS,
    COMPARISON_ALIASES,
    HINDCAST_COMPARISONS,
    __m2o,
)
from .exceptions import KeywordError
from .metrics import ALL_METRICS, METRIC_ALIASES
from .options import OPTIONS
from .prediction import compute_hindcast, compute_perfect_model
from .reference import (
    compute_climatology,
    compute_persistence,
    compute_persistence_from_first_lead,
)
from .stats import dpp

try:
    from .stats import varweighted_mean_period
except ImportError:
    varweighted_mean_period = None  # type: ignore
from .utils import (
    _transpose_and_rechunk_to,
    convert_time_index,
    find_start_dates_for_given_init,
    get_comparison_class,
    get_lead_cftime_shift_args,
    get_metric_class,
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
    """
    Generate ``uninitialized`` by resamplling from ``initialized``.

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
    """
    Bootstrap member, lead, init from control by reshaping.

    Fast track of function
    `bootstrap_uninit_pm_ensemble_from_control_cftime` when lead units is 'years'.
    """
    assert type(init_pm) == type(control)
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


def _bootstrap_hindcast_over_init_dim(
    initialized,
    hist,
    verif,
    dim,
    reference,
    resample_dim,
    iterations,
    metric,
    comparison,
    compute,
    resample_uninit,
    **metric_kwargs,
):
    """Bootstrap hindcast skill over the ``init`` dimension.

    When bootstrapping over the ``member`` dimension, an additional dimension
    ``iteration`` can be added and skill can be computing over that entire
    dimension in parallel, since all members are being aligned the same way.
    However, to our knowledge, when bootstrapping over the ``init`` dimension,
    one must evaluate each iteration independently. I.e., in a looped fashion,
    since alignment of initializations and target dates is unique to each
    iteration.

    See ``bootstrap_compute`` for explanation of inputs.
    """
    pers_skill = []
    bootstrapped_init_skill = []
    bootstrapped_uninit_skill = []
    for i in range(iterations):
        # resample with replacement
        smp_initialized = _resample(initialized, resample_dim)
        # compute init skill
        init_skill = compute(
            smp_initialized,
            verif,
            metric=metric,
            comparison=comparison,
            dim=dim,
            **metric_kwargs,
        )
        # reset inits when probabilistic, otherwise tests fail
        if (
            resample_dim == "init"
            and metric.probabilistic
            and "init" in init_skill.coords
        ):
            init_skill["init"] = initialized.init.values
        bootstrapped_init_skill.append(init_skill)
        if "uninitialized" in reference:
            # generate uninitialized ensemble from hist
            uninit_initialized = resample_uninit(initialized, hist)
            # compute uninit skill
            bootstrapped_uninit_skill.append(
                compute(
                    uninit_initialized,
                    verif,
                    metric=metric,
                    comparison=comparison,
                    dim=dim,
                    **metric_kwargs,
                )
            )
        if "persistence" in reference:
            pers_skill.append(
                compute_persistence(
                    smp_initialized,
                    verif,
                    metric=metric,
                    dim=dim,
                    **metric_kwargs,
                )
            )
    bootstrapped_init_skill = xr.concat(
        bootstrapped_init_skill, dim="iteration", **CONCAT_KWARGS
    )
    if "uninitialized" in reference:
        bootstrapped_uninit_skill = xr.concat(
            bootstrapped_uninit_skill, dim="iteration", **CONCAT_KWARGS
        )
    else:
        bootstrapped_uninit_skill = None
    if "persistence" in reference:
        bootstrapped_pers_skill = xr.concat(
            pers_skill, dim="iteration", **CONCAT_KWARGS
        )
    else:
        bootstrapped_pers_skill = None
    return (bootstrapped_init_skill, bootstrapped_uninit_skill, bootstrapped_pers_skill)


def _get_resample_func(ds):
    """
    Decide for resample function based on input `ds`.

    Returns:
      callable: `_resample_iterations`: if big and chunked `ds`
                `_resample_iterations_idx`: else (if small and eager `ds`)
    """
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
    return resample_func


def _maybe_auto_chunk(ds, dims):
    """Auto-chunk on dimension `dims`.

    Args:
        ds (xr.Dataset): input data.
        dims (list of str or str): Dimensions to auto-chunk in.

    Returns:
        xr.Dataset: auto-chunked along `dims`

    """
    if dask.is_dask_collection(ds) and dims is not []:
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
    climpred_dim_chunksize = 8 * np.product(
        np.array([ds[d].size for d in CLIMPRED_DIMS if d in ds.dims])
    )
    # remaining blocksize for remaining dims considering iteration
    spatial_dim_blocksize = optimal_blocksize / (climpred_dim_chunksize * iterations)
    # size of remaining dims
    chunking_dims_size = np.product(
        np.array([ds[d].size for d in ds.dims if d not in CLIMPRED_DIMS])
    )  # ds.lat.size*ds.lon.size
    # chunks needed to get to optimal blocksize
    chunks_needed = chunking_dims_size / spatial_dim_blocksize
    # get size clon, clat for spatial chunks
    cdim = [1 for i in chunking_dims]
    nchunks = np.product(cdim)
    stepsize = 1
    counter = 0
    while nchunks < chunks_needed:
        for i, d in enumerate(chunking_dims):
            c = cdim[i]
            if c <= ds[d].size:
                c = c + stepsize
                cdim[i] = c
            nchunks = np.product(cdim)
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


def bootstrap_compute(
    initialized,
    verif,
    hist=None,
    alignment="same_verifs",
    metric="pearson_r",
    comparison="m2e",
    dim="init",
    reference=None,
    resample_dim="member",
    sig=95,
    iterations=500,
    pers_sig=None,
    compute=compute_hindcast,
    resample_uninit=bootstrap_uninitialized_ensemble,
    **metric_kwargs,
):
    """Bootstrap compute with replacement.

    Args:
        initialized (xr.Dataset): prediction ensemble.
        verif (xr.Dataset): Verification data.
        hist (xr.Dataset): historical/uninitialized simulation.
        metric (str): `metric`. Defaults to ``"pearson_r"``.
        comparison (str): `comparison`. Defaults to ``"m2e"``.
        dim (str or list): dimension(s) to apply metric over. Defaults to: "init".
        reference (str, list of str): Type of reference forecasts with which to
            verify. One or more of ["persistence", "uninitialized"].
            If None or empty, returns no p value.
        resample_dim (str): dimension to resample from. Defaults to: "member"

            - "member": select a different set of members from initialized
            - "init": select a different set of initializations from initialized

        sig (int): Significance level for uninitialized and
                   initialized skill. Defaults to ``95``.
        pers_sig (int): Significance level for persistence skill confidence levels.
            Defaults to ``sig``.
        iterations (int): number of resampling iterations (bootstrap with replacement).
            Defaults to ``500``.
        compute (Callable): function to compute skill. Choose from
            [:py:func:`climpred.prediction.compute_perfect_model`,
            :py:func:`climpred.prediction.compute_hindcast`].
        resample_uninit (Callable): function to create an uninitialized ensemble
            from a control simulation or uninitialized large ensemble. Choose from:
            [:py:func:`bootstrap_uninitialized_ensemble`,
             :py:func:`bootstrap_uninit_pm_ensemble_from_control`].
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        results: (xr.Dataset): bootstrapped results for the three different skills:

            - ``initialized`` for the initialized hindcast ``initialized`` and
             describes skill due to initialization and external forcing
            - ``uninitialized`` for the uninitialized/historical and approximates skill
             from external forcing
            - ``persistence``
            - ``climatology``

        the different results:
            - ``verify skill``: skill values
            - ``p``: p value
            - ``low_ci`` and ``high_ci``: high and low ends of confidence intervals
             based on significance threshold ``sig``


    Reference:
        :cite:t:`Goddard2013`

    See also:
        * :py:func:`.climpred.bootstrap.bootstrap_hindcast`
        * :py:func:`.climpred.bootstrap.bootstrap_perfect_model`
    """
    warn_if_chunking_would_increase_performance(initialized, crit_size_in_MB=5)
    if pers_sig is None:
        pers_sig = sig
    if isinstance(dim, str):
        dim = [dim]
    if isinstance(reference, str):
        reference = [reference]
    if reference is None:
        reference = []

    compute_persistence_func = compute_persistence_from_first_lead
    if (
        OPTIONS["PerfectModel_persistence_from_initialized_lead_0"]
        and compute.__name__ == "compute_perfect_model"
    ):
        compute_persistence_func = compute_persistence_from_first_lead
        if initialized.lead[0] != 0:
            if OPTIONS["warn_for_failed_PredictionEnsemble_xr_call"]:
                warnings.warn(
                    f"Calculate persistence from lead={int(initialized.lead[0].values)} "
                    "instead of lead=0 (recommended)."
                )
    else:
        compute_persistence_func = compute_persistence

    p, ci_low, ci_high = _p_ci_from_sig(sig)
    p_pers, ci_low_pers, ci_high_pers = _p_ci_from_sig(pers_sig)

    # get metric/comparison function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)
    comparison = COMPARISON_ALIASES.get(comparison, comparison)

    # get class Metric(metric)
    metric = get_metric_class(metric, ALL_METRICS)
    # get comparison function
    comparison = get_comparison_class(comparison, ALL_COMPARISONS)

    # Perfect Model requires `same_inits` setup
    isHindcast = True if comparison.name in HINDCAST_COMPARISONS else False
    reference_alignment = alignment if isHindcast else "same_inits"
    chunking_dims = [d for d in initialized.dims if d not in CLIMPRED_DIMS]

    # carry alignment for compute_reference separately
    metric_kwargs_reference = metric_kwargs.copy()
    metric_kwargs_reference["alignment"] = reference_alignment
    # carry alignment in metric_kwargs
    if isHindcast:
        metric_kwargs["alignment"] = alignment

    if hist is None:  # PM path, use verif = control
        hist = verif

    # slower path for hindcast and resample_dim init
    if resample_dim == "init" and isHindcast:
        warnings.warn("resample_dim=`init` will be slower than resample_dim=`member`.")
        (
            bootstrapped_init_skill,
            bootstrapped_uninit_skill,
            bootstrapped_pers_skill,
        ) = _bootstrap_hindcast_over_init_dim(
            initialized,
            hist,
            verif,
            dim,
            reference,
            resample_dim,
            iterations,
            metric,
            comparison,
            compute,
            resample_uninit,
            **metric_kwargs,
        )
    else:  # faster: first _resample_iterations_idx, then compute skill
        resample_func = _get_resample_func(initialized)
        if not isHindcast:
            if "uninitialized" in reference:
                # create more members than needed in PM to make the uninitialized
                # distribution more robust
                members_to_sample_from = 50
                repeat = members_to_sample_from // initialized.member.size + 1
                uninit_initialized = xr.concat(
                    [resample_uninit(initialized, hist) for i in range(repeat)],
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
                    dim_max=initialized["member"].size,
                )
                bootstrapped_uninit["lead"] = initialized["lead"]
                # effectively only when _resample_iteration_idx which doesnt use dim_max
                bootstrapped_uninit = bootstrapped_uninit.isel(
                    member=slice(None, initialized.member.size)
                )
                bootstrapped_uninit["member"] = np.arange(
                    1, 1 + bootstrapped_uninit.member.size
                )
                if dask.is_dask_collection(bootstrapped_uninit):
                    bootstrapped_uninit = bootstrapped_uninit.chunk({"member": -1})
                    bootstrapped_uninit = _maybe_auto_chunk(
                        bootstrapped_uninit, ["iteration"] + chunking_dims
                    )
        else:  # hindcast
            if "uninitialized" in reference:
                uninit_initialized = resample_uninit(initialized, hist)
                if dask.is_dask_collection(uninit_initialized):
                    # too minimize tasks: ensure uninit_initialized get pre-computed
                    # maybe not needed
                    uninit_initialized = uninit_initialized.compute().chunk()
                bootstrapped_uninit = resample_func(
                    uninit_initialized, iterations, resample_dim
                )
                bootstrapped_uninit = bootstrapped_uninit.isel(
                    member=slice(None, initialized.member.size)
                )
                bootstrapped_uninit["lead"] = initialized["lead"]
                if dask.is_dask_collection(bootstrapped_uninit):
                    bootstrapped_uninit = _maybe_auto_chunk(
                        bootstrapped_uninit.chunk({"lead": 1}),
                        ["iteration"] + chunking_dims,
                    )

        if "uninitialized" in reference:
            bootstrapped_uninit_skill = compute(
                bootstrapped_uninit,
                verif,
                metric=metric,
                comparison="m2o" if isHindcast else comparison,
                dim=dim,
                **metric_kwargs,
            )
            # take mean if 'm2o' comparison forced before
            if isHindcast and comparison != __m2o:
                bootstrapped_uninit_skill = bootstrapped_uninit_skill.mean("member")

        with xr.set_options(keep_attrs=True):
            bootstrapped_initialized = resample_func(
                initialized, iterations, resample_dim
            )
        if dask.is_dask_collection(bootstrapped_initialized):
            bootstrapped_initialized = bootstrapped_initialized.chunk({"member": -1})

        bootstrapped_init_skill = compute(
            bootstrapped_initialized,
            verif,
            metric=metric,
            comparison=comparison,
            dim=dim,
            **metric_kwargs,
        )
        if "persistence" in reference:
            pers_skill = compute_persistence_func(
                initialized,
                verif,
                metric=metric,
                dim=dim,
                **metric_kwargs_reference,
            )
            # bootstrap pers
            if resample_dim == "init":
                bootstrapped_pers_skill = compute_persistence_func(
                    bootstrapped_initialized,
                    verif,
                    metric=metric,
                    **metric_kwargs_reference,
                )
            else:  # member no need to calculate all again
                bootstrapped_pers_skill, _ = xr.broadcast(
                    pers_skill, bootstrapped_init_skill
                )

    # calc mean skill without any resampling
    init_skill = compute(
        initialized,
        verif,
        metric=metric,
        comparison=comparison,
        dim=dim,
        **metric_kwargs,
    )

    if "uninitialized" in reference:
        # uninit skill as mean resampled uninit skill
        unin_skill = bootstrapped_uninit_skill.mean("iteration")  # noqa: F841
    if "persistence" in reference:
        pers_skill = compute_persistence_func(
            initialized, verif, metric=metric, dim=dim, **metric_kwargs_reference
        )
    if "climatology" in reference:
        clim_skill = compute_climatology(
            initialized,
            verif,
            metric=metric,
            dim=dim,
            comparison=comparison,
            **metric_kwargs,
        )
        # get clim_skill into init,lead dimensions
        if "time" in clim_skill.dims and "valid_time" in init_skill.coords:
            # for idea see https://github.com/pydata/xarray/discussions/4593
            valid_time_overlap = init_skill.coords["valid_time"].where(
                init_skill.coords["valid_time"].isin(clim_skill.time)
            )
            clim_skill = clim_skill.rename({"time": "valid_time"})
            clim_skill = clim_skill.sel(
                valid_time=init_skill.coords["valid_time"], method="nearest"
            )
            # mask wrongly taken method nearest values
            clim_skill = clim_skill.where(valid_time_overlap.notnull())
            # print('after special sel', clim_skill.coords, clim_skill.sizes)
        bootstrapped_clim_skill, _ = xr.broadcast(clim_skill, bootstrapped_init_skill)

    # get confidence intervals CI
    init_ci = _distribution_to_ci(bootstrapped_init_skill, ci_low, ci_high)
    if "uninitialized" in reference:
        unin_ci = _distribution_to_ci(  # noqa: F841
            bootstrapped_uninit_skill, ci_low, ci_high
        )
    if "climatology" in reference:
        clim_ci = _distribution_to_ci(  # noqa: F841
            bootstrapped_clim_skill, ci_low, ci_high
        )
    if "persistence" in reference:
        pers_ci = _distribution_to_ci(  # noqa: F841
            bootstrapped_pers_skill, ci_low_pers, ci_high_pers
        )

    # pvalue whether uninit or pers better than init forecast
    if "uninitialized" in reference:
        p_unin_over_init = _pvalue_from_distributions(  # noqa: F841
            bootstrapped_uninit_skill, bootstrapped_init_skill, metric=metric
        )
    if "climatology" in reference:
        p_clim_over_init = _pvalue_from_distributions(  # noqa: F841
            bootstrapped_clim_skill, bootstrapped_init_skill, metric=metric
        )
    if "persistence" in reference:
        p_pers_over_init = _pvalue_from_distributions(  # noqa: F841
            bootstrapped_pers_skill, bootstrapped_init_skill, metric=metric
        )

    # gather return
    # p defined as probability that reference better than
    # initialized, therefore not defined for initialized skill
    # itself
    results = xr.concat(
        [
            init_skill,
            init_skill.where(init_skill == -999),
            init_ci.isel(quantile=0, drop=True),
            init_ci.isel(quantile=1, drop=True),
        ],
        dim="results",
        coords="minimal",
    ).assign_coords(
        results=("results", ["verify skill", "p", "low_ci", "high_ci"]),
        skill="initialized",
    )

    if reference != []:
        for r in reference:
            ref_skill = eval(f"{r[:4]}_skill")
            ref_p = eval(f"p_{r[:4]}_over_init")
            ref_ci_low = eval(f"{r[:4]}_ci").isel(quantile=0, drop=True)
            ref_ci_high = eval(f"{r[:4]}_ci").isel(quantile=1, drop=True)
            ref_results = xr.concat(
                [ref_skill, ref_p, ref_ci_low, ref_ci_high],
                dim="results",
                **CONCAT_KWARGS,
            ).assign_coords(
                skill=r, results=("results", ["verify skill", "p", "low_ci", "high_ci"])
            )
            if "member" in ref_results.dims:
                if not ref_results["member"].identical(results["member"]):
                    ref_results["member"] = results[
                        "member"
                    ]  # fixes m2c different member names in reference forecasts
            results = xr.concat([results, ref_results], dim="skill", **CONCAT_KWARGS)
        results = results.assign_coords(skill=["initialized"] + reference).squeeze()
    else:
        results = results.drop_sel(results="p")
    results = results.squeeze()

    # Ensure that the lead units get carried along for the calculation. The attribute
    # tends to get dropped along the way due to ``xarray`` functionality.
    results["lead"] = initialized["lead"]
    if "units" in initialized["lead"].attrs and "units" not in results["lead"].attrs:
        results["lead"].attrs["units"] = initialized["lead"].attrs["units"]
    return results


def bootstrap_hindcast(
    initialized,
    hist,
    verif,
    alignment="same_verifs",
    metric="pearson_r",
    comparison="e2o",
    dim="init",
    reference=None,
    resample_dim="member",
    sig=95,
    iterations=500,
    pers_sig=None,
    **metric_kwargs,
):
    """Wrap py:func:`bootstrap_compute` for hindcasts.

    Args:
        initialized (xr.Dataset): prediction ensemble.
        verif (xr.Dataset): Verification data.
        hist (xr.Dataset): historical/uninitialized simulation.
        metric (str): `metric`. Defaults to ``"pearson_r"``.
        comparison (str): `comparison`. Defaults to "e2o".
        dim (str): dimension to apply metric over. Defaults to: "init".
        reference (str, list of str): Type of reference forecasts with which to
            verify. One or more of ["persistence", "uninitialized"].
            If None or empty, returns no p value.
        resample_dim (str or list): dimension to resample from.
            Defaults to: ``"member"``.

            - "member": select a different set of members from initialized
            - "init": select a different set of initializations from initialized

        sig (int): Significance level for uninitialized and initialized skill.
            Defaults to ``95``.
        pers_sig (int): Significance level for persistence skill confidence levels.
            Defaults to ``sig``.
        iterations (int): number of resampling iterations (bootstrap with replacement).
            Defaults to 500.
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        results: (xr.Dataset): bootstrapped results for the three different kinds of
            predictions:

            - ``initialized`` for the initialized hindcast ``initialized`` and
             describes skill due to initialization and external forcing
            - ``uninitialized`` for the uninitialized/historical and approximates skill
             from external forcing
            - ``persistence``
            - ``climatology``

        the different results:
            - ``verify skill``: skill values
            - ``p``: p value
            - ``low_ci`` and ``high_ci``: high and low ends of confidence intervals
             based on significance threshold ``sig``

    Reference:
        :cite:t:`Goddard2013`

    See also:
        * :py:func:`.climpred.bootstrap.bootstrap_compute`
        * :py:func:`.climpred.prediction.compute_hindcast`

    """
    # Check that init is int, cftime, or datetime; convert ints or datetime to cftime.
    initialized = convert_time_index(initialized, "init", "initialized[init]")
    if isinstance(hist, xr.Dataset):
        hist = convert_time_index(hist, "time", "uninitialized[time]")
    else:
        hist = False
    verif = convert_time_index(verif, "time", "verif[time]")
    # Put this after `convert_time_index` since it assigns 'years' attribute if the
    # `init` dimension is a `float` or `int`.
    has_valid_lead_units(initialized)

    if ("same_verif" in alignment) & (resample_dim == "init"):
        raise KeywordError(
            "Cannot have both alignment='same_verifs' and "
            "resample_dim='init'. Change `resample_dim` to 'member' to keep "
            "common verification alignment or `alignment` to 'same_inits' to "
            "resample over initializations."
        )

    # Kludge for now. Since we're computing persistence here we need to ensure that
    # all products have a union in their time axis.
    if hist not in [None, False]:
        times = np.sort(
            list(
                set(initialized.init.data) & set(hist.time.data) & set(verif.time.data)
            )
        )
    else:
        times = np.sort(list(set(initialized.init.data) & set(verif.time.data)))
    initialized = initialized.sel(init=times)
    if isinstance(hist, xr.Dataset):
        hist = hist.sel(time=times)
    verif = verif.sel(time=times)

    return bootstrap_compute(
        initialized,
        verif,
        hist=hist,
        alignment=alignment,
        metric=metric,
        comparison=comparison,
        dim=dim,
        reference=reference,
        resample_dim=resample_dim,
        sig=sig,
        iterations=iterations,
        pers_sig=pers_sig,
        compute=compute_hindcast,
        resample_uninit=bootstrap_uninitialized_ensemble,
        **metric_kwargs,
    )


def bootstrap_perfect_model(
    init_pm,
    control,
    metric="pearson_r",
    comparison="m2e",
    dim=None,
    reference=None,
    resample_dim="member",
    sig=95,
    iterations=500,
    pers_sig=None,
    **metric_kwargs,
):
    """Wrap py:func:`bootstrap_compute` for perfect-model framework.

    Args:
        initialized (xr.Dataset): prediction ensemble.
        verif (xr.Dataset): Verification data.
        hist (xr.Dataset): historical/uninitialized simulation.
        metric (str): `metric`. Defaults to ``"pearson_r"``.
        comparison (str): `comparison`. Defaults to ``"m2e"``.
        dim (str): dimension to apply metric over. Defaults to: ``["init", "member"]``.
        reference (str, list of str): Type of reference forecasts with which to
            verify. One or more of ``["persistence", "uninitialized", "climatology"]``.
            If ``None`` or ``[]``, returns no p value.
        resample_dim (str or list): dimension to resample from.
            Defaults to: ``"member"``.

            - "member": select a different set of members from initialized
            - "init": select a different set of initializations from initialized

        sig (int): Significance level for uninitialized and initialized skill.
            Defaults to ``95``.
        pers_sig (int): Significance level for persistence skill confidence levels.
            Defaults to ``sig``.
        iterations (int): number of resampling iterations (bootstrap with replacement).
            Defaults to ``500``.
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        results: (xr.Dataset): bootstrapped results for the three different kinds of
            predictions:

            - ``initialized`` for the initialized hindcast ``initialized`` and
             describes skill due to initialization and external forcing
            - ``uninitialized`` for the uninitialized/historical and approximates skill
             from external forcing
            - ``persistence`` for the persistence forecast computed by
             `compute_persistence` or `compute_persistence_from_first_lead` depending
             on set_options("PerfectModel_persistence_from_initialized_lead_0")
            - ``climatology``

        the different results:
            - ``skill``: skill values
            - ``p``: p value
            - ``low_ci`` and ``high_ci``: high and low ends of confidence intervals
             based on significance threshold ``sig``

    Reference:
        :cite:t:`Goddard2013`

    See also:
        * :py:func:`.climpred.bootstrap.bootstrap_compute`
        * :py:func:`.climpred.prediction.compute_perfect_model`
    """
    if dim is None:
        dim = ["init", "member"]
    # Check init & time is int, cftime, or datetime; convert ints or datetime to cftime.
    init_pm = convert_time_index(
        init_pm, "init", "init_pm[init]", calendar=PM_CALENDAR_STR
    )
    control = convert_time_index(
        control, "time", "control[time]", calendar=PM_CALENDAR_STR
    )
    lead_units_equal_control_time_stride(init_pm, control)
    return bootstrap_compute(
        init_pm,
        control,
        hist=None,
        metric=metric,
        comparison=comparison,
        dim=dim,
        reference=reference,
        resample_dim=resample_dim,
        sig=sig,
        iterations=iterations,
        pers_sig=pers_sig,
        compute=compute_perfect_model,
        resample_uninit=bootstrap_uninit_pm_ensemble_from_control_cftime,
        **metric_kwargs,
    )


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
