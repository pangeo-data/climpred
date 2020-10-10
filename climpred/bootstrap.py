import inspect
import warnings

import dask
import numpy as np
import xarray as xr

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
from .prediction import compute_hindcast, compute_perfect_model
from .reference import compute_persistence
from .stats import dpp, varweighted_mean_period
from .utils import (
    _transpose_and_rechunk_to,
    assign_attrs,
    convert_time_index,
    find_start_dates_for_given_init,
    get_comparison_class,
    get_lead_cftime_shift_args,
    get_metric_class,
    lead_units_equal_control_time_stride,
    rechunk_to_single_chunk_if_more_than_one_chunk_along_dim,
    shift_cftime_singular,
)


def _resample(hind, resample_dim):
    """Resample with replacement in dimension ``resample_dim``.

    Args:
        hind (xr.object): input xr.object to be resampled.
        resample_dim (str): dimension to resample along.

    Returns:
        xr.object: resampled along ``resample_dim``.

    """
    to_be_resampled = hind[resample_dim].values
    smp = np.random.choice(to_be_resampled, len(to_be_resampled))
    smp_hind = hind.sel({resample_dim: smp})
    # ignore because then inits should keep their labels
    if resample_dim != "init":
        smp_hind[resample_dim] = hind[resample_dim].values
    return smp_hind


def _resample_iterations(init, iterations, dim="member", dim_max=None, replace=True):
    """Resample over ``dim`` by index ``iterations`` times.

    .. note::
        This gives the same result as `_resample_iterations_idx`. When using dask, the
        number of tasks in `_resample_iterations` will scale with iterations but
        constant chunksize, whereas the tasks in `_resample_iterations_idx` will stay
        constant with increasing chunksize.

    Args:
        init (xr.DataArray, xr.Dataset): Initialized prediction ensemble.
        iterations (int): Number of bootstrapping iterations.
        dim (str): Dimension name to bootstrap over. Defaults to ``'member'``.
        dim_max (int): Number of items to select in `dim`.
        replace (bool): Bootstrapping with or without replacement. Defaults to ``True``.

    Returns:
        xr.DataArray, xr.Dataset: Bootstrapped data with additional dim ```iteration```

    """
    if dim_max is not None and dim_max <= init[dim].size:
        # select only dim_max items
        select_dim_items = dim_max
        new_dim = init[dim].isel({dim: slice(None, dim_max)})
    else:
        select_dim_items = init[dim].size
        new_dim = init[dim]

    if replace:
        idx = np.random.randint(0, init[dim].size, (iterations, select_dim_items))
    elif not replace:
        # create 2d np.arange()
        idx = np.linspace(
            (np.arange(select_dim_items)),
            (np.arange(select_dim_items)),
            iterations,
            dtype="int",
        )
        # shuffle each line
        for ndx in np.arange(iterations):
            np.random.shuffle(idx[ndx])
    idx_da = xr.DataArray(
        idx,
        dims=("iteration", dim),
        coords=({"iteration": range(iterations), dim: new_dim}),
    )
    init_smp = []
    for i in np.arange(iterations):
        idx = idx_da.sel(iteration=i).data
        init_smp2 = init.isel({dim: idx}).assign_coords({dim: new_dim})
        init_smp.append(init_smp2)
    init_smp = xr.concat(init_smp, dim="iteration", **CONCAT_KWARGS)
    init_smp["iteration"] = np.arange(1, 1 + iterations)
    return init_smp


def _resample_iterations_idx(
    init, iterations, dim="member", replace=True, chunk=True, dim_max=None
):
    """Resample over ``dim`` by index ``iterations`` times.

    .. note::
        This is a much faster way to bootstrap than resampling each iteration
        individually and applying the function to it. However, this will create a
        DataArray with dimension ``iteration`` of size ``iterations``. It is probably
        best to do this out-of-memory with ``dask`` if you are doing a large number
        of iterations or using spatial output (i.e., not time series data).

    Args:
        init (xr.DataArray, xr.Dataset): Initialized prediction ensemble.
        iterations (int): Number of bootstrapping iterations.
        dim (str): Dimension name to bootstrap over. Defaults to ``'member'``.
        replace (bool): Bootstrapping with or without replacement. Defaults to ``True``.
        chunk: (bool): Auto-chunk along chunking_dims to get optimal blocksize
        dim_max (int): Number of indices from `dim` to return. Not implemented.

    Returns:
        xr.DataArray, xr.Dataset: Bootstrapped data with additional dim ```iteration```

    """
    if dask.is_dask_collection(init):
        init = init.chunk({"lead": -1, "member": -1})
        init = init.copy(deep=True)

    def select_bootstrap_indices_ufunc(x, idx):
        """Selects multi-level indices ``idx`` from xarray object ``x`` for all
        iterations."""
        # `apply_ufunc` sometimes adds a singleton dimension on the end, so we squeeze
        # it out here. This leverages multi-level indexing from numpy, so we can
        # select a different set of, e.g., ensemble members for each iteration and
        # construct one large DataArray with ``iterations`` as a dimension.
        return np.moveaxis(x.squeeze()[idx.squeeze().transpose()], 0, -1)

    if dask.is_dask_collection(init):
        if chunk:
            chunking_dims = [d for d in init.dims if d not in CLIMPRED_DIMS]
            init = _chunk_before_resample_iterations_idx(
                init, iterations, chunking_dims
            )

    # resample with or without replacement
    if replace:
        idx = np.random.randint(0, init[dim].size, (iterations, init[dim].size))
    elif not replace:
        # create 2d np.arange()
        idx = np.linspace(
            (np.arange(init[dim].size)),
            (np.arange(init[dim].size)),
            iterations,
            dtype="int",
        )
        # shuffle each line
        for ndx in np.arange(iterations):
            np.random.shuffle(idx[ndx])
    idx_da = xr.DataArray(
        idx,
        dims=("iteration", dim),
        coords=({"iteration": range(iterations), dim: init[dim]}),
    )
    transpose_kwargs = (
        {"transpose_coords": False} if isinstance(init, xr.DataArray) else {}
    )
    return xr.apply_ufunc(
        select_bootstrap_indices_ufunc,
        init.transpose(dim, ..., **transpose_kwargs),
        idx_da,
        dask="parallelized",
        output_dtypes=[float],
    )


def _distribution_to_ci(ds, ci_low, ci_high, dim="iteration"):
    """Get confidence intervals from bootstrapped distribution.

    Needed for bootstrapping confidence intervals and p_values of a metric.

    Args:
        ds (xarray object): distribution.
        ci_low (float): low confidence interval.
        ci_high (float): high confidence interval.
        dim (str): dimension to apply xr.quantile to. Default: 'iteration'

    Returns:
        uninit_hind (xarray object): uninitialize hindcast with hind.coords.
    """
    ds = rechunk_to_single_chunk_if_more_than_one_chunk_along_dim(ds, dim)
    return ds.quantile(q=[ci_low, ci_high], dim=dim, skipna=False)


def _pvalue_from_distributions(simple_fct, init, metric=None):
    """Get probability that skill of a reference forecast (e.g., persistence or
    uninitialized skill) is larger than initialized skill.

    Needed for bootstrapping confidence intervals and p_values of a metric in
    the hindcast framework. Checks whether a simple forecast like persistence
    or uninitialized performs better than initialized forecast. Need to keep in
    mind the orientation of metric (whether larger values are better or worse
    than smaller ones.)

    Args:
        simple_fct (xarray object): persistence or uninitialized skill.
        init (xarray object): hindcast skill.
        metric (Metric): metric class Metric

    Returns:
        pv (xarray object): probability that simple forecast performs better
                            than initialized forecast.
    """
    pv = ((simple_fct - init) > 0).sum("iteration") / init.iteration.size
    if not metric.positive:
        pv = 1 - pv
    return pv


def bootstrap_uninitialized_ensemble(hind, hist):
    """Resample uninitialized hindcast from historical members.

    Note:
        Needed for bootstrapping confidence intervals and p_values of a metric in
        the hindcast framework. Takes hind.lead.size timesteps from historical at
        same forcing and rearranges them into ensemble and member dimensions.

    Args:
        hind (xarray object): hindcast.
        hist (xarray object): historical uninitialized.

    Returns:
        uninit_hind (xarray object): uninitialize hindcast with hind.coords.
    """
    has_dims(hist, "member", "historical ensemble")
    has_dims(hind, "member", "initialized hindcast ensemble")
    # Put this after `convert_time_index` since it assigns 'years' attribute if the
    # `init` dimension is a `float` or `int`.
    has_valid_lead_units(hind)

    # find range for bootstrapping
    first_init = max(hist.time.min(), hind["init"].min())

    n, freq = get_lead_cftime_shift_args(hind.lead.attrs["units"], hind.lead.size)
    hist_last = shift_cftime_singular(hist.time.max(), -1 * n, freq)
    last_init = min(hist_last, hind["init"].max())

    hind = hind.sel(init=slice(first_init, last_init))

    uninit_hind = []
    for init in hind.init.values:
        # take uninitialized members from hist at init forcing
        # (Goddard et al. allows 5 year forcing range here)
        uninit_at_one_init_year = hist.sel(
            time=slice(
                shift_cftime_singular(init, 1, freq),
                shift_cftime_singular(init, n, freq),
            ),
        ).rename({"time": "lead"})
        uninit_at_one_init_year["lead"] = np.arange(
            1, 1 + uninit_at_one_init_year["lead"].size
        )
        uninit_hind.append(uninit_at_one_init_year)
    uninit_hind = xr.concat(uninit_hind, "init")
    uninit_hind["init"] = hind["init"].values
    uninit_hind.lead.attrs["units"] = hind.lead.attrs["units"]
    uninit_hind["member"] = hist["member"].values
    return (
        _transpose_and_rechunk_to(
            uninit_hind, hind.isel(member=[0] * uninit_hind.member.size)
        )
        if dask.is_dask_collection(uninit_hind)
        else uninit_hind
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
        init_pm (xarray object): initialized ensemble simulation.
        control (xarray object): control simulation.

    Returns:
        uninit_pm (xarray object): uninitialized ensemble generated from control run.
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
        """Select time segments from control from ``suitable_start_dates`` based on
        year ``start_year_int``."""
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


def _bootstrap_by_stacking(init_pm, control):
    """Bootstrap member, lead, init from control by reshaping. Fast track of function
    `bootstrap_uninit_pm_ensemble_from_control_cftime` when lead units is 'years'."""
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
    hind,
    hist,
    verif,
    dim,
    reference,
    resample_dim,
    iterations,
    metric,
    comparison,
    compute,
    reference_compute,
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
        smp_hind = _resample(hind, resample_dim)
        # compute init skill
        init_skill = compute(
            smp_hind,
            verif,
            metric=metric,
            comparison=comparison,
            add_attrs=False,
            dim=dim,
            **metric_kwargs,
        )
        # reset inits when probabilistic, otherwise tests fail
        if (
            resample_dim == "init"
            and metric.probabilistic
            and "init" in init_skill.coords
        ):
            init_skill["init"] = hind.init.values
        bootstrapped_init_skill.append(init_skill)
        if "uninitialized" in reference:
            # generate uninitialized ensemble from hist
            uninit_hind = resample_uninit(hind, hist)
            # compute uninit skill
            bootstrapped_uninit_skill.append(
                compute(
                    uninit_hind,
                    verif,
                    metric=metric,
                    comparison=comparison,
                    dim=dim,
                    add_attrs=False,
                    **metric_kwargs,
                )
            )
        if "persistence" in reference:
            # compute persistence skill
            # impossible for probabilistic
            if not metric.probabilistic:
                pers_skill.append(
                    reference_compute(
                        smp_hind,
                        verif,
                        metric=metric,
                        dim=dim,
                        add_attrs=False,
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
        if pers_skill != []:
            bootstrapped_pers_skill = xr.concat(
                pers_skill, dim="iteration", **CONCAT_KWARGS
            )
    else:
        bootstrapped_pers_skill = None
    return (
        bootstrapped_init_skill,
        bootstrapped_uninit_skill,
        bootstrapped_pers_skill,
    )


def _get_resample_func(ds):
    """Decide for resample function based on input `ds`.

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
        ds (xr.object): input data.
        dims (list of str or str): Dimensions to auto-chunk in.

    Returns:
        xr.object: auto-chunked along `dims`

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
    """Chunk ds so small that after _resample_iteration_idx chunks have optimal size
    `optimal_blocksize`.

    Args:
        ds (xr.obejct): input data`.
        iterations (int): number of bootstrap iterations in `_resample_iterations_idx`.
        chunking_dims (list of str or str): Dimension(s) to chunking in.
        optimal_blocksize (int): dask blocksize to aim at in bytes.
            Defaults to 100000000.

    Returns:
        xr.object: chunked to have blocksize: optimal_blocksize/iterations.

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
    hind,
    verif,
    hist=None,
    alignment="same_verifs",
    metric="pearson_r",
    comparison="m2e",
    dim="init",
    reference=["uninitialized", "persistence"],
    resample_dim="member",
    sig=95,
    iterations=500,
    pers_sig=None,
    compute=compute_hindcast,
    resample_uninit=bootstrap_uninitialized_ensemble,
    reference_compute=compute_persistence,
    **metric_kwargs,
):
    """Bootstrap compute with replacement.

    Args:
        hind (xr.Dataset): prediction ensemble.
        verif (xr.Dataset): Verification data.
        hist (xr.Dataset): historical/uninitialized simulation.
        metric (str): `metric`. Defaults to 'pearson_r'.
        comparison (str): `comparison`. Defaults to 'm2e'.
        dim (str or list): dimension(s) to apply metric over. default: 'init'.
        reference (str, list of str): Type of reference forecasts with which to
            verify. One or more of ['persistence', 'uninitialized'].
            If None or empty, returns no p value.
        resample_dim (str): dimension to resample from. default: 'member'::

            - 'member': select a different set of members from hind
            - 'init': select a different set of initializations from hind

        sig (int): Significance level for uninitialized and
                   initialized skill. Defaults to 95.
        pers_sig (int): Significance level for persistence skill confidence levels.
                        Defaults to sig.
        iterations (int): number of resampling iterations (bootstrap
                         with replacement). Defaults to 500.
        compute (func): function to compute skill.
                        Choose from
                        [:py:func:`climpred.prediction.compute_perfect_model`,
                         :py:func:`climpred.prediction.compute_hindcast`].
        resample_uninit (func): function to create an uninitialized ensemble
                        from a control simulation or uninitialized large
                        ensemble. Choose from:
                        [:py:func:`bootstrap_uninitialized_ensemble`,
                         :py:func:`bootstrap_uninit_pm_ensemble_from_control`].
        reference_compute (func): function to compute a reference forecast skill with.
                        Default: :py:func:`climpred.prediction.compute_persistence`.
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        results: (xr.Dataset): bootstrapped results for the three different skills:

            - `initialized` for the initialized hindcast `hind` and describes skill due
             to initialization and external forcing
            - `uninitialized` for the uninitialized/historical and approximates skill
             from external forcing
            - `persistence` for the persistence forecast computed by
              `compute_persistence`

        the different results:
            - `verify skill`: skill values
            - `p`: p value
            - `low_ci` and `high_ci`: high and low ends of confidence intervals based
             on significance threshold `sig`


    Reference:
        * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
          Gonzalez, V. Kharin, et al. “A Verification Framework for
          Interannual-to-Decadal Predictions Experiments.” Climate
          Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
          https://doi.org/10/f4jjvf.

    See also:
        * climpred.bootstrap.bootstrap_hindcast
        * climpred.bootstrap.bootstrap_perfect_model
    """
    warn_if_chunking_would_increase_performance(hind, crit_size_in_MB=5)
    if pers_sig is None:
        pers_sig = sig
    if isinstance(dim, str):
        dim = [dim]
    if isinstance(reference, str):
        reference = [reference]
    if reference is None:
        reference = []

    p = (100 - sig) / 100
    ci_low = p / 2
    ci_high = 1 - p / 2
    p_pers = (100 - pers_sig) / 100
    ci_low_pers = p_pers / 2
    ci_high_pers = 1 - p_pers / 2

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
    chunking_dims = [d for d in hind.dims if d not in CLIMPRED_DIMS]

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
            hind,
            hist,
            verif,
            dim,
            reference,
            resample_dim,
            iterations,
            metric,
            comparison,
            compute,
            reference_compute,
            resample_uninit,
            **metric_kwargs,
        )
    else:  # faster: first _resample_iterations_idx, then compute skill
        resample_func = _get_resample_func(hind)
        if not isHindcast:
            if "uninitialized" in reference:
                # create more members than needed in PM to make the uninitialized
                # distribution more robust
                members_to_sample_from = 50
                repeat = members_to_sample_from // hind.member.size + 1
                uninit_hind = xr.concat(
                    [resample_uninit(hind, hist) for i in range(repeat)],
                    dim="member",
                    **CONCAT_KWARGS,
                )
                uninit_hind["member"] = np.arange(1, 1 + uninit_hind.member.size)
                if dask.is_dask_collection(uninit_hind):
                    # too minimize tasks: ensure uninit_hind get pre-computed
                    # alternativly .chunk({'member':-1})
                    uninit_hind = uninit_hind.compute().chunk()
                # resample uninit always over member and select only hind.member.size
                bootstrapped_uninit = resample_func(
                    uninit_hind,
                    iterations,
                    "member",
                    replace=False,
                    dim_max=hind["member"].size,
                )
                bootstrapped_uninit["lead"] = hind["lead"]
                # effectively only when _resample_iteration_idx which doesnt use dim_max
                bootstrapped_uninit = bootstrapped_uninit.isel(
                    member=slice(None, hind.member.size)
                )
                if dask.is_dask_collection(bootstrapped_uninit):
                    bootstrapped_uninit = bootstrapped_uninit.chunk({"member": -1})
                    bootstrapped_uninit = _maybe_auto_chunk(
                        bootstrapped_uninit, ["iteration"] + chunking_dims
                    )
        else:  # hindcast
            if "uninitialized" in reference:
                uninit_hind = resample_uninit(hind, hist)
                if dask.is_dask_collection(uninit_hind):
                    # too minimize tasks: ensure uninit_hind get pre-computed
                    # maybe not needed
                    uninit_hind = uninit_hind.compute().chunk()
                bootstrapped_uninit = resample_func(
                    uninit_hind, iterations, resample_dim
                )
                bootstrapped_uninit = bootstrapped_uninit.isel(
                    member=slice(None, hind.member.size)
                )
                bootstrapped_uninit["lead"] = hind["lead"]
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
                add_attrs=False,
                **metric_kwargs,
            )
            # take mean if 'm2o' comparison forced before
            if isHindcast and comparison != __m2o:
                bootstrapped_uninit_skill = bootstrapped_uninit_skill.mean("member")

        bootstrapped_hind = resample_func(hind, iterations, resample_dim)
        if dask.is_dask_collection(bootstrapped_hind):
            bootstrapped_hind = bootstrapped_hind.chunk({"member": -1})

        bootstrapped_init_skill = compute(
            bootstrapped_hind,
            verif,
            metric=metric,
            comparison=comparison,
            add_attrs=False,
            dim=dim,
            **metric_kwargs,
        )
        if "persistence" in reference:
            if not metric.probabilistic:
                pers_skill = reference_compute(
                    hind, verif, metric=metric, dim=dim, **metric_kwargs_reference,
                )
                # bootstrap pers
                if resample_dim == "init":
                    bootstrapped_pers_skill = reference_compute(
                        bootstrapped_hind,
                        verif,
                        metric=metric,
                        **metric_kwargs_reference,
                    )
                else:  # member
                    _, bootstrapped_pers_skill = xr.broadcast(
                        bootstrapped_init_skill, pers_skill, exclude=CLIMPRED_DIMS
                    )
            else:
                bootstrapped_pers_skill = bootstrapped_init_skill.isnull()

    # calc mean skill without any resampling
    init_skill = compute(
        hind, verif, metric=metric, comparison=comparison, dim=dim, **metric_kwargs,
    )

    if "uninitialized" in reference:
        # uninit skill as mean resampled uninit skill
        uninit_skill = bootstrapped_uninit_skill.mean("iteration")
    if "persistence" in reference:
        if not metric.probabilistic:
            pers_skill = reference_compute(
                hind, verif, metric=metric, dim=dim, **metric_kwargs_reference
            )
        else:
            pers_skill = init_skill.isnull()

        # align to prepare for concat
        if set(bootstrapped_pers_skill.coords) != set(bootstrapped_init_skill.coords):
            if (
                "time" in bootstrapped_pers_skill.dims
                and "init" in bootstrapped_init_skill.dims
            ):
                bootstrapped_pers_skill = bootstrapped_pers_skill.rename(
                    {"time": "init"}
                )
            # allow member to be broadcasted
            bootstrapped_init_skill, bootstrapped_pers_skill = xr.broadcast(
                bootstrapped_init_skill,
                bootstrapped_pers_skill,
                exclude=("init", "lead", "time"),
            )

    # get confidence intervals CI
    init_ci = _distribution_to_ci(bootstrapped_init_skill, ci_low, ci_high)
    if "uninitialized" in reference:
        uninit_ci = _distribution_to_ci(bootstrapped_uninit_skill, ci_low, ci_high)

    # probabilistic metrics wont have persistence forecast
    # therefore only get CI if persistence was computed
    if "persistence" in reference:
        if "iteration" in bootstrapped_pers_skill.dims:
            pers_ci = _distribution_to_ci(
                bootstrapped_pers_skill, ci_low_pers, ci_high_pers
            )
        else:
            # otherwise set all persistence outputs to false
            pers_ci = init_ci == -999

    # pvalue whether uninit or pers better than init forecast
    if "uninitialized" in reference:
        p_uninit_over_init = _pvalue_from_distributions(
            bootstrapped_uninit_skill, bootstrapped_init_skill, metric=metric
        )
    if "persistence" in reference:
        p_pers_over_init = _pvalue_from_distributions(
            bootstrapped_pers_skill, bootstrapped_init_skill, metric=metric
        )

    # wrap results together in one xr object
    if reference == []:
        results = xr.concat(
            [
                init_skill,
                init_ci.isel(quantile=0, drop=True),
                init_ci.isel(quantile=1, drop=True),
            ],
            dim="results",
        )
        results["results"] = ["verify skill", "low_ci", "high_ci"]
        results["skill"] = ["initialized"]
        results = results.squeeze()

    elif reference == ["persistence"]:
        skill = xr.concat([init_skill, pers_skill], dim="skill", **CONCAT_KWARGS)
        skill["skill"] = ["initialized", "persistence"]

        # ci for each skill
        ci = xr.concat([init_ci, pers_ci], "skill", coords="minimal").rename(
            {"quantile": "results"}
        )
        ci["skill"] = ["initialized", "persistence"]

        results = xr.concat([skill, p_pers_over_init], dim="results", **CONCAT_KWARGS)
        results["results"] = ["verify skill", "p"]
        if set(results.coords) != set(ci.coords):
            res_drop = [c for c in results.coords if c not in ci.coords]
            ci_drop = [c for c in ci.coords if c not in results.coords]
            results = results.drop_vars(res_drop)
            ci = ci.drop_vars(ci_drop)
        results = xr.concat([results, ci], dim="results", **CONCAT_KWARGS)
        results["results"] = ["verify skill", "p", "low_ci", "high_ci"]

    elif reference == ["uninitialized"]:
        skill = xr.concat([init_skill, uninit_skill], dim="skill", **CONCAT_KWARGS)
        skill["skill"] = ["initialized", "uninitialized"]

        # ci for each skill
        ci = xr.concat([init_ci, uninit_ci], "skill", coords="minimal").rename(
            {"quantile": "results"}
        )
        ci["skill"] = ["initialized", "uninitialized"]

        results = xr.concat([skill, p_uninit_over_init], dim="results", **CONCAT_KWARGS)
        results["results"] = ["verify skill", "p"]
        if set(results.coords) != set(ci.coords):
            res_drop = [c for c in results.coords if c not in ci.coords]
            ci_drop = [c for c in ci.coords if c not in results.coords]
            results = results.drop_vars(res_drop)
            ci = ci.drop_vars(ci_drop)
        results = xr.concat([results, ci], dim="results", **CONCAT_KWARGS)
        results["results"] = ["verify skill", "p", "low_ci", "high_ci"]

    elif set(reference) == set(["uninitialized", "persistence"]):
        skill = xr.concat(
            [init_skill, uninit_skill, pers_skill], dim="skill", **CONCAT_KWARGS
        )
        skill["skill"] = ["initialized", "uninitialized", "persistence"]

        # probability that i beats init
        p = xr.concat(
            [p_uninit_over_init, p_pers_over_init], dim="skill", **CONCAT_KWARGS
        )
        p["skill"] = ["uninitialized", "persistence"]

        # ci for each skill
        ci = xr.concat([init_ci, uninit_ci, pers_ci], "skill", coords="minimal").rename(
            {"quantile": "results"}
        )
        ci["skill"] = ["initialized", "uninitialized", "persistence"]

        results = xr.concat([skill, p], dim="results", **CONCAT_KWARGS)
        results["results"] = ["verify skill", "p"]
        if set(results.coords) != set(ci.coords):
            res_drop = [c for c in results.coords if c not in ci.coords]
            ci_drop = [c for c in ci.coords if c not in results.coords]
            results = results.drop_vars(res_drop)
            ci = ci.drop_vars(ci_drop)
        results = xr.concat([results, ci], dim="results", **CONCAT_KWARGS)
        results["results"] = ["verify skill", "p", "low_ci", "high_ci"]
    else:
        raise ValueError("results not created")

    # Attach climpred compute information to skill
    metadata_dict = {
        "confidence_interval_levels": f"{ci_high}-{ci_low}",
        "bootstrap_iterations": iterations,
        "reference": reference,
    }
    if reference is not None:
        metadata_dict[
            "p"
        ] = "probability that reference performs better than initialized"
    metadata_dict.update(metric_kwargs)
    results = assign_attrs(
        results,
        hind,
        alignment=alignment,
        metric=metric,
        comparison=comparison,
        dim=dim,
        function_name=inspect.stack()[0][3],  # take function.__name__
        metadata_dict=metadata_dict,
    )
    # Ensure that the lead units get carried along for the calculation. The attribute
    # tends to get dropped along the way due to ``xarray`` functionality.
    results["lead"] = hind["lead"]
    if "units" in hind["lead"].attrs and "units" not in results["lead"].attrs:
        results["lead"].attrs["units"] = hind["lead"].attrs["units"]
    return results


def bootstrap_hindcast(
    hind,
    hist,
    verif,
    alignment="same_verifs",
    metric="pearson_r",
    comparison="e2o",
    dim="init",
    reference=["uninitialized", "persistence"],
    resample_dim="member",
    sig=95,
    iterations=500,
    pers_sig=None,
    reference_compute=compute_persistence,
    **metric_kwargs,
):
    """Bootstrap compute with replacement. Wrapper of
     py:func:`bootstrap_compute` for hindcasts.

    Args:
        hind (xr.Dataset): prediction ensemble.
        verif (xr.Dataset): Verification data.
        hist (xr.Dataset): historical/uninitialized simulation.
        metric (str): `metric`. Defaults to 'pearson_r'.
        comparison (str): `comparison`. Defaults to 'e2o'.
        dim (str): dimension to apply metric over. default: 'init'.
        reference (str, list of str): Type of reference forecasts with which to
            verify. One or more of ['persistence', 'uninitialized'].
            If None or empty, returns no p value.
        resample_dim (str or list): dimension to resample from. default: 'member'.

            - 'member': select a different set of members from hind
            - 'init': select a different set of initializations from hind

        sig (int): Significance level for uninitialized and
                   initialized skill. Defaults to 95.
        pers_sig (int): Significance level for persistence skill confidence levels.
                        Defaults to sig.
        iterations (int): number of resampling iterations (bootstrap
                         with replacement). Defaults to 500.
        reference_compute (func): function to compute a reference forecast skill with.
                        Default: :py:func:`climpred.prediction.compute_persistence`.
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        results: (xr.Dataset): bootstrapped results for the three different kinds of
                               predictions:

            - `initialized` for the initialized hindcast `hind` and describes skill due
             to initialization and external forcing
            - `uninitialized` for the uninitialized/historical and approximates skill
             from external forcing
            - `persistence` for the persistence forecast computed by
             `compute_persistence`

        the different results:
            - `verify skill`: skill values
            - `p`: p value
            - `low_ci` and `high_ci`: high and low ends of confidence intervals based
             on significance threshold `sig`

    Reference:
        * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
          Gonzalez, V. Kharin, et al. “A Verification Framework for
          Interannual-to-Decadal Predictions Experiments.” Climate
          Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
          https://doi.org/10/f4jjvf.

    See also:
        * climpred.bootstrap.bootstrap_compute
        * climpred.prediction.compute_hindcast

    Example:
        >>> hind = climpred.tutorial.load_dataset('CESM-DP-SST')['SST']
        >>> hist = climpred.tutorial.load_dataset('CESM-LE')['SST']
        >>> obs = load_dataset('ERSST')['SST']
        >>> bootstrapped_skill = climpred.bootstrap.bootstrap_hindcast(hind, hist, obs)
        >>> bootstrapped_skill.coords
        Coordinates:
          * lead     (lead) int64 1 2 3 4 5 6 7 8 9 10
          * kind     (kind) object 'initialized' 'persistence' 'uninitialized'
          * results  (results) <U7 'verify skill' 'p' 'low_ci' 'high_ci'

    """
    # Check that init is int, cftime, or datetime; convert ints or datetime to cftime.
    hind = convert_time_index(hind, "init", "hind[init]")
    hist = convert_time_index(hist, "time", "uninitialized[time]")
    verif = convert_time_index(verif, "time", "verif[time]")
    # Put this after `convert_time_index` since it assigns 'years' attribute if the
    # `init` dimension is a `float` or `int`.
    has_valid_lead_units(hind)

    if ("same_verif" in alignment) & (resample_dim == "init"):
        raise KeywordError(
            "Cannot have both alignment='same_verifs' and "
            "resample_dim='init'. Change `resample_dim` to 'member' to keep "
            "common verification alignment or `alignment` to 'same_inits' to "
            "resample over initializations."
        )

    # Kludge for now. Since we're computing persistence here we need to ensure that
    # all products have a union in their time axis.
    times = np.sort(
        list(set(hind.init.data) & set(hist.time.data) & set(verif.time.data))
    )
    hind = hind.sel(init=times)
    hist = hist.sel(time=times)
    verif = verif.sel(time=times)

    return bootstrap_compute(
        hind,
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
        reference_compute=reference_compute,
        **metric_kwargs,
    )


def bootstrap_perfect_model(
    init_pm,
    control,
    metric="pearson_r",
    comparison="m2e",
    dim=["init", "member"],
    reference=["uninitialized", "persistence"],
    resample_dim="member",
    sig=95,
    iterations=500,
    pers_sig=None,
    reference_compute=compute_persistence,
    **metric_kwargs,
):
    """Bootstrap compute with replacement. Wrapper of
     py:func:`bootstrap_compute` for perfect-model framework.

    Args:
        hind (xr.Dataset): prediction ensemble.
        verif (xr.Dataset): Verification data.
        hist (xr.Dataset): historical/uninitialized simulation.
        metric (str): `metric`. Defaults to 'pearson_r'.
        comparison (str): `comparison`. Defaults to 'm2e'.
        dim (str): dimension to apply metric over. default: ['init', 'member'].
        reference (str, list of str): Type of reference forecasts with which to
            verify. One or more of ['persistence', 'uninitialized'].
            If None or empty, returns no p value.
        resample_dim (str or list): dimension to resample from. default: 'member'.

            - 'member': select a different set of members from hind
            - 'init': select a different set of initializations from hind

        sig (int): Significance level for uninitialized and
                   initialized skill. Defaults to 95.
        pers_sig (int): Significance level for persistence skill confidence levels.
                        Defaults to sig.
        iterations (int): number of resampling iterations (bootstrap
                         with replacement). Defaults to 500.
        reference_compute (func): function to compute a reference forecast skill with.
                        Default: :py:func:`climpred.prediction.compute_persistence`.
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        results: (xr.Dataset): bootstrapped results for the three different kinds of
                               predictions:

            - `initialized` for the initialized hindcast `hind` and describes skill due
             to initialization and external forcing
            - `uninitialized` for the uninitialized/historical and approximates skill
             from external forcing
            - `pers` for the reference forecast computed by `reference_compute`, which
             defaults to `compute_persistence`

        the different results:
            - `skill`: skill values
            - `p`: p value
            - `low_ci` and `high_ci`: high and low ends of confidence intervals based
             on significance threshold `sig`

    Reference:
        * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
          Gonzalez, V. Kharin, et al. “A Verification Framework for
          Interannual-to-Decadal Predictions Experiments.” Climate
          Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
          https://doi.org/10/f4jjvf.

    See also:
        * climpred.bootstrap.bootstrap_compute
        * climpred.prediction.compute_perfect_model

    Example:
        >>> init = climpred.tutorial.load_dataset('MPI-PM-DP-1D')
        >>> control = climpred.tutorial.load_dataset('MPI-control-1D')
        >>> bootstrapped_s = climpred.bootstrap.bootstrap_perfect_model(init, control)
        >>> bootstrapped_s.coords
        Coordinates:
          * lead     (lead) int64 1 2 3 4 5 6 7 8 9 10
          * kind     (kind) object 'initialized' 'persistence' 'uninitialized'
          * results  (results) <U7 'verify skill' 'p' 'low_ci' 'high_ci'
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
        reference_compute=reference_compute,
        **metric_kwargs,
    )


def _bootstrap_func(
    func, ds, resample_dim, sig=95, iterations=500, *func_args, **func_kwargs,
):
    """Sig % threshold of function based on iterations resampling with replacement.

    Reference:
    * Mason, S. J., and G. M. Mimmack. “The Use of Bootstrap Confidence
     Intervals for the Correlation Coefficient in Climatology.” Theoretical and
      Applied Climatology 45, no. 4 (December 1, 1992): 229–33.
      https://doi.org/10/b6fnsv.

    Args:
        func (function): function to be bootstrapped.
        ds (xr.object): first input argument of func. `chunk` ds on `dim` other
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
        * Feng, X., T. DelSole, and P. Houser. “Bootstrap Estimated Seasonal
          Potential Predictability of Global Temperature and Precipitation.”
          Geophysical Research Letters 38, no. 7 (2011).
          https://doi.org/10/ft272w.

    See also:
        * climpred.bootstrap._bootstrap_func
        * climpred.stats.dpp
    """
    return _bootstrap_func(
        dpp, control, dim, sig=sig, iterations=iterations, **dpp_kwargs
    )


def varweighted_mean_period_threshold(control, sig=95, iterations=500, time_dim="time"):
    """Calc the variance-weighted mean period significance levels from re-sampled
    dataset.

    See also:
        * climpred.bootstrap._bootstrap_func
        * climpred.stats.varweighted_mean_period
    """
    return _bootstrap_func(
        varweighted_mean_period, control, time_dim, sig=sig, iterations=iterations,
    )
