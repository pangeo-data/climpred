import inspect

import dask
import numpy as np
import xarray as xr

from .checks import (
    has_dims,
    has_valid_lead_units,
    warn_if_chunking_would_increase_performance,
)
from .comparisons import ALL_COMPARISONS, COMPARISON_ALIASES
from .metrics import ALL_METRICS, METRIC_ALIASES
from .prediction import compute_hindcast, compute_perfect_model, compute_persistence
from .stats import dpp, varweighted_mean_period
from .utils import (
    _transpose_and_rechunk_to,
    assign_attrs,
    convert_time_index,
    get_comparison_class,
    get_lead_cftime_shift_args,
    get_metric_class,
    shift_cftime_singular,
)


def _resample(hind, resample_dim, to_be_resampled):
    """Resample with replacement in dimension `resample_dim` from values of
    `to_be_resampled`

    Args:
        hind (xr.object): input xr.object to be resampled.
        resample_dim (str): dimension to resample along.
        to_be_resampled (list, xr.DataArray.values, np.ndarray): values to resample
            from.

    Returns:
        xr.object: resampled along `resample_dim`.

    """
    smp = np.random.choice(to_be_resampled, len(to_be_resampled))
    smp_hind = hind.sel({resample_dim: smp})
    # ignore because then inits should keep their labels
    if resample_dim != 'init':
        smp_hind[resample_dim] = hind[resample_dim]
    return smp_hind


def my_quantile(ds, q=0.95, dim='bootstrap'):
    """Compute quantile `q` faster than `xr.quantile` and allows lazy computation."""
    # dim='bootstrap' doesnt lead anywhere, but want to keep xr.quantile API
    # concat_dim is always first, therefore axis=0 implementation works in compute

    def _dask_percentile(arr, axis=0, q=95):
        """Daskified np.percentile."""
        if len(arr.chunks[axis]) > 1:
            arr = arr.rechunk({axis: -1})
        return dask.array.map_blocks(np.percentile, arr, axis=axis, q=q, drop_axis=axis)

    def _percentile(arr, axis=0, q=95):
        """percentile function for chunked and non-chunked `arr`."""
        if dask.is_dask_collection(arr):
            return _dask_percentile(arr, axis=axis, q=q)
        else:
            return np.percentile(arr, axis=axis, q=q)

    axis = 0
    if not isinstance(q, list):
        q = [q]
    quantile = []
    for qi in q:
        quantile.append(ds.reduce(_percentile, q=qi * 100, axis=axis))
    quantile = xr.concat(quantile, 'quantile')
    quantile['quantile'] = q
    return quantile.squeeze()


def _distribution_to_ci(ds, ci_low, ci_high, dim='bootstrap'):
    """Get confidence intervals from bootstrapped distribution.

    Needed for bootstrapping confidence intervals and p_values of a metric.

    Args:
        ds (xarray object): distribution.
        ci_low (float): low confidence interval.
        ci_high (float): high confidence interval.
        dim (str): dimension to apply xr.quantile to. Default: 'bootstrap'

    Returns:
        uninit_hind (xarray object): uninitialize hindcast with hind.coords.
    """
    # TODO: re-implement xr.quantile once fast
    return my_quantile(ds, q=[ci_low, ci_high], dim='bootstrap')


def _pvalue_from_distributions(simple_fct, init, metric=None):
    """Get probability that skill of a simple forecast (e.g., persistence or
    uninitlaized skill) is larger than initialized skill.

    Needed for bootstrapping confidence intervals and p_values of a metric in
    the hindcast framework. Checks whether a simple forecast like persistence
    or uninitialized performs better than initialized forecast. Need to keep in
    mind the orientation of metric (whether larger values are better or worse
    than smaller ones.)

    Args:
        simple_fct (xarray object): persistence or uninit skill.
        init (xarray object): hindcast skill.
        metric (Metric): metric class Metric

    Returns:
        pv (xarray object): probability that simple forecast performs better
                            than initialized forecast.
    """
    pv = ((simple_fct - init) > 0).sum('bootstrap') / init.bootstrap.size
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
    has_dims(hist, 'member', 'historical ensemble')
    has_dims(hind, 'member', 'initialized hindcast ensemble')
    # Put this after `convert_time_index` since it assigns 'years' attribute if the
    # `init` dimension is a `float` or `int`.
    has_valid_lead_units(hind)

    # find range for bootstrapping
    first_init = max(hist.time.min(), hind['init'].min())

    n, freq = get_lead_cftime_shift_args(hind.lead.attrs['units'], hind.lead.size)
    hist_last = shift_cftime_singular(hist.time.max(), -1 * n, freq)
    last_init = min(hist_last, hind['init'].max())

    hind = hind.sel(init=slice(first_init, last_init))

    uninit_hind = []
    for init in hind.init.values:
        random_members = np.random.choice(hist.member.values, hind.member.size)
        # take random uninitialized members from hist at init forcing
        # (Goddard allows 5 year forcing range here)
        # TODO: implement these 5 years
        uninit_at_one_init_year = hist.sel(
            time=slice(
                shift_cftime_singular(init, 1, freq),
                shift_cftime_singular(init, n, freq),
            ),
            member=random_members,
        ).rename({'time': 'lead'})
        uninit_at_one_init_year['lead'] = np.arange(
            1, 1 + uninit_at_one_init_year['lead'].size
        )
        uninit_at_one_init_year['member'] = np.arange(1, 1 + len(random_members))
        uninit_hind.append(uninit_at_one_init_year)
    uninit_hind = xr.concat(uninit_hind, 'init')
    uninit_hind['init'] = hind['init'].values
    uninit_hind.lead.attrs['units'] = hind.lead.attrs['units']
    return (
        _transpose_and_rechunk_to(uninit_hind, hind)
        if dask.is_dask_collection(uninit_hind)
        else uninit_hind
    )


def bootstrap_uninit_pm_ensemble_from_control(ds, control):
    """
    Create a pseudo-ensemble from control run.

    Note:
        Needed for block bootstrapping confidence intervals of a metric in perfect
        model framework. Takes randomly segments of length of ensemble dataset from
        control and rearranges them into ensemble and member dimensions.

    Args:
        ds (xarray object): ensemble simulation.
        control (xarray object): control simulation.

    Returns:
        ds_e (xarray object): pseudo-ensemble generated from control run.
    """
    nens = ds.init.size
    nmember = ds.member.size
    length = ds.lead.size
    c_start = 0
    c_end = control['time'].size
    lead_time = ds['lead']

    def isel_years(control, year_s, length):
        new = control.isel(time=slice(year_s, year_s + length))
        new = new.rename({'time': 'lead'})
        new['lead'] = lead_time
        return new

    def create_pseudo_members(control):
        startlist = np.random.randint(c_start, c_end - length - 1, nmember)
        return xr.concat(
            (isel_years(control, start, length) for start in startlist), 'member',
        )

    uninit = xr.concat((create_pseudo_members(control) for _ in range(nens)), 'init')
    # chunk to same dims
    return (
        _transpose_and_rechunk_to(uninit, ds)
        if dask.is_dask_collection(uninit)
        else uninit
    )


def _bootstrap_func(
    func, ds, resample_dim, sig=95, bootstrap=500, *func_args, **func_kwargs
):
    """Sig percent threshold of function based on resampling with replacement.

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
        bootstrap (int): number of resample iterations. Defaults to 500.
        *func_args (type): `*func_args`.
        **func_kwargs (type): `**func_kwargs`.

    Returns:
        sig_level: bootstrapped significance levels with
                   dimensions of ds and len(sig) if sig is list
    """
    if not callable(func):
        raise ValueError(f'Please provide func as a function, found {type(func)}')
    warn_if_chunking_would_increase_performance(ds)
    if isinstance(sig, list):
        psig = [i / 100 for i in sig]
    else:
        psig = sig / 100

    bootstraped_results = []
    resample_dim_values = ds[resample_dim].values
    for _ in range(bootstrap):
        smp_ds = _resample(ds, resample_dim, resample_dim_values)
        bootstraped_results.append(func(smp_ds, *func_args, **func_kwargs))
    sig_level = xr.concat(bootstraped_results, 'bootstrap')
    # TODO: reimplement xr.quantile once fast
    sig_level = my_quantile(sig_level, dim='bootstrap', q=psig)
    return sig_level


def dpp_threshold(control, sig=95, bootstrap=500, dim='time', **dpp_kwargs):
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
        dpp, control, dim, sig=sig, bootstrap=bootstrap, **dpp_kwargs
    )


def varweighted_mean_period_threshold(control, sig=95, bootstrap=500, time_dim='time'):
    """Calc the variance-weighted mean period significance levels from re-sampled dataset.

    See also:
        * climpred.bootstrap._bootstrap_func
        * climpred.stats.varweighted_mean_period
    """
    return _bootstrap_func(
        varweighted_mean_period, control, time_dim, sig=sig, bootstrap=bootstrap,
    )


def bootstrap_compute(
    hind,
    verif,
    hist=None,
    alignment='same_inits',
    metric='pearson_r',
    comparison='m2e',
    dim='init',
    resample_dim='member',
    sig=95,
    bootstrap=500,
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
        resample_dim (str): dimension to resample from. default: 'member'::

            - 'member': select a different set of members from hind
            - 'init': select a different set of initializations from hind

        sig (int): Significance level for uninitialized and
                   initialized skill. Defaults to 95.
        pers_sig (int): Significance level for persistence skill confidence levels.
                        Defaults to sig.
        bootstrap (int): number of resampling iterations (bootstrap
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
        results: (xr.Dataset): bootstrapped results for the three different kinds of
                               predictions:

            - `init` for the initialized hindcast `hind` and describes skill due to
             initialization and external forcing
            - `uninit` for the uninitialized historical `hist` and approximates skill
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
        * climpred.bootstrap.bootstrap_hindcast
        * climpred.bootstrap.bootstrap_perfect_model
    """
    warn_if_chunking_would_increase_performance(hind)
    if pers_sig is None:
        pers_sig = sig

    p = (100 - sig) / 100
    ci_low = p / 2
    ci_high = 1 - p / 2
    p_pers = (100 - pers_sig) / 100
    ci_low_pers = p_pers / 2
    ci_high_pers = 1 - p_pers / 2

    init = []
    uninit = []
    pers = []

    # get metric/comparison function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)
    comparison = COMPARISON_ALIASES.get(comparison, comparison)

    # get class Metric(metric)
    metric = get_metric_class(metric, ALL_METRICS)
    # get comparison function
    comparison = get_comparison_class(comparison, ALL_COMPARISONS)

    to_be_resampled = hind[resample_dim].values

    for i in range(bootstrap):
        # resample with replacement
        smp_hind = _resample(hind, resample_dim, to_be_resampled)
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
            resample_dim == 'init'
            and metric.probabilistic
            and 'init' in init_skill.coords
        ):
            init_skill['init'] = hind.init.values
        init.append(init_skill)
        # generate uninitialized ensemble from hist
        if hist is None:  # PM path, use verif = control
            hist = verif
        uninit_hind = resample_uninit(hind, hist)
        # compute uninit skill
        uninit.append(
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
        # compute persistence skill
        # impossible for probabilistic
        if not metric.probabilistic:
            pers.append(
                reference_compute(smp_hind, verif, metric=metric, **metric_kwargs)
            )
    init = xr.concat(init, dim='bootstrap')
    # remove useless member = 0 coords after m2c
    if 'member' in init.coords and init.member.size == 1:
        if init.member.size == 1:
            del init['member']
    uninit = xr.concat(uninit, dim='bootstrap')
    # when persistence is not computed set flag
    if pers != []:
        pers = xr.concat(pers, dim='bootstrap')
        pers_output = True
    else:
        pers_output = False

    # get confidence intervals CI
    init_ci = _distribution_to_ci(init, ci_low, ci_high)
    uninit_ci = _distribution_to_ci(uninit, ci_low, ci_high)
    # probabilistic metrics wont have persistence forecast
    # therefore only get CI if persistence was computed
    if pers_output:
        if set(pers.coords) != set(init.coords):
            init, pers = xr.broadcast(init, pers)
        pers_ci = _distribution_to_ci(pers, ci_low_pers, ci_high_pers)
    else:
        # otherwise set all persistence outputs to false
        pers = init.isnull()
        pers_ci = init_ci == -999

    # pvalue whether uninit or pers better than init forecast
    p_uninit_over_init = _pvalue_from_distributions(uninit, init, metric=metric)
    p_pers_over_init = _pvalue_from_distributions(pers, init, metric=metric)

    # calc mean skill without any resampling
    init_skill = compute(
        hind, verif, metric=metric, comparison=comparison, dim=dim, **metric_kwargs,
    )
    if 'init' in init_skill:
        init_skill = init_skill.mean('init')
    # remove useless member = 0 coords after m2c
    if 'member' in init_skill.coords and init_skill.member.size == 1:
        del init_skill['member']
    # uninit skill as mean resampled uninit skill
    uninit_skill = uninit.mean('bootstrap')
    if not metric.probabilistic:
        pers_skill = reference_compute(hind, verif, metric=metric, **metric_kwargs)
    else:
        pers_skill = init_skill.isnull()
    # align to prepare for concat
    if set(pers_skill.coords) != set(init_skill.coords):
        init_skill, pers_skill = xr.broadcast(init_skill, pers_skill)

    # wrap results together in one dataarray
    skill = xr.concat([init_skill, uninit_skill, pers_skill], 'kind')
    skill['kind'] = ['init', 'uninit', 'pers']

    # probability that i beats init
    p = xr.concat([p_uninit_over_init, p_pers_over_init], 'kind')
    p['kind'] = ['uninit', 'pers']

    # ci for each skill
    ci = xr.concat([init_ci, uninit_ci, pers_ci], 'kind').rename(
        {'quantile': 'results'}
    )
    ci['kind'] = ['init', 'uninit', 'pers']

    results = xr.concat([skill, p], 'results')
    results['results'] = ['skill', 'p']
    if set(results.coords) != set(ci.coords):
        res_drop = [c for c in results.coords if c not in ci.coords]
        ci_drop = [c for c in ci.coords if c not in results.coords]
        results = results.drop_vars(res_drop)
        ci = ci.drop_vars(ci_drop)
    results = xr.concat([results, ci], 'results')
    results['results'] = ['skill', 'p', 'low_ci', 'high_ci']
    # Attach climpred compute information to skill
    metadata_dict = {
        'confidence_interval_levels': f'{ci_high}-{ci_low}',
        'bootstrap_iterations': bootstrap,
        'p': 'probability that uninitialized ensemble performs better than initialized',
        'reference_compute': reference_compute.__name__,
    }
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
    if 'units' in hind['lead'].attrs and 'units' not in results['lead'].attrs:
        results['lead'].attrs['units'] = hind['lead'].attrs['units']
    return results


def bootstrap_hindcast(
    hind,
    hist,
    verif,
    alignment='same_inits',
    metric='pearson_r',
    comparison='e2o',
    dim='init',
    resample_dim='member',
    sig=95,
    bootstrap=500,
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
        resample_dim (str or list): dimension to resample from. default: 'member'.

            - 'member': select a different set of members from hind
            - 'init': select a different set of initializations from hind

        sig (int): Significance level for uninitialized and
                   initialized skill. Defaults to 95.
        pers_sig (int): Significance level for persistence skill confidence levels.
                        Defaults to sig.
        bootstrap (int): number of resampling iterations (bootstrap
                         with replacement). Defaults to 500.
        reference_compute (func): function to compute a reference forecast skill with.
                        Default: :py:func:`climpred.prediction.compute_persistence`.
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        results: (xr.Dataset): bootstrapped results for the three different kinds of
                               predictions:

            - `init` for the initialized hindcast `hind` and describes skill due to
             initialization and external forcing
            - `uninit` for the uninitialized historical `hist` and approximates skill
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
        * climpred.prediction.compute_hindcast

    Example:
        >>> hind = climpred.tutorial.load_dataset('CESM-DP-SST')['SST']
        >>> hist = climpred.tutorial.load_dataset('CESM-LE')['SST']
        >>> obs = load_dataset('ERSST')['SST']
        >>> bootstrapped_skill = climpred.bootstrap.bootstrap_hindcast(hind, hist, obs)
        >>> bootstrapped_skill.coords
        Coordinates:
          * lead     (lead) int64 1 2 3 4 5 6 7 8 9 10
          * kind     (kind) object 'init' 'pers' 'uninit'
          * results  (results) <U7 'skill' 'p' 'low_ci' 'high_ci'

    """
    # Check that init is int, cftime, or datetime; convert ints or cftime to datetime.
    hind = convert_time_index(hind, 'init', 'hind[init]')
    hist = convert_time_index(hist, 'time', 'uninitialized[time]')
    verif = convert_time_index(verif, 'time', 'verif[time]')
    # Put this after `convert_time_index` since it assigns 'years' attribute if the
    # `init` dimension is a `float` or `int`.
    has_valid_lead_units(hind)

    return bootstrap_compute(
        hind,
        verif,
        hist=hist,
        alignment=alignment,
        metric=metric,
        comparison=comparison,
        dim=dim,
        resample_dim=resample_dim,
        sig=sig,
        bootstrap=bootstrap,
        pers_sig=pers_sig,
        compute=compute_hindcast,
        resample_uninit=bootstrap_uninitialized_ensemble,
        reference_compute=reference_compute,
        **metric_kwargs,
    )


def bootstrap_perfect_model(
    ds,
    control,
    metric='pearson_r',
    comparison='m2e',
    dim=None,
    resample_dim='member',
    sig=95,
    bootstrap=500,
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
        resample_dim (str or list): dimension to resample from. default: 'member'.

            - 'member': select a different set of members from hind
            - 'init': select a different set of initializations from hind

        sig (int): Significance level for uninitialized and
                   initialized skill. Defaults to 95.
        pers_sig (int): Significance level for persistence skill confidence levels.
                        Defaults to sig.
        bootstrap (int): number of resampling iterations (bootstrap
                         with replacement). Defaults to 500.
        reference_compute (func): function to compute a reference forecast skill with.
                        Default: :py:func:`climpred.prediction.compute_persistence`.
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        results: (xr.Dataset): bootstrapped results for the three different kinds of
                               predictions:

            - `init` for the initialized hindcast `hind` and describes skill due to
             initialization and external forcing
            - `uninit` for the uninitialized historical `hist` and approximates skill
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
          * kind     (kind) object 'init' 'pers' 'uninit'
          * results  (results) <U7 'skill' 'p' 'low_ci' 'high_ci'
    """

    if dim is None:
        dim = ['init', 'member']
    return bootstrap_compute(
        ds,
        control,
        hist=None,
        metric=metric,
        comparison=comparison,
        dim=dim,
        resample_dim=resample_dim,
        sig=sig,
        bootstrap=bootstrap,
        pers_sig=pers_sig,
        compute=compute_perfect_model,
        resample_uninit=bootstrap_uninit_pm_ensemble_from_control,
        reference_compute=reference_compute,
        **metric_kwargs,
    )
