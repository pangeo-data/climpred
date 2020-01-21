import inspect

import numpy as np
import xarray as xr
from tqdm.auto import tqdm

from .checks import has_dims, has_valid_lead_units
from .constants import ALL_COMPARISONS, ALL_METRICS, COMPARISON_ALIASES, METRIC_ALIASES
from .prediction import compute_hindcast, compute_perfect_model, compute_persistence
from .stats import dpp, varweighted_mean_period
from .utils import (
    assign_attrs,
    convert_time_index,
    get_comparison_class,
    get_lead_cftime_shift_args,
    get_metric_class,
    shift_cftime_singular,
)


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
    # Loads into memory if a dask array.
    ds = ds.compute()
    ds_ci = ds.quantile(q=[ci_low, ci_high], dim=dim)
    return ds_ci


def _pvalue_from_distributions(simple_fct, init, metric='pearson_r'):
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
    return uninit_hind


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
            (isel_years(control, start, length) for start in startlist), 'member'
        )

    return xr.concat((create_pseudo_members(control) for _ in range(nens)), 'init')


def _bootstrap_func(
    func, ds, resample_dim, sig=95, bootstrap=500, *func_args, **func_kwargs
):
    """Sig% threshold of function based on resampling with replacement.

    Reference:
    * Mason, S. J., and G. M. Mimmack. “The Use of Bootstrap Confidence
     Intervals for the Correlation Coefficient in Climatology.” Theoretical and
      Applied Climatology 45, no. 4 (December 1, 1992): 229–33.
      https://doi.org/10/b6fnsv.

    Args:
        func (function): function to be bootstrapped.
        ds (xr.object): first input argument of func.
        resample_dim (str): dimension to resample from.
        sig (int,float,list): significance levels to return. Defaults to 95.
        bootstrap (int): number of resample iterations. Defaults to 500.
        *func_args (type): `*func_args`.
        **func_kwargs (type): `**func_kwargs`.

    Returns:
        sig_level: bootstrapped significance levels with
                   dimensions of ds and len(sig) if sig is list
    """
    if isinstance(sig, list):
        psig = [i / 100 for i in sig]
    else:
        psig = sig / 100
    bootstraped_results = []
    resample_dim_values = ds[resample_dim].values
    for _ in range(bootstrap):
        smp_resample_dim = np.random.choice(
            resample_dim_values, len(resample_dim_values)
        )
        smp_ds = ds.sel({resample_dim: smp_resample_dim})
        smp_ds[resample_dim] = resample_dim_values
        bootstraped_results.append(func(smp_ds, *func_args, **func_kwargs))
    sig_level = xr.concat(bootstraped_results, 'bootstrap').quantile(psig, 'bootstrap')
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
    """Calc the variance-weighted mean period significance levels from re-sampled
    dataset.

    See also:
        * climpred.bootstrap._bootstrap_func
        * climpred.stats.varweighted_mean_period
    """
    return _bootstrap_func(
        varweighted_mean_period, control, time_dim, sig=sig, bootstrap=bootstrap
    )


def bootstrap_compute(
    hind,
    verif,
    hist=None,
    metric='pearson_r',
    comparison='m2e',
    dim='init',
    sig=95,
    bootstrap=500,
    pers_sig=None,
    compute=compute_hindcast,
    resample_uninit=bootstrap_uninitialized_ensemble,
    **metric_kwargs,
):
    """Bootstrap compute with replacement.

    Args:
        hind (xr.Dataset): prediction ensemble.
        verif (xr.Dataset): Verification data.
        hist (xr.Dataset): historical/uninitialized simulation.
        metric (str): `metric`. Defaults to 'pearson_r'.
        comparison (str): `comparison`. Defaults to 'm2e'.
        dim (str or list): dimension to apply metric over. default: 'init'
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
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        results: (xr.Dataset): bootstrapped results
            * init_ci (xr.Dataset): confidence levels of init_skill
            * uninit_ci (xr.Dataset): confidence levels of uninit_skill
            * p_uninit_over_init (xr.Dataset): p value of the hypothesis
                                               that the difference of
                                               skill between the
                                               initialized and uninitialized
                                               simulations is smaller or
                                               equal to zero based on
                                               bootstrapping with
                                               replacement.
                                               Defaults to None.
            * pers_ci (xr.Dataset): confidence levels of pers_skill
            * p_pers_over_init (xr.Dataset): p value of the hypothesis
                                             that the difference of
                                             skill between the
                                             initialized and persistence
                                             simulations is smaller or
                                             equal to zero based on
                                             bootstrapping with
                                             replacement.
                                             Defaults to None.

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

    # which dim should be resampled: member or init
    if dim == 'member' and 'member' in hind.dims:
        members = hind.member.values
        to_be_shuffled = members
        shuffle_dim = 'member'
    elif 'init' in dim and 'init' in hind.dims:
        # also allows ['init','member']
        inits = hind.init.values
        to_be_shuffled = inits
        shuffle_dim = 'init'
    else:
        raise ValueError('Shuffle either `member` or `init`; not', dim)
    # resample with replacement
    # DoTo: parallelize loop
    for _ in tqdm(range(bootstrap), desc='bootstrapping iteration'):
        smp = np.random.choice(to_be_shuffled, len(to_be_shuffled))
        smp_hind = hind.sel({shuffle_dim: smp})
        if shuffle_dim == 'member':
            smp_hind['member'] = np.arange(1, 1 + smp_hind.member.size)
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
            (shuffle_dim == 'init')
            and metric.probabilistic
            and ('init' in init_skill.coords)
        ):
            init_skill['init'] = inits
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
                compute_persistence(smp_hind, verif, metric=metric, **metric_kwargs)
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
    p_pers_over_init = _pvalue_from_distributions(pers, init, metric)

    # calc mean skill without any resampling
    init_skill = compute(
        hind, verif, metric=metric, comparison=comparison, dim=dim, **metric_kwargs
    )
    if 'init' in init_skill:
        init_skill = init_skill.mean('init')
    # remove useless member = 0 coords after m2c
    if 'member' in init_skill.coords and init_skill.member.size == 1:
        del init_skill['member']
    # uninit skill as mean resampled uninit skill
    uninit_skill = uninit.mean('bootstrap')
    if not metric.probabilistic:
        pers_skill = compute_persistence(hind, verif, metric=metric, **metric_kwargs)
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
        'p': 'probability that initialized forecast performs \
                          better than verification data',
    }
    metadata_dict.update(metric_kwargs)
    results = assign_attrs(
        results,
        hind,
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
    metric='pearson_r',
    comparison='e2o',
    dim='init',
    sig=95,
    bootstrap=500,
    pers_sig=None,
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
        dim (str): dimension to apply metric over. default: 'init'
        sig (int): Significance level for uninitialized and
                   initialized skill. Defaults to 95.
        pers_sig (int): Significance level for persistence skill confidence levels.
                        Defaults to sig.
        bootstrap (int): number of resampling iterations (bootstrap
                         with replacement). Defaults to 500.
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        results: (xr.Dataset): bootstrapped results
            * init_ci (xr.Dataset): confidence levels of init_skill
            * uninit_ci (xr.Dataset): confidence levels of uninit_skill
            * p_uninit_over_init (xr.Dataset): p value of the hypothesis
                                               that the difference of
                                               skill between the
                                               initialized and uninitialized
                                               simulations is smaller or
                                               equal to zero based on
                                               bootstrapping with
                                               replacement.
                                               Defaults to None.
            * pers_ci (xr.Dataset): confidence levels of pers_skill
            * p_pers_over_init (xr.Dataset): p value of the hypothesis
                                             that the difference of
                                             skill between the
                                             initialized and persistence
                                             simulations is smaller or
                                             equal to zero based on
                                             bootstrapping with
                                             replacement.
                                             Defaults to None.

    Reference:
        * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
          Gonzalez, V. Kharin, et al. “A Verification Framework for
          Interannual-to-Decadal Predictions Experiments.” Climate
          Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
          https://doi.org/10/f4jjvf.

    See also:
        * climpred.bootstrap.bootstrap_compute
        * climpred.prediction.compute_hindcast
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
        metric=metric,
        comparison=comparison,
        dim=dim,
        sig=sig,
        bootstrap=bootstrap,
        pers_sig=pers_sig,
        compute=compute_hindcast,
        resample_uninit=bootstrap_uninitialized_ensemble,
        **metric_kwargs,
    )


def bootstrap_perfect_model(
    ds,
    control,
    metric='pearson_r',
    comparison='m2e',
    dim=None,
    sig=95,
    bootstrap=500,
    pers_sig=None,
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
        dim (str): dimension to apply metric over. default: ['init', 'member']
        sig (int): Significance level for uninitialized and
                   initialized skill. Defaults to 95.
        pers_sig (int): Significance level for persistence skill confidence levels.
                        Defaults to sig.
        bootstrap (int): number of resampling iterations (bootstrap
                         with replacement). Defaults to 500.
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        results: (xr.Dataset): bootstrapped results
            * init_ci (xr.Dataset): confidence levels of init_skill
            * uninit_ci (xr.Dataset): confidence levels of uninit_skill
            * p_uninit_over_init (xr.Dataset): p value of the hypothesis
                                               that the difference of
                                               skill between the
                                               initialized and uninitialized
                                               simulations is smaller or
                                               equal to zero based on
                                               bootstrapping with
                                               replacement.
                                               Defaults to None.
            * pers_ci (xr.Dataset): confidence levels of pers_skill
            * p_pers_over_init (xr.Dataset): p value of the hypothesis
                                             that the difference of
                                             skill between the
                                             initialized and persistence
                                             simulations is smaller or
                                             equal to zero based on
                                             bootstrapping with
                                             replacement.
                                             Defaults to None.

    Reference:
        * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
          Gonzalez, V. Kharin, et al. “A Verification Framework for
          Interannual-to-Decadal Predictions Experiments.” Climate
          Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
          https://doi.org/10/f4jjvf.

    See also:
        * climpred.bootstrap.bootstrap_compute
        * climpred.prediction.compute_perfect_model
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
        sig=sig,
        bootstrap=bootstrap,
        pers_sig=pers_sig,
        compute=compute_perfect_model,
        resample_uninit=bootstrap_uninit_pm_ensemble_from_control,
        **metric_kwargs,
    )
