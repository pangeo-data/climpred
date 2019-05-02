import numpy as np
import xarray as xr

from .prediction import compute_perfect_model, compute_persistence_pm
from .stats import DPP, xr_varweighted_mean_period


def pseudo_ens(ds, control):
    """
    Create a pseudo-ensemble from control run.

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
        return xr.concat([isel_years(control, start, length) for start in
                          startlist], 'member')

    return xr.concat([create_pseudo_members(control) for _ in range(nens)],
                     'init')


def DPP_threshold(control, sig=95, bootstrap=500, **dpp_kwargs):
    """Calc DPP from re-sampled dataset.

    Reference:
    * Feng, X., T. DelSole, and P. Houser. “Bootstrap Estimated Seasonal
        Potential Predictability of Global Temperature and Precipitation.”
        Geophysical Research Letters 38, no. 7 (2011).
        https://doi.org/10/ft272w.

    """
    bootstraped_results = []
    time = control.time.values
    for _ in range(bootstrap):
        smp_time = np.random.choice(time, len(time))
        smp_control = control.sel(time=smp_time)
        smp_control['time'] = time
        bootstraped_results.append(DPP(smp_control, **dpp_kwargs))
    threshold = xr.concat(bootstraped_results, 'bootstrap').quantile(
        sig / 100, 'bootstrap')
    return threshold


def xr_varweighted_mean_period_threshold(control,
                                         sig=95,
                                         bootstrap=500,
                                         **vwmp_kwargs):
    """Calc vwmp from re-sampled dataset.

    """
    bootstraped_results = []
    time = control.time.values
    for _ in range(bootstrap):
        smp_time = np.random.choice(time, len(time))
        smp_control = control.sel(time=smp_time)
        smp_control['time'] = time
        bootstraped_results.append(
            xr_varweighted_mean_period(smp_control, **vwmp_kwargs))
    threshold = xr.concat(bootstraped_results, 'bootstrap').quantile(
        sig / 100, 'bootstrap')
    return threshold


def bootstrap_perfect_model(ds,
                            control,
                            metric='pearson_r',
                            comparison='m2e',
                            sig=95,
                            bootstrap=500,
                            compute_uninitialized_skill=True,
                            compute_persistence_skill=True,
                            pers_sig=None,
                            compute_ci=True,
                            nlags=None,
                            running=None,
                            reference_period='MK'):
    """Bootstrap perfect-model ensemble simulations with replacement.

    Reference:
      * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
            Gonzalez, V. Kharin, et al. “A Verification Framework for
            Interannual-to-Decadal Predictions Experiments.” Climate
            Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
            https://doi.org/10/f4jjvf.

    Args:
        ds (xr.Dataset): prediction ensemble.
        control (xr.Dataset): control simulation.
        metric (str): `metric`. Defaults to 'pearson_r'.
        comparison (str): `comparison`. Defaults to 'm2e'.
        sig (int): Significance level for uninitialized and
                   initialized skill. Defaults to 95.
        bootstrap (int): number of resampling iterations (bootstrap
                         with replacement). Defaults to 500.
        compute_uninitialized_skill (bool): Defaults to True.
        compute_persistence_skill (bool): Defaults to True.
        nlags (type): number of lags persistence forecast skill.
                      Defaults to ds.lead.size.

    Returns:
        init_ci (xr.Dataset): confidence levels of init_skill
        uninit_ci (xr.Dataset): confidence levels of uninit_skill
        p_uninit_over_init (xr.Dataset): p-value of the hypothesis
                                         that the difference of
                                         skill between the
                                         initialized and uninitialized
                                         simulations is smaller or
                                         equal to zero based on
                                         bootstrapping with
                                         replacement.
                                         Defaults to None.
        pers_ci (xr.Dataset): confidence levels of pers_skill
        p_pers_over_init (xr.Dataset): p-value of the hypothesis
                                       that the difference of
                                       skill between the
                                       initialized and persistence
                                       simulations is smaller or
                                       equal to zero based on
                                       bootstrapping with
                                       replacement.
                                       Defaults to None.

    """
    if pers_sig is None:
        pers_sig = sig

    # TODO: calc normalized persistence forecasts
    if metric not in ['pearson_r', 'rmse', 'mse', 'mae']:
        compute_persistence_skill = False

    if nlags is None:
        nlags = ds.lead.size
    p = (100 - sig) / 100  # 0.05
    ci_low = p / 2  # 0.025
    ci_high = 1 - p / 2  # 0.975
    p_pers = (100 - pers_sig) / 100  # 0.5
    ci_low_pers = p_pers / 2
    ci_high_pers = 1 - p_pers / 2

    inits = ds.init.values
    init = []
    uninit = []
    pers = []
    # resample with replacement
    # DoTo: parallelize loop
    for _ in range(bootstrap):
        smp = np.random.choice(inits, len(inits))
        smp_ds = ds.sel(init=smp)
        # compute init skill
        init.append(
            compute_perfect_model(
                smp_ds,
                control,
                metric=metric,
                comparison=comparison,
                running=running,
                reference_period=reference_period))
        if compute_uninitialized_skill:
            # generate uninitialized ensemble from control
            uninit_ds = pseudo_ens(ds, control)  # .isel(lead=0)
            # compute uninit skill
            uninit.append(
                compute_perfect_model(
                    uninit_ds,
                    control,
                    metric=metric,
                    comparison=comparison,
                    running=running,
                    reference_period=reference_period))
        # compute persistence skill
        if compute_persistence_skill:
            pers.append(
                compute_persistence_pm(
                    smp_ds, control, nlags=nlags, dim='time', metric=metric))
    init = xr.concat(init, dim='bootstrap')
    if compute_uninitialized_skill:
        uninit = xr.concat(uninit, dim='bootstrap')
    if compute_persistence_skill:
        pers = xr.concat(pers, dim='bootstrap')

    def _distribution_to_ci(ds, ci_low, ci_high, dim='bootstrap'):
        try:
            if len(ds.chunks) >= 1:
                ds = ds.compute()
        except:
            pass
        ds_ci = ds.quantile(q=[ci_low, ci_high], dim=dim)
        return ds_ci

    if compute_ci:
        init_ci = _distribution_to_ci(init, ci_low, ci_high)
        if compute_uninitialized_skill:
            uninit_ci = _distribution_to_ci(uninit, ci_low, ci_high)
        if compute_persistence_skill:
            pers_ci = _distribution_to_ci(pers, ci_low_pers, ci_high_pers)

    else:
        init_ci = None
        pers_ci = None
        uninit_ci = None

    def _pvalue_from_distributions(simple_fct, init, metric=metric):
        """Get probability that skill of simple_fct is larger than
        init skill."""
        pv = ((simple_fct - init) > 0).sum('bootstrap') / init.bootstrap.size
        if metric not in ['pearson_r', 'ppp']:  # positively oriented metrics
            pv = 1 - pv
        return pv

    if compute_uninitialized_skill:
        p_uninit_over_init = _pvalue_from_distributions(uninit, init)
    else:
        p_uninit_over_init, uninit_ci = None, None

    if compute_persistence_skill:
        p_pers_over_init = _pvalue_from_distributions(pers, init)

    else:
        p_pers_over_init, pers_ci = None, None

    # calc skill
    init_skill = compute_perfect_model(
        ds, control, metric=metric, comparison=comparison, running=running,
        reference_period=reference_period)
    uninit_skill = uninit.mean('bootstrap')
    pers_skill = compute_persistence_pm(
        ds, control, nlags=nlags, dim='time', metric=metric)

    # somehow there may be a member dim, which lets concat crash
    if 'member' in init_skill:
        del init_skill['member']
    # wrap results together in one dataarray
    skill = xr.concat([init_skill.squeeze(), uninit_skill, pers_skill], 'i')
    skill['i'] = ['init', 'uninit', 'pers']

    # probability that i beats init
    p = xr.concat([p_uninit_over_init, p_pers_over_init], 'i')
    p['i'] = ['uninit', 'pers']

    # ci for each skill
    ci = xr.concat([init_ci, uninit_ci, pers_ci],
                   'i').rename({'quantile': 'results'})
    ci['i'] = ['init', 'uninit', 'pers']

    results = xr.concat([skill, p], 'results')
    results['results'] = ['skill', 'p']
    # (RXB) Drop any spurious coordinates that came along with results due to
    # input ds. You can't concatenate if results and ci don't exactly match
    # in coordinates. Maybe we can create a decorator to check this before
    # anytime we merge.
    dims = list(results.dims)
    droplist = [coord for coord in results.coords if coord not in dims]
    results = xr.concat([results.drop(droplist), ci], 'results')
    return results
