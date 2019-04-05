import numpy as np
import xarray as xr
from .prediction import compute_perfect_model, compute_persistence_pm
from .stats import DPP, xr_varweighted_mean_period


def _pseudo_ens(ds, control):
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
    nens = ds.initialization.size
    nmember = ds.member.size
    length = ds.time.size
    c_start = 0
    c_end = control['time'].size
    time = ds['time']

    def isel_years(control, year_s, m=None, length=length):
        new = control.isel(time=slice(year_s, year_s + length))
        new['time'] = time
        return new

    def create_pseudo_members(control):
        startlist = np.random.randint(c_start, c_end - length - 1, nmember)
        return xr.concat([isel_years(control, start) for start in startlist],
                         'member')

    return xr.concat([create_pseudo_members(control) for _ in range(nens)],
                     'initialization')


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
                            pers_sig=50,
                            bootstrap=500,
                            compute_uninitized_skill=True,
                            compute_persistence_skill=True,
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
        pers_sig (int): Significance level for persistence forecast.
                        Defaults to 50.
        bootstrap (int): number of resampling iterations (bootstrap
                         with replacement). Defaults to 500.
        compute_uninitized_skill (bool): Defaults to True.
        compute_persistence_skill (bool): Defaults to True.
        nlags (type): number of lags persistence forecast skill.
                      Defaults to ds.time.size.

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
    if nlags is None:
        nlags = ds.time.size
    p = (100 - sig) / 100  # 0.05
    ci_low = p / 2  # 0.025
    ci_high = 1 - p / 2  # 0.975
    p_pers = (100 - pers_sig) / 100  # 0.5
    ci_low_pers = p_pers / 2
    ci_high_pers = 1 - p_pers / 2

    inits = ds.initialization.values
    init = []
    uninit = []
    pers = []
    # resample with replacement
    # DoTo: parallelize loop
    for _ in range(bootstrap):
        smp = np.random.choice(inits, len(inits))
        smp_ds = ds.sel(initialization=smp)
        # compute init skill
        init.append(
            compute_perfect_model(
                smp_ds,
                control,
                metric=metric,
                comparison=comparison,
                running=running,
                reference_period=reference_period))
        if compute_uninitized_skill:
            # generate uninitialized ensemble from control
            uninit_ds = _pseudo_ens(ds, control).isel(time=0)
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
    if compute_uninitized_skill:
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
        if compute_uninitized_skill:
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

    if compute_uninitized_skill:
        p_uninit_over_init = _pvalue_from_distributions(uninit, init)
    else:
        p_uninit_over_init, uninit_ci = None, None

    if compute_persistence_skill:
        p_pers_over_init = _pvalue_from_distributions(pers, init)
    else:
        p_pers_over_init, pers_ci = None, None

    return init_ci, uninit_ci, p_uninit_over_init, pers_ci, p_pers_over_init
