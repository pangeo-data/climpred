import xarray as xr

from .alignment import return_inits_and_verif_dates
from .checks import has_valid_lead_units, is_xarray
from .comparisons import (
    ALL_COMPARISONS,
    COMPARISON_ALIASES,
    HINDCAST_COMPARISONS,
    PM_COMPARISONS,
    __e2c,
)
from .constants import CLIMPRED_DIMS, M2M_MEMBER_DIM
from .metrics import (
    ALL_METRICS,
    DETERMINISTIC_HINDCAST_METRICS,
    METRIC_ALIASES,
    PM_METRICS,
    _rename_dim,
)
from .utils import (
    assign_attrs,
    convert_time_index,
    get_comparison_class,
    get_lead_cftime_shift_args,
    get_metric_class,
    shift_cftime_index,
)


def persistence(verif, inits, verif_dates, lead):
    """Persistence forecast prescribes values from initialization into the future."""
    #print('verif_dates',verif_dates[lead],'inits',inits[lead].init.to_index())
    lforecast = verif.where(verif.time.isin(inits[lead]), drop=True)
    #lforecast = lforecast.assign_coords(
    #    init=(("lead", "time"), inits[lead].expand_dims("lead"))
    #)
    lverif = verif.sel(time=verif_dates[lead])
    #print(f'inside persistence lead {lead} forecast',lforecast.coords)
    #print(f'inside persistence lead {lead} verif',lverif.coords)
    # set time equal to compute lagged metric
    # was before
    #lforecast["time"] = lverif.time
    # if dim is member
    lverif["time"] = lforecast.time
    return lforecast, lverif


def climatology(verif, inits, verif_dates, lead):
    """Climatology forecasts climatological mean in the future."""
    climatology_day = verif.groupby("time.dayofyear").mean()
    # enlarge times to get climatology_forecast times
    # this prevents errors if verification.time and hindcast.init are too much apart
    verif_hind_union = xr.DataArray(
        verif.time.to_index().union(inits[lead]["init"].to_index()), dims="time"
    )

    climatology_forecast = climatology_day.sel(
        dayofyear=verif_hind_union.time.dt.dayofyear, method="nearest"
    ).drop("dayofyear")

    lforecast = climatology_forecast.where(
        climatology_forecast.time.isin(inits[lead]), drop=True
    )
    lverif = verif.sel(time=verif_dates[lead])
    # was before
    #lforecast["time"] = lverif.time
    lverif['time']=lforecast['time']
    return lforecast, lverif


def uninitialized(hist, verif, inits, verif_dates, lead, alignment):
    """Uninitialized forecast uses a simulation without any initialization (assimilation/nudging). Also called historical in some communities."""
    print('alignemnt',alignment)
    if alignment=='same_verif':
        #print('use same_verifs')
        #lforecast = hist.sel(time=verif_dates[lead])
        #lverif = verif.sel(time=verif_dates[lead])
        lforecast = (hist.sel(time=verif_dates[lead]).assign_coords(init=(("lead", "time"), inits[lead].expand_dims("lead")))
        .assign_coords(lead=[lead])
    )
        lverif = (verif.sel(time=verif_dates[lead]).assign_coords(init=(("lead", "time"), inits[lead].expand_dims("lead")))
        .assign_coords(lead=[lead])
    )
    elif alignment in ['same_init','maximize']:
        #print('use same_inits')
        #print('hist',hist.coords)
        #print('verif',verif.coords)
        #print('inits[lead]',inits[lead])
        sel_time = verif_dates[lead].intersection(hist.time.to_index())
        #print('sel_time',sel_time)
        new_inits = inits[lead].expand_dims("lead").assign_coords(lead=[lead])
        #print('hist',hist.coords)
        #print('new_inits',new_inits.squeeze().to_index())
        lforecast = hist.sel(time=sel_time).assign_coords(init=(("lead", "time"), new_inits)).assign_coords(lead=[lead])
        #print('lforecast.coords',lforecast.coords)
        lverif = verif.sel(time=sel_time).assign_coords(init=(("lead", "time"), new_inits)).assign_coords(lead=[lead])
    #elif alignment=='maximize' and False:
    #    verif_dates_inits_intersection = verif_dates[lead].intersection(inits[lead].to_index())
    #    lforecast = hist.sel(time=verif_dates_inits_intersection)
    #    lverif = verif.sel(time=verif_dates_inits_intersection)
    #lverif['time']=lforecast['time'] #new
    return lforecast, lverif


# needed for PerfectModelEnsemble.verify(reference=...) and PredictionEnsemble.bootstrap
# TODO: should be refactored for better non-functional use within verify and bootstrap


def _adapt_member_for_reference_forecast(lforecast, lverif, metric, comparison, dim):
    """Maybe drop member from dim or add single-member dimension. Used in reference forecasts: climatology, uninitialized, persistence."""
    # persistence or climatology forecasts wont have member dimension, create if required
    # some metrics dont allow member dimension, remove and try mean
    # delete member from dim if needed
    if "member" in dim:
        if (
            "member" in lforecast.dims
            and "member" not in lverif.dims
            and not metric.requires_member_dim
        ):
            dim = dim.copy()
            dim.remove("member")
        elif "member" not in lforecast.dims and "member" not in lverif.dims:
            dim = dim.copy()
            dim.remove("member")
    # for probabilistic metrics requiring member dim, add single-member dimension
    if metric.requires_member_dim:
        if "member" not in lforecast.dims:
            lforecast = lforecast.expand_dims("member")  # add fake member dim
            if "member" not in dim:
                dim = dim.copy()
                dim.append("member")
        assert "member" in lforecast.dims and "member" not in lverif.dims
    else:  # member not required by metric and not in dim but present in forecast
        if "member" in lforecast.dims and "member" not in dim:
            lforecast = lforecast.mean("member")
    # multi member comparisons expect member dim
    if (
        comparison.name in ["m2o", "m2m", "m2c", "m2e"]
        and "member" not in lforecast.dims
        and metric.requires_member_dim
    ):  # not triggered in any test
        lforecast = lforecast.expand_dims("member")  # add fake member dim
    return lforecast, dim


def compute_climatology(
    hind,
    verif=None,
    metric="pearson_r",
    comparison="m2e",
    alignment="same_init",
    dim="init",
    **metric_kwargs,
):
    """Computes the skill of a climatology forecast.

    Args:
        hind (xarray object): The initialized ensemble.
        verif (xarray object): control data, not needed
        metric (str): Metric name to apply at each lag for the persistence computation.
            Default: 'pearson_r'
        dim (str or list of str): dimension to apply metric over
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        clim (xarray object): Results of climatology forecast with the input metric
            applied.
    """
    if isinstance(dim, str):
        dim = [dim]
    # Check that init is int, cftime, or datetime; convert ints or cftime to datetime.
    hind = convert_time_index(hind, "init", "hind[init]")
    verif = convert_time_index(verif, "time", "verif[time]")
    # Put this after `convert_time_index` since it assigns 'years' attribute if the
    # `init` dimension is a `float` or `int`.
    has_valid_lead_units(hind)

    # get metric/comparison function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)
    comparison = COMPARISON_ALIASES.get(comparison, comparison)

    comparison = get_comparison_class(comparison, ALL_COMPARISONS)
    metric = get_metric_class(metric, ALL_METRICS)

    if "iteration" in hind.dims:
        hind = hind.isel(iteration=0, drop=True)

    kind = "hindcast" if comparison.hindcast else "perfect"

    inits, verif_dates = return_inits_and_verif_dates(
        hind, verif, alignment=alignment, reference="climatology"
    )

    if kind == "perfect":
        forecast, verif = comparison.function(hind, metric=metric)
        climatology_day = verif.groupby("init.dayofyear").mean()
    else:
        forecast, verif = comparison.function(hind, verif, metric=metric)
        climatology_day = verif.groupby("time.dayofyear").mean()

    climatology_day_forecast = climatology_day.sel(
        dayofyear=forecast["time"].dt.dayofyear, method="nearest"
    ).drop("dayofyear")


    # ensure overlap
    if kind == "hindcast" and False:
        climatology_day_forecast = (
            climatology_day_forecast.drop("time")
            .rename({"init": "time"})
            .sel(time=verif.time)
        )
    if kind == 'hindcast':
        if alignment in ['same_init']:
            climatology_day_forecast = climatology_day_forecast.sel(init=inits[1])
            verif = verif.sel(time=hind.time, method="nearest").assign_coords(
            time=hind.time).sel(init=inits[1]) # verif with init lead
        elif alignment=='maximize':
            climatology_day_forecast = climatology_day_forecast.sel(init=inits[1].drop('lead'))
            verif = verif.sel(time=hind.time, method="nearest").assign_coords(
            time=hind.time).sel(init=inits[1].drop('lead'))
        elif alignment=='same_verif':
            climatology_day_forecast = init_to_time_dim(climatology_day_forecast)
        #assert 'time' in climatology_day_forecast.dims
        if alignment=='same_verif':
            time_intersection = verif_dates[1]
        elif alignment in ['maximize'] and False:
            time_intersection = climatology_day_forecast.time.to_index().intersection(verif.time.to_index())
            time_intersection = time_intersection.intersection(verif_dates[1])
        if alignment == 'same_verif':
            climatology_day_forecast = climatology_day_forecast.sel(time=time_intersection)

        if 'time' in climatology_day_forecast.dims and False:
            if climatology_day_forecast.isnull().any('time') and 'init' in dim: ## TODO: investigate why needed
                climatology_day_forecast = climatology_day_forecast.ffill('time').bfill('time')

        if alignment =='same_verif':
            verif=verif.sel(time=time_intersection)
            verif=verif.sel(time=climatology_day_forecast.time, method='nearest')
            climatology_day_forecast = climatology_day_forecast.sel(time=time_intersection,method='nearest')
        elif alignment == 'same_inits':
            climatology_day_forecast = climatology_day_forecast.sel(init=inits[1])
            verif=verif.sel(init=inits[1])
        elif alignment=='maximize':
            print('verif',verif.coords,verif.dims)
            verif = xr.concat([verif.sel(lead=lead).sel(init=inits[lead]) for lead in hind.lead.values],'lead')
            climatology_day_forecast = xr.concat([climatology_day_forecast.sel(lead=lead).sel(init=inits[lead]) for lead in hind.lead.values],'lead')

    #print('climatology_day_forecast',climatology_day_forecast.sel(lead=[1,2]).SST)
    #print('verif',verif.time)


    dim = _rename_dim(dim, climatology_day_forecast, verif)
    if metric.normalize:
        metric_kwargs["comparison"] = __e2c

    climatology_day_forecast, dim = _adapt_member_for_reference_forecast(
        climatology_day_forecast, verif, metric, comparison, dim
    )

    if ( # why is this needed? # TODO:
        "lead" in climatology_day_forecast.dims and "lead" not in verif.dims
    ):  # issue: https://github.com/pangeo-data/climpred/issues/528
        verif = (
            verif.expand_dims("lead")
            .isel(lead=[0] * climatology_day_forecast.lead.size)
            .assign_coords(lead=climatology_day_forecast.lead)
        )
    if 'SST' in verif.data_vars:
        print('climatology_day_forecast',climatology_day_forecast.SST)
        print('verif',verif.SST)
    #climatology_day_forecast=climatology_day_forecast.fillna(0.)
    clim_skill = metric.function(
        climatology_day_forecast, verif, dim=dim, **metric_kwargs
    )
    #print('clim_skill',clim_skill.dims,clim_skill.coords,clim_skill.SST)
    if "time" in clim_skill.dims and 'init' in clim_skill.coords:
        #clim_skill = clim_skill.swap_dims({'time':'init'})
        #print('clim_skill',clim_skill.coords,clim_skill.dims,clim_skill.SST)
        try:
            clim_skill = time_to_init_dim_2(clim_skill)
        except:
            clim_skill = time_to_init_dim(clim_skill)
    if 'time' in clim_skill.dims and 'init' not in clim_skill.coords:
        clim_skill = clim_skill.rename({'time':'init'}) # TODO cehck required?
    if M2M_MEMBER_DIM in clim_skill.dims:
        clim_skill = clim_skill.mean(M2M_MEMBER_DIM)
    print('clim_skill',clim_skill.dims,clim_skill.coords)
    return clim_skill


@is_xarray([0, 1])
def compute_persistence(
    hind,
    verif,
    metric="pearson_r",
    alignment="same_verif",
    dim="init",
    comparison="m2o",
    **metric_kwargs,
):
    """Computes the skill of a persistence forecast from a simulation.

    Args:
        hind (xarray object): The initialized ensemble.
        verif (xarray object): Verification data.
        metric (str): Metric name to apply at each lag for the persistence computation.
            Default: 'pearson_r'
        alignment (str): which inits or verification times should be aligned?
            - maximize/None: maximize the degrees of freedom by slicing ``hind`` and
            ``verif`` to a common time frame at each lead.
            - same_inits: slice to a common init frame prior to computing
            metric. This philosophy follows the thought that each lead should be based
            on the same set of initializations.
            - same_verif: slice to a common/consistent verification time frame prior to
            computing metric. This philosophy follows the thought that each lead
            should be based on the same set of verification dates.
        dim (str or list of str): dimension to apply metric over.
        ** metric_kwargs (dict): additional keywords to be passed to metric
            (see the arguments required for a given metric in :ref:`Metrics`).

    Returns:
        pers (xarray object): Results of persistence forecast with the input metric
            applied.

    Reference:
        * Chapter 8 (Short-Term Climate Prediction) in Van den Dool, Huug.
          Empirical methods in short-term climate prediction.
          Oxford University Press, 2007.

    """
    if isinstance(dim, str):
        dim = [dim]
    # Check that init is int, cftime, or datetime; convert ints or cftime to datetime.
    hind = convert_time_index(hind, "init", "hind[init]")
    verif = convert_time_index(verif, "time", "verif[time]")
    # Put this after `convert_time_index` since it assigns 'years' attribute if the
    # `init` dimension is a `float` or `int`.
    has_valid_lead_units(hind)

    # get metric/comparison function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)
    comparison = COMPARISON_ALIASES.get(comparison, comparison)
    comparison = get_comparison_class(comparison, ALL_COMPARISONS)
    metric = get_metric_class(metric, ALL_METRICS)

    # If lead 0, need to make modifications to get proper persistence, since persistence
    # at lead 0 is == 1.
    if [0] in hind.lead.values:
        hind = hind.copy()
        with xr.set_options(keep_attrs=True):  # keeps lead.attrs['units']
            hind["lead"] = hind["lead"] + 1
        n, freq = get_lead_cftime_shift_args(hind.lead.attrs["units"], 1)
        # Shift backwards shift for lead zero.
        hind["init"] = shift_cftime_index(hind, "init", -1 * n, freq)

    inits, verif_dates = return_inits_and_verif_dates(
        hind, verif, alignment=alignment, reference="persistence"
    )
    #print('inits',inits[1], '\n verif_dates',verif_dates[1])
    #print('inits',inits[2], '\n verif_dates',verif_dates[2])

    if metric.normalize:
        metric_kwargs["comparison"] = __e2c
    dim = _rename_dim(dim, hind, verif)

    plag = []
    for i in hind.lead.values:
        lforecast = (
            verif.sel(time=inits[i], method="nearest")
            .swap_dims({"init": "time"})
            .drop("init")
        )
        lverif = verif.sel(time=verif_dates[i])
        lverif["time"] = lforecast[
            "time"
        ]  # important to overwrite time here for xr.concat() alignment
        #if 'member' in dim and 'init' in lforecast.dims and 'init' in lverif.dims:
        #    lforecast['init']=lverif['init']
        lforecast, dim = _adapt_member_for_reference_forecast(
            lforecast, lverif, metric, comparison, dim
        )
        dim = _rename_dim(dim, lforecast, lverif)
        # comparison expected for normalized metrics
        plag.append(metric.function(lforecast, lverif, dim=dim, **metric_kwargs))
    pers_skill = xr.concat(plag, "lead")
    pers_skill['lead']=hind.lead
    print('pers_skill',pers_skill.dims,pers_skill.coords)
    if 'time' in pers_skill.dims and 'init' not in pers_skill:
        pers_skill = pers_skill.rename({'time':'init'})
    if "time" in pers_skill.dims and False:
        new_init= pers_skill.coords['init'].all('lead').rename({"time":'init'}).init.to_index()#.dropna('time')
        #print('new_init',new_init)
        pers_skill = pers_skill.drop('init').rename({'time':'init'})#.swap_dims({"time": "init"})#.dropna('init')
        pers_skill=pers_skill.assign_coords(init=new_init)
        #print('new',pers_skill.dims,pers_skill.coords)
        #import pandas as pd
        #assert isinstance(pers_skill.init, pd.RangeIndex)
    if M2M_MEMBER_DIM in pers_skill.dims:
        pers_skill = pers_skill.mean(M2M_MEMBER_DIM)
    if 'member' not in pers_skill.dims and 'member' in pers_skill.coords:
        del pers_skill.coords['member']
    return pers_skill


@is_xarray([0, 1])
def compute_uninitialized(
    hind,
    uninit,
    verif,
    metric="pearson_r",
    comparison="e2o",
    dim="time",
    alignment="same_verif",
    **metric_kwargs,
):
    """Verify an uninitialized ensemble against verification data.

    .. note::
        Based on Decadal Prediction protocol, this should only be computed for the
        first lag and then projected out to any further lags being analyzed.

    Args:
        hind (xarray object): Initialized ensemble.
        uninit (xarray object): Uninitialized ensemble.
        verif (xarray object): Verification data with some temporal overlap with the
            uninitialized ensemble.
        metric (str):
            Metric used in comparing the uninitialized ensemble with the verification
            data.
        comparison (str):
            How to compare the uninitialized ensemble to the verification data:
                * e2o : ensemble mean to verification data (Default)
                * m2o : each member to the verification data
        dim (str or list of str): dimension to apply metric over.
        alignment (str): which inits or verification times should be aligned?
            - maximize/None: maximize the degrees of freedom by slicing ``hind`` and
            ``verif`` to a common time frame at each lead.
            - same_inits: slice to a common init frame prior to computing
            metric. This philosophy follows the thought that each lead should be based
            on the same set of initializations.
            - same_verif: slice to a common/consistent verification time frame prior to
            computing metric. This philosophy follows the thought that each lead
            should be based on the same set of verification dates.
        ** metric_kwargs (dict): additional keywords to be passed to metric

    Returns:
        u (xarray object): Results from comparison at the first lag.

    """
    if isinstance(dim, str):
        dim = [dim]
    # Check that init is int, cftime, or datetime; convert ints or cftime to datetime.
    hind = convert_time_index(hind, "init", "hind[init]")
    uninit = convert_time_index(uninit, "time", "uninit[time]")
    verif = convert_time_index(verif, "time", "verif[time]")
    has_valid_lead_units(hind)

    # get metric/comparison function name, not the alias
    metric = METRIC_ALIASES.get(metric, metric)
    comparison = COMPARISON_ALIASES.get(comparison, comparison)
    comparison = get_comparison_class(comparison, ALL_COMPARISONS)
    metric = get_metric_class(metric, ALL_METRICS)
    forecast, verif = comparison.function(uninit, verif, metric=metric)

    inits, verif_dates = return_inits_and_verif_dates(
        hind, verif, alignment=alignment, reference="uninitialized", hist=uninit
    )

    if metric.normalize:
        metric_kwargs["comparison"] = comparison

    if (
        "iteration" in forecast.dims and "iteration" not in verif.dims
    ):  # issue: https://github.com/pangeo-data/climpred/issues/528
        verif = (
            verif.expand_dims("iteration")
            .isel(iteration=[0] * forecast.iteration.size)
            .assign_coords(iteration=forecast.iteration)
        )

    plag = []
    # TODO: `same_verifs` does not need to go through the loop, since it's a fixed
    # skill over all leads
    for lead in hind["lead"].values:
        # Ensure that the uninitialized reference has all of the
        # dates for alignment.
        dates = list(set(forecast["time"].values) & set(verif_dates[lead]))
        # select forecast and verification at lead
        if alignment=='same_verif':
            lforecast = forecast.sel(time=dates).assign_coords(init=(("lead", "time"), inits[lead].expand_dims("lead"))).assign_coords(lead=[lead])
        # select verification at lead
            lverif = verif.sel(time=dates).assign_coords(init=(("lead", "time"), inits[lead].expand_dims("lead"))).assign_coords(lead=[lead])
        elif alignment in ['same_init','maximize']:
            # work for maximize but not for same_inits TODO
            #sel_time = inits[lead].to_index().intersection(forecast.time.to_index())
            #lforecast = forecast.sel(time=sel_time)
            #lverif = verif.sel(time=sel_time)
            sel_time = verif_dates[lead].intersection(forecast.time.to_index())
            #print('sel_time',sel_time)
            new_inits = inits[lead].expand_dims("lead").assign_coords(lead=[lead])
            #print('hist',hist.coords)
            #print('new_inits',new_inits.squeeze().to_index())
            lforecast = forecast.sel(time=sel_time).assign_coords(init=(("lead", "time"), new_inits)).assign_coords(lead=[lead])
            #print('lforecast.coords',lforecast.coords)
            lverif = verif.sel(time=sel_time).assign_coords(init=(("lead", "time"), new_inits)).assign_coords(lead=[lead])

        lforecast, dim = _adapt_member_for_reference_forecast(
            lforecast, lverif, metric, comparison, dim
        )
        dim = _rename_dim(dim, lforecast, lverif)
        lforecast["time"] = lverif["time"]
        # comparison expected for normalized metrics
        plag.append(metric.function(lforecast, lverif, dim=dim, **metric_kwargs))
    uninit_skill = xr.concat(plag, "lead")
    uninit_skill["lead"] = hind.lead.values
    #print('uninit_skill',uninit_skill.dims, uninit_skill.coords)
    #if "time" in uninit_skill.dims and 'init' in uninit_skill.coords:
    #    uninit_skill = uninit_skill.swap_dims({'time':'init'})
    if "time" in uninit_skill.dims:
        if len(uninit_skill.time.coords)==1:
            if not uninit_skill.time.to_index().is_monotonic_increasing:
                # TODO: check if used at all
                #print('uninit_skill',uninit_skill.dims, uninit_skill.coords)
                uninit_skill = uninit_skill.sortby(
                    uninit_skill.time
                )
                #print('sorted uninit_skill')
                #print('uninit_skill',uninit_skill.dims, uninit_skill.coords)
    if 'time' in uninit_skill.dims and 'time' in uninit_skill.coords and 'init' in uninit_skill.coords and 'init' not in uninit_skill.dims:
        if len(uninit_skill.init.dims)==2:
            #print('uninit_skill.time',uninit_skill.time)
            #for i in uninit_skill.lead.values:
            #    print('uninit_skill.init lead',i,uninit_skill.init.sel(lead=i))
            uninit_skill = time_to_init_dim_2(uninit_skill)
    if 'time' in uninit_skill and 'time' in uninit_skill.coords and 'init' not in uninit_skill.dims and 'init' not in uninit_skill.coords:
        print('renaming uninit')
        uninit_skill = uninit_skill.rename({'time':'init'})
    if M2M_MEMBER_DIM in uninit_skill.dims:
        uninit_skill = uninit_skill.mean(M2M_MEMBER_DIM)
    #print('uninit_skill',uninit_skill.dims, uninit_skill.coords)
    return uninit_skill


def time_to_init_dim(r):
    return xr.concat(
    [r.sel(lead=i).swap_dims({"time": "init"}) for i in r.lead],
    dim="lead",
        # compat="override",
        # coords="minimal",
    )

def time_to_init_dim_2(r):
    return xr.concat(
    [r.sel(lead=i).swap_dims({"time": "init"}).dropna('init') for i in r.lead],
    dim="lead",
        # compat="override",
        # coords="minimal",
    )

def init_to_time_dim(r):
    return xr.concat(
        [r.sel(lead=i).swap_dims({"init": "time"}) for i in r.lead],
        dim="lead",
        # compat="override",
        # coords="minimal",
    )
