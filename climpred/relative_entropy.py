import warnings

import numpy as np
import xarray as xr

try:
    from eofs.xarray import Eof
except ImportError:
    Eof = None


def _relative_entropy_formula(sigma_b, sigma_x, mu_x, mu_b, neofs):
    """
    Compute the relative entropy formula given in Branstator and Teng, (2010).

    Args:
        sigma_b (xr.DataArray): covariance matrix of baseline distribution
        sigma_x (xr.DataArray): covariance matrix of forecast distribution
        mu_b (xr.DataArray): mean state vector of the baseline distribution
        mu_x (xr.DataArray): mean state vector of the forecast distribution
        neofs (int): number of EOFs used

    Returns:
        R (float): relative entropy
        dispersion (float): dispersion component
        signal (float): signal component

    References:
        * Branstator, Grant, and Haiyan Teng. “Two Limits of Initial-Value
          Decadal Predictability in a CGCM.” Journal of Climate 23, no. 23
          (August 27, 2010): 6292–6311. https://doi.org/10/bwq92h.
        * Kleeman, Richard. “Measuring Dynamical Prediction Utility Using
          Relative Entropy.” Journal of the Atmospheric Sciences 59, no. 13
          (July 1, 2002): 2057–72. https://doi.org/10/fqwxpk.

    """
    fac = 0.5
    dispersion = fac * (
        np.log(np.linalg.det(sigma_b) / np.linalg.det(sigma_x))
        + np.trace(sigma_x / sigma_b)
        - neofs
    )
    # https://stackoverflow.com/questions/
    # 7160162/left-matrix-division-and-numpy-solve
    # (A.T\B)*A
    x, _, _, _ = np.linalg.lstsq(
        sigma_b, mu_x - mu_b, rcond=None
    )  # sigma_b \ (mu_x - mu_b)
    signal = fac * np.matmul((mu_x.values - mu_b.values), x)
    R = dispersion + signal
    return R, dispersion, signal


def _bootstrap_dim(control, nlead_years, dim, dim_label):
    """
    Add a `len(dim_label)` dimension `dim` to uninitialized control by random
    resampling.
    """
    c_start = 0
    c_end = control["time"].size
    leads = np.arange(1, 1 + nlead_years)

    def isel_years(control, year_s, length=nlead_years):
        new = control.isel(time=slice(year_s, year_s + length))
        new = new.rename({"time": "lead"})
        new["lead"] = leads
        return new

    def create_pseudo_members(control):
        startlist = np.random.randint(c_start, c_end - nlead_years - 1, len(dim_label))
        return xr.concat([isel_years(control, start) for start in startlist], dim)

    control_uninitialized = create_pseudo_members(control)
    control_uninitialized[dim] = dim_label
    return control_uninitialized


# TODO: refactoring needed. proposed steps:
# first calculate all EOFs. save those. then calc compute_relative_entropy
def compute_relative_entropy(
    initialized,
    control,
    anomaly_data=False,
    neofs=None,
    curv=True,
    nlead=None,
    nmember_control=10,
):
    """
    Compute relative entropy.

    Calculates EOFs from anomalies. Projects fields on EOFs to receive
    pseudo-Principle Components per init and lead year. Calculate
    relative entropy based on _relative_entropy_formula.

    Args:
        initialized (xr.Dataset): anomaly ensemble data with dimensions
                                    lead, member, time and
                                    spatial [lon (x), lat(y)].
                                    DPLE or PM_ds
        control (xr.Dataset): anomaly control distribution with
                                              non-spatial dimensions:
                                              spatial [lon (x), lat(y)].
                                              - LENS: member, time
                                              - PM_control: time
        anomaly_data (bool): Input data is anomaly alread. Default: False.
        neofs (int): number of EOFs to use.
                     Default: initialized.member.size.
        curv (bool): if curvilinear grids disables EOF weights.
        nlead (int): number of timesteps calculated.
        nmember_control (int): number of members created from
                               bootstrapping from control

    Returns:
        rel_ent (xr.Dataset): relative entropy
    """
    if Eof is None:
        raise ImportError(
            "eofs is not installed; see"
            "https://ajdawson.github.io/eofs/latest/index.html"
        )
    # Defaults
    if neofs is None:
        neofs = initialized.member.size
    if nlead is None:
        nlead = initialized.lead.size

    # case if you submit control with dim time and member, LENS case
    if "member" in control.dims:
        control_uninitialized = _bootstrap_dim(
            control,
            initialized.lead.size,
            dim="init",
            dim_label=list(initialized.init.values),
        )

    # case if you only submit control with dim time, PM case
    else:
        control_uninitialized = xr.concat(
            [
                _bootstrap_dim(
                    control,
                    initialized.lead.size,
                    dim="member",
                    dim_label=np.arange(nmember_control),
                )
                for _ in range(initialized.init.size)
            ],
            dim="init",
        )
        control_uninitialized["init"] = initialized.init.values

    # initialized and control_uninitialized are allowed to have different
    # dims as I need more members to sample my control distr. properly
    if set(initialized.dims) != set(control_uninitialized.dims):
        warnings.warn(
            "Warning: initialized and control_uninitialized have different coords."
        )
        # print(initialized, control_uninitialized)

    # convert to xr.Data.Array
    if isinstance(control_uninitialized, xr.Dataset):
        control_uninitialized = control_uninitialized.to_array().squeeze()
    if isinstance(initialized, xr.Dataset):
        initialized = initialized.to_array().squeeze()

    # detrend
    non_spatial_dims = set(control_uninitialized.dims).intersection(["init", "member"])
    non_spatial_dims = list(non_spatial_dims)
    if not anomaly_data:  # if ds, control are raw values
        anom_x = initialized - control_uninitialized.mean(non_spatial_dims)
        anom_b = control_uninitialized - control_uninitialized.mean(non_spatial_dims)
    else:  # leave as is when already anomalies
        anom_x = initialized
        anom_b = control_uninitialized

    # prepare for EOF
    if curv:  # if curvilinear lon(x,y), lat(x,y) data inputs
        wgts = None
    else:  # assumes there is 'lat' in coords
        coslat = np.cos(np.deg2rad(anom_x.coords["lat"].values))
        wgts = np.sqrt(coslat)[..., np.newaxis]

    # EOF requires xr.dataArray
    if isinstance(control, xr.Dataset):
        control = control.to_array().squeeze()

    if "member" in control.dims:  # LENS
        # stack member and init into time dim, make time first
        non_spatial_control_dims = list(
            set(control.dims).intersection(["time", "member"])
        )

        transpose_dims = list(control.dims)
        transpose_dims.remove("member")
        transpose_dims.remove("time")
        dims = tuple(["time"] + transpose_dims)

        base_to_calc_eofs = (
            control.stack(new=tuple(non_spatial_control_dims))
            .rename({"new": "time"})
            .set_index({"time": "time"})
            .transpose(*dims)
        )
    else:
        # PM_control
        base_to_calc_eofs = control

    solver = Eof(base_to_calc_eofs, weights=wgts)

    re_leadtime_list = []
    leads = initialized.lead.values[:nlead]
    inits = initialized.init.values
    # DoTo: parallelize this double loop
    for init in inits:  # loop over inits
        rl, sl, dl = ([] for _ in range(3))  # lists to store results in
        for lead in leads:  # loop over lead time
            # P_b base distribution # eofs require time
            pc_b = solver.projectField(
                anom_b.sel(init=init, lead=lead)
                .drop_vars("lead")
                .rename({"member": "time"}),
                neofs=neofs,
                eofscaling=0,
                weighted=False,
            ).rename({"time": "lead"})

            mu_b = pc_b.mean("lead")
            sigma_b = xr.DataArray(np.cov(pc_b.T))

            # P_x init distribution
            pc_x = solver.projectField(
                anom_x.sel(init=init, lead=lead)
                .drop_vars("lead")
                .rename({"member": "time"}),
                neofs=neofs,
                eofscaling=0,
                weighted=False,
            ).rename({"time": "lead"})

            mu_x = pc_x.mean("lead")
            sigma_x = xr.DataArray(np.cov(pc_x.T))

            r, d, s = _relative_entropy_formula(sigma_b, sigma_x, mu_x, mu_b, neofs)

            rl.append(r)
            sl.append(s)
            dl.append(d)

        re_leadtime_list.append(
            xr.Dataset({"R": ("lead", rl), "S": ("lead", sl), "D": ("lead", dl)})
        )

    re = xr.concat(re_leadtime_list, dim="init").assign(init=inits, lead=leads)

    return re


def bootstrap_relative_entropy(
    initialized,
    control,
    sig=95,
    bootstrap=100,
    curv=True,
    neofs=None,
    nlead=None,
    anomaly_data=False,
    nmember_control=15,
):
    """
    Bootstrap relative entropy threshold.

    Generates a random uninitialized init and calculates the relative
    entropy. sig-th percentile determines threshold level.

    Args:
        initialized (xr.DataArray): initialized ensemble with dimensions
                                    init, member, time, lon (x),
                                    lat(y).
        control_uninitialized (xr.DataArray): control distribution with
                                              dimensions time,
                                              lon (x), lat(y).
        sig (int): significance level for threshold.
        bootstrap (int): number of bootstrapping iterations.
        neofs (int): number of EOFs to use. Default: initialized.member.size
        nlead (int): number of lead timestep to calculate.
                     Default: initialized.lead.size.
        curv (bool): if curvilinear grids are provided disables EOF weights.

    Returns:
        rel_ent (pd.DataFrame): relative entropy sig-th percentile threshold.

    """
    if neofs is None:
        neofs = initialized.member.size
    if nlead is None:
        nlead = initialized.lead.size
    if bootstrap < nlead:
        bootstrap = nlead + 1

    def _create_uninitialized_ensemble_from_control(ds, control, member_label):
        """Create uninitialized ensemble from control."""
        control_uninitialized = xr.concat(
            [
                _bootstrap_dim(
                    control, ds.lead.size, dim="member", dim_label=member_label
                )
                for _ in range(ds.init.size)
            ],
            dim="init",
        )
        control_uninitialized["init"] = ds.init.values
        return control_uninitialized

    results_list = []
    for _ in range(min(1, int(bootstrap / initialized.lead.size))):
        if "member" in control.dims:  # resample from lens
            uninitialized_initialized = _bootstrap_dim(
                control,
                initialized.lead.size,
                dim="init",
                dim_label=initialized.init.values,
            )
        else:  # PM
            uninitialized_initialized = _create_uninitialized_ensemble_from_control(
                initialized, control, list(initialized.member.values)
            )
        ds_pseudo_rel_ent = compute_relative_entropy(
            uninitialized_initialized,
            control,
            neofs=neofs,
            curv=curv,
            nlead=nlead,
            anomaly_data=anomaly_data,
            nmember_control=nmember_control,
        )
        results_list.append(ds_pseudo_rel_ent)
    ds_pseudo_metric = xr.concat(results_list, dim="bootstrap")
    qsig = sig / 100
    sig_level = ds_pseudo_metric.quantile(q=qsig, dim=["bootstrap", "lead", "init"])
    return sig_level
