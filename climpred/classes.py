import warnings
from copy import deepcopy

import cf_xarray
import numpy as np
import xarray as xr
from dask import is_dask_collection
from IPython.display import display_html
from xarray.core.formatting_html import dataset_repr
from xarray.core.options import OPTIONS as XR_OPTIONS
from xarray.core.utils import Frozen

from .alignment import return_inits_and_verif_dates
from .bias_removal import bias_correction, gaussian_bias_removal, xclim_sdba
from .bootstrap import (
    bootstrap_hindcast,
    bootstrap_perfect_model,
    bootstrap_uninit_pm_ensemble_from_control_cftime,
)
from .checks import (
    _check_valid_reference,
    _check_valud_alignment,
    attach_long_names,
    attach_standard_names,
    has_dataset,
    has_dims,
    has_valid_lead_units,
    is_xarray,
    match_calendars,
    match_initialized_dims,
    match_initialized_vars,
    rename_to_climpred_dims,
)
from .constants import (
    BIAS_CORRECTION_BIAS_CORRECTION_METHODS,
    BIAS_CORRECTION_TRAIN_TEST_SPLIT_METHODS,
    CLIMPRED_DIMS,
    CONCAT_KWARGS,
    CROSS_VALIDATE_METHODS,
    INTERNAL_BIAS_CORRECTION_METHODS,
    M2M_MEMBER_DIM,
    XCLIM_BIAS_CORRECTION_METHODS,
)
from .exceptions import DimensionError, VariableError
from .graphics import plot_ensemble_perfect_model, plot_lead_timeseries_hindcast
from .logging import log_compute_hindcast_header
from .options import OPTIONS
from .prediction import (
    _apply_metric_at_given_lead,
    _get_metric_comparison_dim,
    compute_perfect_model,
)
from .reference import compute_climatology, compute_persistence
from .smoothing import (
    _reset_temporal_axis,
    smooth_goddard_2013,
    spatial_smoothing_xesmf,
    temporal_smoothing,
)
from .utils import (
    broadcast_metric_kwargs_for_rps,
    convert_time_index,
    convert_Timedelta_to_lead_units,
)


def _display_metadata(self):
    """
    This is called in the following case:

    ```
    dp = cp.HindcastEnsemble(dple)
    print(dp)
    ```
    """
    SPACE = "    "
    header = f"<climpred.{type(self).__name__}>"
    summary = header + "\nInitialized Ensemble:\n"
    summary += SPACE + str(self._datasets["initialized"].data_vars)[18:].strip() + "\n"
    if isinstance(self, HindcastEnsemble):
        # Prints out observations and associated variables if they exist.
        # If not, just write "None".
        summary += "Observations:\n"
        if any(self._datasets["observations"]):
            num_obs = len(self._datasets["observations"].data_vars)
            for i in range(1, num_obs + 1):
                summary += (
                    SPACE
                    + str(self._datasets["observations"].data_vars)
                    .split("\n")[i]
                    .strip()
                    + "\n"
                )
        else:
            summary += SPACE + "None\n"
    elif isinstance(self, PerfectModelEnsemble):
        summary += "Control:\n"
        # Prints out control variables if a control is appended. If not,
        # just write "None".
        if any(self._datasets["control"]):
            num_ctrl = len(self._datasets["control"].data_vars)
            for i in range(1, num_ctrl + 1):
                summary += (
                    SPACE
                    + str(self._datasets["control"].data_vars).split("\n")[i].strip()
                    + "\n"
                )
        else:
            summary += SPACE + "None\n"
    if any(self._datasets["uninitialized"]):
        summary += "Uninitialized:\n"
        summary += SPACE + str(self._datasets["uninitialized"].data_vars)[18:].strip()
    else:
        summary += "Uninitialized:\n"
        summary += SPACE + "None"
    return summary


def _display_metadata_html(self):
    header = f"<h4>climpred.{type(self).__name__}</h4>"
    display_html(header, raw=True)
    init_repr_str = dataset_repr(self._datasets["initialized"])
    init_repr_str = init_repr_str.replace("xarray.Dataset", "Initialized Ensemble")
    display_html(init_repr_str, raw=True)

    if isinstance(self, HindcastEnsemble):
        if any(self._datasets["observations"]):
            obs_repr_str = dataset_repr(self._datasets["observations"])
            obs_repr_str = obs_repr_str.replace("xarray.Dataset", "Observations")
            display_html(obs_repr_str, raw=True)
    elif isinstance(self, PerfectModelEnsemble):
        if any(self._datasets["control"]):
            control_repr_str = dataset_repr(self._datasets["control"])
            control_repr_str = control_repr_str.replace(
                "xarray.Dataset", "Control Simulation"
            )
            display_html(control_repr_str, raw=True)

    if any(self._datasets["uninitialized"]):
        uninit_repr_str = dataset_repr(self._datasets["uninitialized"])
        uninit_repr_str = uninit_repr_str.replace("xarray.Dataset", "Uninitialized")
        display_html(uninit_repr_str, raw=True)
    # better would be to aggregate repr_strs and then all return but this fails
    # TypeError: __repr__ returned non-string (type NoneType)
    # workaround return empty string
    return ""


class PredictionEnsemble:
    """
    The main object. This is the super of both `PerfectModelEnsemble` and
    `HindcastEnsemble`. This cannot be called directly by a user, but
    should house functions that both ensemble types can use.
    """

    @is_xarray(1)
    def __init__(self, xobj):
        if isinstance(xobj, xr.DataArray):
            # makes applying prediction functions easier, etc.
            xobj = xobj.to_dataset()
        xobj = rename_to_climpred_dims(xobj)
        has_dims(xobj, ["init", "lead"], "PredictionEnsemble")
        # Check that init is int, cftime, or datetime; convert ints or cftime to
        # datetime.
        xobj = convert_time_index(xobj, "init", "xobj[init]")
        # Put this after `convert_time_index` since it assigns 'years' attribute if the
        # `init` dimension is a `float` or `int`.
        xobj = convert_Timedelta_to_lead_units(xobj)
        has_valid_lead_units(xobj)
        # add metadata
        xobj = attach_standard_names(xobj)
        xobj = attach_long_names(xobj)
        xobj = xobj.cf.add_canonical_attributes(
            verbose=False, override=True, skip="units"
        )
        del xobj.attrs["history"]
        # Add initialized dictionary and reserve sub-dictionary for an uninitialized
        # run.
        self._datasets = {"initialized": xobj, "uninitialized": {}}
        self.kind = "prediction"
        self._temporally_smoothed = None
        self._is_annual_lead = None
        self._warn_if_chunked_along_init_member_time()

    @property
    def coords(self):
        """Dictionary of xarray.DataArray objects corresponding to coordinate
        variables available in all PredictionEnsemble._datasets.
        """
        pe_coords = self.get_initialized().coords.to_dataset()
        for ds in self._datasets.values():
            if isinstance(ds, xr.Dataset):
                pe_coords.update(ds.coords.to_dataset())
        return pe_coords.coords

    @property
    def nbytes(self) -> int:
        """Bytes sizes of all PredictionEnsemble._datasets."""
        return sum(
            [
                sum(v.nbytes for v in ds.variables.values())
                for ds in self._datasets.values()
                if isinstance(ds, xr.Dataset)
            ]
        )

    @property
    def sizes(self):
        """Mapping from dimension names to lengths for all PredictionEnsemble._datasets."""
        pe_dims = dict(self.get_initialized().dims)
        for ds in self._datasets.values():
            if isinstance(ds, xr.Dataset):
                pe_dims.update(dict(ds.dims))
        return pe_dims

    @property
    def dims(self):
        """Mapping from dimension names to lengths all PredictionEnsemble._datasets."""
        return Frozen(self.sizes)

    @property
    def chunks(self):
        """Mapping from chunks all PredictionEnsemble._datasets."""
        pe_chunks = dict(self.get_initialized().chunks)
        for ds in self._datasets.values():
            if isinstance(ds, xr.Dataset):
                for d in ds.chunks:
                    if d not in pe_chunks:
                        pe_chunks.update({d: ds.chunks[d]})
        return Frozen(pe_chunks)

    @property
    def data_vars(self):
        """Dictionary of DataArray objects corresponding to data variables available in all PredictionEnsemble._datasets."""
        varset = set(self.get_initialized().data_vars)
        for ds in self._datasets.values():
            if isinstance(ds, xr.Dataset):
                # take union
                varset = varset & set(ds.data_vars)
        varlist = list(varset)
        return self.get_initialized()[varlist].data_vars

    # when you just print it interactively
    # https://stackoverflow.com/questions/1535327/how-to-print-objects-of-class-using-print
    def __repr__(self):
        if XR_OPTIONS["display_style"] == "html":
            return _display_metadata_html(self)
        else:
            return _display_metadata(self)

    def __len__(self):
        """Number of all variables in all PredictionEnsemble._datasets."""
        return len(self.data_vars)

    def __iter__(self):
        """Iterate over underlying xr.Datasets for initialized, uninitialized, observations."""
        return iter(self._datasets.values())

    def __delitem__(self, key):
        """Remove a variable from this PredictionEnsemble."""
        del self._datasets["initialized"][key]
        for ds in self._datasets.values():
            if isinstance(ds, xr.Dataset):
                if key in ds.data_vars:
                    del ds[key]

    def __contains__(self, key):
        """The 'in' operator will return true or false depending on whether
        'key' is an array in all PredictionEnsemble._datasets or not.
        """
        contained = True
        for ds in self._datasets.values():
            if isinstance(ds, xr.Dataset):
                if key not in ds.data_vars:
                    contained = False
        return contained

    def equals(self, other):
        """Two PredictionEnsembles are equal if they have matching variables and
        coordinates, all of which are equal.
        PredictionEnsembles can still be equal (like pandas objects) if they have NaN
        values in the same locations.
        This method is necessary because `v1 == v2` for ``PredictionEnsembles``
        does element-wise comparisons (like numpy.ndarrays)."""
        if not isinstance(other, PredictionEnsemble):
            return False
        if other.kind != self.kind:
            return False
        equal = True
        try:
            for ds_name in self._datasets.keys():
                if isinstance(self._datasets[ds_name], xr.Dataset):
                    if not self._datasets[ds_name].equals(other._datasets[ds_name]):
                        equal = False
        except Exception:
            return False
        return equal

    def identical(self, other):
        """Like equals, but also checks all dataset attributes and the
        attributes on all variables and coordinates."""
        if not isinstance(other, PredictionEnsemble):
            return False
        if other.kind != self.kind:
            return False
        id = True
        try:
            for ds_name in self._datasets.keys():
                if not self._datasets[ds_name].identical(other._datasets[ds_name]):
                    id = False
        except Exception:
            return False
        return id

    def plot(self, variable=None, ax=None, show_members=False, cmap=None):
        """Plot datasets from PredictionEnsemble.

        Args:
            variable (str or None): `variable` to show. Defaults to first in data_vars.
            ax (plt.axes): Axis to use in plotting. By default, creates a new axis.
            show_members (bool): whether to display all members individually.
                Defaults to False.
            cmap (str): Name of matplotlib-recognized colorbar. Defaults to `jet` for
                `HindcastEnsemble` and `tab10` for `PerfectModelEnsemble`.

        Returns:
            ax: plt.axes

        """
        if self.kind == "hindcast":
            if cmap is None:
                cmap = "jet"
            return plot_lead_timeseries_hindcast(
                self, variable=variable, ax=ax, show_members=show_members, cmap=cmap
            )
        elif self.kind == "perfect":
            if cmap is None:
                cmap = "tab10"
            return plot_ensemble_perfect_model(
                self, variable=variable, ax=ax, show_members=show_members, cmap=cmap
            )

    def _math(self, other, operator):
        """Helper function for __add__, __sub__, __mul__, __truediv__.

        Allows math operations with type:
            - int
            - float
            - np.ndarray
            - xr.DataArray without new dimensions
            - xr.Dataset without new dimensions or variables

        """
        assert isinstance(operator, str)

        def add(a, b):
            return a + b

        def sub(a, b):
            return a - b

        def mul(a, b):
            return a * b

        def div(a, b):
            return a / b

        ALLOWED_TYPES_FOR_MATH_OPERATORS = [
            int,
            float,
            np.ndarray,
            xr.DataArray,
            xr.Dataset,
            type(self),
        ]
        OPERATOR_STR = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
        }
        error_str = f"Cannot use {type(self)} {OPERATOR_STR[operator]} {type(other)}"

        # catch undefined types for other
        if not isinstance(other, tuple(ALLOWED_TYPES_FOR_MATH_OPERATORS)):
            raise TypeError(
                f"{error_str} because type {type(other)} not supported. "
                f"Please choose from {ALLOWED_TYPES_FOR_MATH_OPERATORS}."
            )
        # catch other dimensions in other
        if isinstance(other, tuple([xr.Dataset, xr.DataArray])):
            if not set(other.dims).issubset(self._datasets["initialized"].dims):
                raise DimensionError(f"{error_str} containing new dimensions.")
        # catch xr.Dataset with different data_vars
        if isinstance(other, xr.Dataset):
            if list(other.data_vars) != list(self._datasets["initialized"].data_vars):
                raise VariableError(
                    f"{error_str} with new `data_vars`. Please use {type(self)} "
                    f"{operator} {type(other)} only with same `data_vars`. Found "
                    f"initialized.data_vars = "
                    f' {list(self._datasets["initialized"].data_vars)} vs. '
                    f"other.data_vars = {list(other.data_vars)}."
                )

        operator = eval(operator)

        if isinstance(other, PredictionEnsemble):
            # Create temporary copy to modify to avoid inplace operation.
            datasets = self._datasets.copy()
            for dataset in datasets:
                # Some pre-allocated entries might be empty, such as 'uninitialized'
                if isinstance(other._datasets[dataset], xr.Dataset) and isinstance(
                    self._datasets[dataset], xr.Dataset
                ):
                    datasets[dataset] = operator(
                        datasets[dataset], other._datasets[dataset]
                    )
            return self._construct_direct(datasets, kind=self.kind)
        else:
            return self._apply_func(operator, other)

    def __add__(self, other):
        return self._math(other, operator="add")

    def __sub__(self, other):
        return self._math(other, operator="sub")

    def __mul__(self, other):
        return self._math(other, operator="mul")

    def __truediv__(self, other):
        return self._math(other, operator="div")

    def __getitem__(self, varlist):
        """Allows subsetting data variable from PredictionEnsemble as from xr.Dataset.

        Args:
            * varlist (list of str, str): list of names or name of data variable(s) to
                subselect
        """
        if isinstance(varlist, str):
            varlist = [varlist]
        if not isinstance(varlist, list):
            raise ValueError(
                "Please subset PredictionEnsemble as you would subset an xr.Dataset "
                "with a list or single string of variable name(s), found "
                f"{type(varlist)}."
            )

        def sel_vars(ds, varlist):
            return ds[varlist]

        return self._apply_func(sel_vars, varlist)

    def __getattr__(self, name):
        """Allows for xarray methods to be applied to our prediction objects.

        Args:
            * name: Function, e.g., .isel() or .sum().
        """

        def wrapper(*args, **kwargs):
            """Applies arbitrary function to all datasets in the PredictionEnsemble
            object.

            Got this from: https://stackoverflow.com/questions/41919499/
            how-to-call-undefined-methods-sequentially-in-python-class
            """

            def _apply_xr_func(v, name, *args, **kwargs):
                """Handles exceptions in our dictionary comprehension.

                In other words, this will skip applying the arbitrary function
                to a sub-dataset if a ValueError is thrown. This specifically
                targets cases where certain datasets don't have the given
                dim that's being called. E.g., ``.isel(lead=0)`` should only
                be applied to the initialized dataset.

                Reference:
                  * https://stackoverflow.com/questions/1528237/
                    how-to-handle-exceptions-in-a-list-comprehensions
                """
                try:
                    return getattr(v, name)(*args, **kwargs)
                # ValueError : Cases such as .sum(dim='time'). This doesn't apply
                #              it to the given dataset if the dimension doesn't exist.
                # KeyError : Cases where a function calls the index of a Dataset. Such
                #            as ds[dim] and the dim doesn't exist as a key.
                # DimensionError: This accounts for our custom error when applying
                # some stats functions.
                except (ValueError, KeyError, DimensionError) as e:
                    if args == tuple():
                        func_name = False
                    else:
                        if callable(args[0]):
                            func_name = args[0].__name__
                        else:  # for xarray calls like pe.mean()
                            func_name = False
                    dim = kwargs.get("dim", False)
                    error_type = type(e).__name__
                    if func_name:
                        if len(args) > 1:
                            msg = f"{func_name}({args[1:]}, {kwargs}) failed\n{error_type}: {e}"
                        else:
                            msg = f"{func_name}({kwargs}) failed\n{error_type}: {e}"
                    else:
                        msg = f"xr.{name}({args}, {kwargs}) failed\n{error_type}: {e}"
                    if set(["lead", "init"]).issubset(set(v.dims)):  # initialized
                        if dim not in v.dims:
                            if OPTIONS["warn_for_failed_PredictionEnsemble_xr_call"]:
                                warnings.warn(f"Error due to initialized:  {msg}")
                    elif set(["time"]).issubset(
                        set(v.dims)
                    ):  # uninitialized, control, verification
                        if dim not in v.dims:
                            if OPTIONS["warn_for_failed_PredictionEnsemble_xr_call"]:
                                warnings.warn(
                                    f"Error due to verification/control/uninitialized: {msg}"
                                )
                    else:
                        if OPTIONS["warn_for_failed_PredictionEnsemble_xr_call"]:
                            warnings.warn(msg)
                    return v

            return self._apply_func(_apply_xr_func, name, *args, **kwargs)

        return wrapper

    @classmethod
    def _construct_direct(cls, datasets, kind):
        """Shortcut around __init__ for internal use to avoid inplace
        operations.

        Pulled from xarrray Dataset class.
        https://github.com/pydata/xarray/blob/master/xarray/core/dataset.py
        """
        obj = object.__new__(cls)
        obj._datasets = datasets
        obj.kind = kind
        obj._warn_if_chunked_along_init_member_time()
        return obj

    def _apply_func(self, func, *args, **kwargs):
        """Apply a function to all datasets in a `PredictionEnsemble`."""
        # Create temporary copy to modify to avoid inplace operation.
        datasets = self._datasets.copy()

        # More explicit than nested dictionary comprehension.
        for key, ds in datasets.items():
            # If ds is xr.Dataset, apply the function directly to it. else, e.g. for {} ignore
            if isinstance(ds, xr.Dataset):
                dim = kwargs.get("dim", "")
                if "_or_" in dim:
                    dims = dim.split("_or_")
                    if set(dims).issubset(ds.dims):
                        raise ValueError(
                            f"{dims} cannot be both in {key} dataset, found {ds.dims}"
                        )
                    kwargs_dim0 = kwargs.copy()
                    kwargs_dim0["dim"] = dims[0]
                    kwargs_dim1 = kwargs.copy()
                    kwargs_dim1["dim"] = dims[1]
                    if dims[0] in ds.dims and dims[1] not in ds.dims:
                        datasets.update({key: func(ds, *args, **kwargs_dim0)})
                    if dims[1] in ds.dims and dims[0] not in ds.dims:
                        datasets.update({key: func(ds, *args, **kwargs_dim1)})
                else:
                    datasets.update({key: func(ds, *args, **kwargs)})
        # Instantiates new object with the modified datasets.
        return self._construct_direct(datasets, kind=self.kind)

    def get_initialized(self):
        """Returns the xarray dataset for the initialized ensemble."""
        return self._datasets["initialized"]

    def get_uninitialized(self):
        """Returns the xarray dataset for the uninitialized ensemble."""
        return self._datasets["uninitialized"]

    def smooth(self, smooth_kws=None, how="mean", **xesmf_kwargs):
        """Smooth all entries of PredictionEnsemble in the same manner to be
        able to still calculate prediction skill afterwards.

        Args:
            smooth_kws (dict or str): Dictionary to specify the dims to
                smooth compatible with
                :py:func:`~climpred.smoothing.spatial_smoothing_xesmf` or
                :py:func:`~climpred.smoothing.temporal_smoothing`.
                Shortcut for Goddard et al. 2013 recommendations:
                'goddard2013'. Defaults to None.
            how (str): how to smooth temporally. From ['mean','sum']. Defaults to
                'mean'.
            **xesmf_kwargs (args): kwargs passed to
                :py:func:`~climpred.smoothing.spatial_smoothing_xesmf`

        Examples:
            >>> PerfectModelEnsemble.get_initialized().lead.size
            20
            >>> PerfectModelEnsemble.smooth({'lead':4}, how='sum').get_initialized().lead.size
            17

            >>> HindcastEnsemble_3D.smooth({'lon':1, 'lat':1})
            <climpred.HindcastEnsemble>
            Initialized Ensemble:
                SST      (init, lead, lat, lon) float64 -0.3236 -0.3161 -0.3083 ... 0.0 0.0
            Observations:
                SST      (time, lat, lon) float64 0.002937 0.001561 0.002587 ... 0.0 0.0 0.0
            Uninitialized:
                None

            ``smooth`` simultaneously aggregates spatially listening to ``lon`` and ``lat`` and temporally listening to ``lead`` or ``time``.

            >>> HindcastEnsemble_3D.smooth({'lead': 2, 'lat': 5, 'lon': 4}).get_initialized().coords
            Coordinates:
              * init     (init) object 1954-01-01 00:00:00 ... 2017-01-01 00:00:00
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9
              * lon      (lon) float64 250.8 254.8 258.8 262.8
              * lat      (lat) float64 -9.75 -4.75
            >>> HindcastEnsemble_3D.smooth('goddard2013').get_initialized().coords
            Coordinates:
              * init     (init) object 1954-01-01 00:00:00 ... 2017-01-01 00:00:00
              * lead     (lead) int32 1 2 3 4 5 6 7
              * lon      (lon) float64 250.8 255.8 260.8 265.8
              * lat      (lat) float64 -9.75 -4.75


        """
        if not smooth_kws:
            return self
        # get proper smoothing function based on smooth args
        if isinstance(smooth_kws, str):
            if "goddard" in smooth_kws:
                if self._is_annual_lead:
                    smooth_fct = smooth_goddard_2013
                    tsmooth_kws = {"lead": 4}  # default
                    d_lon_lat_kws = {"lon": 5, "lat": 5}  # default
                else:
                    raise ValueError(
                        "`goddard2013` smoothing only available for annual leads."
                    )
            else:
                raise ValueError(
                    'Please provide from list of available smoothings: \
                     ["goddard2013"]'
                )
        # TODO: actively searches for lot and lat in dims. Maybe this part of the code
        # could be more robust in how it finds these two spatial dimensions regardless
        # of name. Optional work in progress comment.
        elif isinstance(smooth_kws, dict):
            non_time_dims = [
                dim for dim in smooth_kws.keys() if dim not in ["time", "lead"]
            ]
            if len(non_time_dims) > 0:
                non_time_dims = non_time_dims[0]
            # goddard when time_dim and lon/lat given
            if ("lon" in smooth_kws or "lat" in smooth_kws) and (
                "lead" in smooth_kws or "time" in smooth_kws
            ):
                smooth_fct = smooth_goddard_2013
                # separate lon, lat keywords into d_lon_lat_kws
                d_lon_lat_kws = dict()
                tsmooth_kws = dict()
                for c in ["lon", "lat"]:
                    if c in smooth_kws:
                        d_lon_lat_kws[c] = smooth_kws[c]
                for c in ["lead", "time"]:
                    if c in smooth_kws:
                        tsmooth_kws[c] = smooth_kws[c]
            # else only one smoothing operation
            elif "lon" in smooth_kws or "lat" in smooth_kws:
                smooth_fct = spatial_smoothing_xesmf
                d_lon_lat_kws = smooth_kws
                tsmooth_kws = None
            elif "lead" in smooth_kws or "time" in smooth_kws:
                smooth_fct = temporal_smoothing
                d_lon_lat_kws = None
                tsmooth_kws = smooth_kws
            else:
                raise ValueError(
                    'Please provide kwargs to fulfill functions: \
                     ["spatial_smoothing_xesmf", "temporal_smoothing"].'
                )
        else:
            raise ValueError(
                "Please provide kwargs as dict or str and not", type(smooth_kws)
            )
        self = self.map(
            smooth_fct,
            tsmooth_kws=tsmooth_kws,
            d_lon_lat_kws=d_lon_lat_kws,
            how=how,
            **xesmf_kwargs,
        )
        if smooth_fct == smooth_goddard_2013 or smooth_fct == temporal_smoothing:
            self._temporally_smoothed = tsmooth_kws
        return self

    def _warn_if_chunked_along_init_member_time(self):
        """Warn upon instantiation when CLIMPRED_DIMS except ``lead`` are chunked with
        more than one chunk to show how to circumvent ``xskillscore`` chunking
        ``ValueError``."""
        suggest_one_chunk = []
        for d in self.chunks:
            if d in ["time", "init", "member"]:
                if len(self.chunks[d]) > 1:
                    suggest_one_chunk.append(d)
        if len(suggest_one_chunk) > 0:
            name = (
                str(type(self))
                .replace("<class 'climpred.classes.", "")
                .replace("'>", "")
            )
            # init cannot be dim when time chunked
            suggest_one_chunk_time_to_init = suggest_one_chunk.copy()
            if "time" in suggest_one_chunk_time_to_init:
                suggest_one_chunk_time_to_init.remove("time")
                suggest_one_chunk_time_to_init.append("init")
            msg = f"{name} is chunked along dimensions {suggest_one_chunk} with more than one chunk. `{name}.chunks={self.chunks}`.\nYou cannot call `{name}.verify` or `{name}.bootstrap` in combination with any of {suggest_one_chunk_time_to_init} passed as `dim`. In order to do so, please rechunk {suggest_one_chunk} with `{name}.chunk({{dim:-1}}).verify(dim=dim).`\nIf you do not want to use dimensions {suggest_one_chunk_time_to_init} in `{name}.verify(dim=dim)`, you can disregard this warning."
            # chunk lead:1 in HindcastEnsemble
            if self.kind == "hindcast":
                msg += '\nIn `HindcastEnsemble`s you may also create one chunk per lead, as the `climpred` internally loops over lead, e.g. `.chunk({{"lead": 1}}).verify().`'
            # chunk auto on non-climpred dims
            ndims = list(self.sizes)
            for d in CLIMPRED_DIMS:
                if d in ndims:
                    ndims.remove(d)
            if len(ndims) > 0:
                msg += f'\nConsider chunking embarassingly parallel dimensions such as {ndims} automatically, i.e. `{name}.chunk({ndims[0]}="auto").verify(...).'
            warnings.warn(msg)


class PerfectModelEnsemble(PredictionEnsemble):
    """An object for "perfect model" climate prediction ensembles.

    `PerfectModelEnsemble` is a sub-class of `PredictionEnsemble`. It tracks
    the control run used to initialize the ensemble for easy computations,
    bootstrapping, etc.

    This object is built on `xarray` and thus requires the input object to
    be an `xarray` Dataset or DataArray.
    """

    def __init__(self, xobj):
        """Create a `PerfectModelEnsemble` object by inputting output from the
        control run in `xarray` format.

        Args:
          xobj (xarray object):
            decadal prediction ensemble output.

        Attributes:
            control: Dictionary of control run associated with the initialized
                     ensemble.
            uninitialized: Dictionary of uninitialized run that is
                           bootstrapped from the initialized run.
        """

        super().__init__(xobj)
        # Reserve sub-dictionary for the control simulation.
        self._datasets.update({"control": {}})
        self.kind = "perfect"

    def _apply_climpred_function(self, func, input_dict=None, **kwargs):
        """Helper function to loop through observations and apply an arbitrary climpred
        function.

        Args:
            func (function): climpred function to apply to object.
            input_dict (dict): dictionary with the following things:
                * ensemble: initialized or uninitialized ensemble.
                * control: control dictionary from HindcastEnsemble.
                * init (bool): True if the initialized ensemble, False if uninitialized.
        """
        ensemble = input_dict["ensemble"]
        control = input_dict["control"]
        init = input_dict["init"]
        init_vars, ctrl_vars = self._vars_to_drop(init=init)
        ensemble = ensemble.drop_vars(init_vars)
        if control:
            control = control.drop_vars(ctrl_vars)
        return func(ensemble, control, **kwargs)

    def _vars_to_drop(self, init=True):
        """Returns list of variables to drop when comparing
        initialized/uninitialized to a control.

        This is useful if the two products being compared do not share the same
        variables. I.e., if the control has ['SST'] and the initialized has
        ['SST', 'SALT'], this will return a list with ['SALT'] to be dropped
        from the initialized.

        Args:
          init (bool, default True):
            If `True`, check variables on the initialized.
            If `False`, check variables on the uninitialized.

        Returns:
          Lists of variables to drop from the initialized/uninitialized
          and control Datasets.
        """
        init_str = "initialized" if init else "uninitialized"
        init_vars = list(self._datasets[init_str])
        # only drop if control present
        if self._datasets["control"]:
            ctrl_vars = list(self._datasets["control"])
            # Make lists of variables to drop that aren't in common
            # with one another.
            init_vars_to_drop = list(set(init_vars) - set(ctrl_vars))
            ctrl_vars_to_drop = list(set(ctrl_vars) - set(init_vars))
        else:
            init_vars_to_drop, ctrl_vars_to_drop = [], []
        return init_vars_to_drop, ctrl_vars_to_drop

    @is_xarray(1)
    def add_control(self, xobj):
        """Add the control run that initialized the climate prediction
        ensemble.

        Args:
            xobj (xarray object): Dataset/DataArray of the control run.
        """
        # NOTE: These should all be decorators.
        if isinstance(xobj, xr.DataArray):
            xobj = xobj.to_dataset()
        match_initialized_dims(self._datasets["initialized"], xobj)
        match_initialized_vars(self._datasets["initialized"], xobj)
        # Check that init is int, cftime, or datetime; convert ints or cftime to
        # datetime.
        xobj = convert_time_index(xobj, "time", "xobj[init]")
        # Check that converted/original cftime calendar is the same as the
        # initialized calendar to avoid any alignment errors.
        match_calendars(self._datasets["initialized"], xobj, kind2="control")
        datasets = self._datasets.copy()
        datasets.update({"control": xobj})
        return self._construct_direct(datasets, kind="perfect")

    def generate_uninitialized(self):
        """Generate an uninitialized ensemble by bootstrapping the
        initialized prediction ensemble.

        Returns:
            Bootstrapped (uninitialized) ensemble as a Dataset.
        """
        has_dataset(
            self._datasets["control"], "control", "generate an uninitialized ensemble."
        )

        uninit = bootstrap_uninit_pm_ensemble_from_control_cftime(
            self._datasets["initialized"], self._datasets["control"]
        )
        datasets = self._datasets.copy()
        datasets.update({"uninitialized": uninit})
        return self._construct_direct(datasets, kind="perfect")

    def get_control(self):
        """Returns the control as an xarray dataset."""
        return self._datasets["control"]

    def verify(
        self,
        metric=None,
        comparison=None,
        dim=None,
        reference=None,
        **metric_kwargs,
    ):
        """Verify initialized predictions against a configuration of other ensemble members.

        .. note::
            The configuration of the other ensemble members is based off of the
            ``comparison`` keyword argument.

        Args:
            metric (str, :py:class:`~climpred.metrics.Metric`): Metric to apply in the
                comparison. See `metrics </metrics.html>`_.
            comparison (str, :py:class:`~climpred.comparisons.Comparison`): How to
                compare the initialized prediction ensemble with itself, see
                `comparisons </comparisons.html>`_.
            dim (str, list of str): Dimension(s) over which to apply ``metric``.
                ``dim`` is passed on to xskillscore.{metric} and includes xskillscore's
                ``member_dim``. ``dim`` should contain ``member`` when ``comparison``
                is probabilistic but should not contain ``member`` when
                ``comparison=e2c``. Defaults to ``None`` meaning that all dimensions
                other than ``lead`` are reduced.
            reference (str, list of str): Type of reference forecasts with which to
                verify. One or more of ['uninitialized', 'persistence', 'climatology'].
            **metric_kwargs (optional): Arguments passed to ``metric``.

        Returns:
            Dataset with dimension skill reduced by dim containing initialized and
            reference skill(s) if specified.

        Example:
            Root mean square error (``rmse``) comparing every member with the
            ensemble mean forecast (``m2e``) for all leads reducing dimensions
            ``init`` and ``member``:

            >>> PerfectModelEnsemble.verify(metric='rmse', comparison='m2e',
            ...     dim=['init','member'])
            <xarray.Dataset>
            Dimensions:  (lead: 20)
            Coordinates:
              * lead     (lead) int64 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
            Data variables:
                tos      (lead) float32 0.1028 0.1249 0.1443 0.1707 ... 0.2113 0.2452 0.2297


            Pearson's Anomaly Correlation ('acc') comparing every member to every
            other member (``m2m``) reducing dimensions ``member`` and ``init`` while
            also calculating reference skill for the ``persistence``, ``climatology``
            and ``uninitialized`` forecast.

            >>> PerfectModelEnsemble.verify(metric='acc', comparison='m2m',
            ...     dim=['init', 'member'],
            ...     reference=['persistence', 'climatology' ,'uninitialized'])
            <xarray.Dataset>
            Dimensions:  (skill: 4, lead: 20)
            Coordinates:
              * lead     (lead) int64 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
              * skill    (skill) <U13 'initialized' 'persistence' ... 'uninitialized'
            Data variables:
                tos      (skill, lead) float64 0.7941 0.7489 0.5623 ... 0.1327 0.4547 0.3253
        """
        reference = _check_valid_reference(reference)
        input_dict = {
            "ensemble": self._datasets["initialized"],
            "control": self._datasets["control"]
            if isinstance(self._datasets["control"], xr.Dataset)
            else None,
            "init": True,
        }
        result = self._apply_climpred_function(
            compute_perfect_model,
            input_dict=input_dict,
            metric=metric,
            comparison=comparison,
            dim=dim,
            add_attrs=False,
            **metric_kwargs,
        )
        if self._temporally_smoothed:
            result = _reset_temporal_axis(result, self._temporally_smoothed, dim="lead")
            result["lead"].attrs = self.get_initialized().lead.attrs
        # compute reference skills
        if reference:
            for r in reference:
                dim_orig = deepcopy(dim)  # preserve dim, because
                ref_compute_kwargs = metric_kwargs.copy()  # persistence changes dim
                ref_compute_kwargs.update({"dim": dim_orig, "metric": metric})
                if r != "persistence":
                    ref_compute_kwargs["comparison"] = comparison
                ref = getattr(self, f"_compute_{r}")(**ref_compute_kwargs)
                result = xr.concat([result, ref], dim="skill", **CONCAT_KWARGS)
            result = result.assign_coords(skill=["initialized"] + reference)
        return result.squeeze()

    def _compute_uninitialized(
        self, metric=None, comparison=None, dim=None, **metric_kwargs
    ):
        """Verify the bootstrapped uninitialized run against itself.

        .. note::
            The configuration of the other ensemble members is based off of the
            ``comparison`` keyword argument.

        Args:
            metric (str, :py:class:`~climpred.metrics.Metric`): Metric to apply in the
                comparison. See `metrics </metrics.html>`_.
            comparison (str, :py:class:`~climpred.comparisons.Comparison`): How to
                compare the uninitialized against itself, see
                `comparisons </comparisons.html>`_.
            dim (str, list of str): Dimension(s) over which to apply metric.
                ``dim`` is passed on to xskillscore.{metric} and includes xskillscore's
                ``member_dim``. ``dim`` should contain ``member`` when ``comparison``
                is probabilistic but should not contain ``member`` when
                ``comparison=e2c``. Defaults to ``None``, meaning that all dimensions
                other than ``lead`` are reduced.
            **metric_kwargs (optional): Arguments passed to ``metric``.

        Returns:
            Dataset with dimension skill containing initialized and reference skill(s).
        """
        has_dataset(
            self._datasets["uninitialized"],
            "uninitialized",
            "compute an uninitialized metric",
        )
        input_dict = {
            "ensemble": self._datasets["uninitialized"],
            "control": self._datasets["control"]
            if isinstance(self._datasets["control"], xr.Dataset)
            else None,
            "init": False,
        }
        if dim is None:
            dim = list(self._datasets["initialized"].isel(lead=0).dims)
        res = self._apply_climpred_function(
            compute_perfect_model,
            input_dict=input_dict,
            metric=metric,
            comparison=comparison,
            dim=dim,
            **metric_kwargs,
        )
        if self._temporally_smoothed:
            res = _reset_temporal_axis(res, self._temporally_smoothed, dim="lead")
            res["lead"].attrs = self.get_initialized().lead.attrs
        return res

    def _compute_persistence(self, metric=None, dim=None, **metric_kwargs):
        """Verify a simple persistence forecast of the control run against itself.

        Args:
            metric (str, :py:class:`~climpred.metrics.Metric`): Metric to use when
            verifying skill of the persistence forecast. See `metrics </metrics.html>`_.
            dim (str, list of str): Dimension(s) over which to apply metric.
                ``dim`` is passed on to xskillscore.{metric} and includes xskillscore's
                ``member_dim``. ``dim`` should contain ``member`` when ``comparison``
                is probabilistic but should not contain ``member`` when
                ``comparison=e2c``. Defaults to ``None``, meaning that all dimensions
                other than ``lead`` are reduced.
            **metric_kwargs (optional): Arguments passed to ``metric``.

        Returns:
            Dataset of persistence forecast results.

        Reference:
            * Chapter 8 (Short-Term Climate Prediction) in
              Van den Dool, Huug. Empirical methods in short-term climate
              prediction. Oxford University Press, 2007.
        """
        has_dataset(
            self._datasets["control"], "control", "compute a persistence forecast"
        )
        input_dict = {
            "ensemble": self._datasets["initialized"],
            "control": self._datasets["control"],
            "init": True,
        }
        if dim is None:
            dim = list(self._datasets["initialized"].isel(lead=0).dims)
        res = self._apply_climpred_function(
            compute_persistence,
            input_dict=input_dict,
            metric=metric,
            alignment="same_inits",
            dim=dim,
            **metric_kwargs,
        )
        if self._temporally_smoothed:
            res = _reset_temporal_axis(res, self._temporally_smoothed, dim="lead")
            res["lead"].attrs = self.get_initialized().lead.attrs
        return res

    def _compute_climatology(
        self, metric=None, comparison=None, dim=None, **metric_kwargs
    ):
        """Verify a climatology forecast of the control run against itself.

        Args:
            metric (str, :py:class:`~climpred.metrics.Metric`): Metric to use when
            verifying skill of the persistence forecast. See `metrics </metrics.html>`_.
            dim (str, list of str): Dimension(s) over which to apply metric.
                ``dim`` is passed on to xskillscore.{metric} and includes xskillscore's
                ``member_dim``. ``dim`` should contain ``member`` when ``comparison``
                is probabilistic but should not contain ``member`` when
                ``comparison=e2c``. Defaults to ``None``, meaning that all dimensions
                other than ``lead`` are reduced.
            **metric_kwargs (optional): Arguments passed to ``metric``.

        Returns:
            Dataset of persistence forecast results.

        Reference:
            * Chapter 8 (Short-Term Climate Prediction) in
              Van den Dool, Huug. Empirical methods in short-term climate
              prediction. Oxford University Press, 2007.
        """
        input_dict = {
            "ensemble": self._datasets["initialized"],
            "control": self._datasets["control"]
            if isinstance(self._datasets["control"], xr.Dataset)
            else None,
            "init": True,
        }
        if dim is None:
            dim = list(self.get_initialized().isel(lead=0).dims)
        res = self._apply_climpred_function(
            compute_climatology,
            input_dict=input_dict,
            metric=metric,
            comparison=comparison,
            dim=dim,
            **metric_kwargs,
        )
        if self._temporally_smoothed:
            res = _reset_temporal_axis(res, self._temporally_smoothed, dim="lead")
            res["lead"].attrs = self.get_initialized().lead.attrs
        return res

    def bootstrap(
        self,
        metric=None,
        comparison=None,
        dim=None,
        reference=None,
        iterations=None,
        sig=95,
        pers_sig=None,
        **metric_kwargs,
    ):
        """Bootstrap with replacement according to Goddard et al. 2013.

        Args:
            metric (str, :py:class:`~climpred.metrics.Metric`): Metric to verify
                bootstrapped skill, see `metrics </metrics.html>`_.
            comparison (str, :py:class:`~climpred.comparisons.Comparison`): Comparison
                passed to verify, see `comparisons </comparisons.html>`_.
            dim (str, list of str): Dimension(s) over which to apply metric.
                ``dim`` is passed on to xskillscore.{metric} and includes xskillscore's
                ``member_dim``. ``dim`` should contain ``member`` when ``comparison``
                is probabilistic but should not contain ``member`` when
                ``comparison=e2c``. Defaults to ``None`` meaning that all dimensions
                other than ``lead`` are reduced.
            reference (str, list of str): Type of reference forecasts with which to
                verify. One or more of ['uninitialized', 'persistence', 'climatology'].
                If None or empty, returns no p value.
            iterations (int): Number of resampling iterations for bootstrapping with
                replacement. Recommended >= 500.
            sig (int, default 95): Significance level in percent for deciding whether
                uninitialized and persistence beat initialized skill.
            pers_sig (int): If not ``None``, the separate significance level for
                persistence. Defaults to ``None``, or the same significance as ``sig``.
            **metric_kwargs (optional): arguments passed to ``metric``.

        Returns:
            xr.Datasets: with dimensions ``results`` (holding ``verify skill``, ``p``,
            ``low_ci`` and ``high_ci``) and ``skill`` (holding ``initialized``,
            ``persistence`` and/or ``uninitialized``):
                * results='verify skill', skill='initialized':
                    mean initialized skill
                * results='high_ci', skill='initialized':
                    high confidence interval boundary for initialized skill
                * results='p', skill='uninitialized':
                    p value of the hypothesis that the
                    difference of skill between the initialized and
                    uninitialized simulations is smaller or equal to zero
                    based on bootstrapping with replacement.
                * results='p', skill='persistence':
                    p value of the hypothesis that the
                    difference of skill between the initialized and persistenceistence
                    simulations is smaller or equal to zero based on
                    bootstrapping with replacement.

        Reference:
            * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
              Gonzalez, V. Kharin, et al. “A Verification Framework for
              Interannual-to-Decadal Predictions Experiments.” Climate
              Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
              https://doi.org/10/f4jjvf.

        Example:
            Calculate the Pearson's Anomaly Correlation ('acc') comparing every member
            to every other member (``m2m``) reducing dimensions ``member`` and
            ``init`` 50 times after resampling ``member`` dimension with replacement.
            Also calculate reference skill for the ``persistence``, ``climatology``
            and ``uninitialized`` forecast and compare whether initialized skill is
            better than reference skill: Returns verify skill, probability that
            reference forecast performs better than initialized and the lower and
            upper bound of the resample.

            >>> PerfectModelEnsemble.bootstrap(metric='acc', comparison='m2m',
            ...     dim=['init', 'member'], iterations=50, resample_dim='member',
            ...     reference=['persistence', 'climatology' ,'uninitialized'])
            <xarray.Dataset>
            Dimensions:  (skill: 4, results: 4, lead: 20)
            Coordinates:
              * lead     (lead) int64 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
              * results  (results) <U12 'verify skill' 'p' 'low_ci' 'high_ci'
              * skill    (skill) <U13 'initialized' 'persistence' ... 'uninitialized'
            Data variables:
                tos      (skill, results, lead) float64 0.7941 0.7489 ... 0.1494 0.1466
            Attributes:
                prediction_skill:            calculated by climpred https://climpred.read...
                number_of_initializations:   12
                number_of_members:           10
                alignment:                   same_verifs
                metric:                      pearson_r
                comparison:                  m2m
                dim:                         ['init', 'member']
                units:                       None
                confidence_interval_levels:  0.975-0.025
                bootstrap_iterations:        50
                p:                           probability that reference performs better t...

        """
        if iterations is None:
            raise ValueError("Designate number of bootstrapping `iterations`.")
        reference = _check_valid_reference(reference)
        has_dataset(self._datasets["control"], "control", "iteration")
        input_dict = {
            "ensemble": self._datasets["initialized"],
            "control": self._datasets["control"],
            "init": True,
        }
        return self._apply_climpred_function(
            bootstrap_perfect_model,
            input_dict=input_dict,
            metric=metric,
            comparison=comparison,
            dim=dim,
            reference=reference,
            sig=sig,
            iterations=iterations,
            pers_sig=pers_sig,
            **metric_kwargs,
        )


class HindcastEnsemble(PredictionEnsemble):
    """An object for climate prediction ensembles initialized by a data-like
    product.

    `HindcastEnsemble` is a sub-class of `PredictionEnsemble`. It tracks a single
    verification dataset (i.e., observations) associated with the hindcast ensemble
    for easy computation across multiple variables.

    This object is built on `xarray` and thus requires the input object to
    be an `xarray` Dataset or DataArray.
    """

    def __init__(self, xobj):
        """Create a `HindcastEnsemble` object by inputting output from a
        prediction ensemble in `xarray` format.

        Args:
          xobj (xarray object):
            decadal prediction ensemble output.

        Attributes:
          observations: Dictionary of verification data to associate with the decadal
              prediction ensemble.
          uninitialized: Dictionary of companion (or bootstrapped)
              uninitialized ensemble run.
        """
        super().__init__(xobj)
        self._datasets.update({"observations": {}})
        self.kind = "hindcast"

    def _apply_climpred_function(self, func, init, **kwargs):
        """Helper function to loop through verification data and apply an arbitrary
        climpred function.

        Args:
            func (function): climpred function to apply to object.
            init (bool): Whether or not it's the initialized ensemble.
        """
        hind = self._datasets["initialized"]
        verif = self._datasets["observations"]
        drop_init, drop_obs = self._vars_to_drop(init=init)
        return func(hind.drop_vars(drop_init), verif.drop_vars(drop_obs), **kwargs)

    def _vars_to_drop(self, init=True):
        """Returns list of variables to drop when comparing
        initialized/uninitialized to observations.

        This is useful if the two products being compared do not share the same
        variables. I.e., if the observations have ['SST'] and the initialized has
        ['SST', 'SALT'], this will return a list with ['SALT'] to be dropped
        from the initialized.

        Args:
          init (bool, default True):
            If ``True``, check variables on the initialized.
            If ``False``, check variables on the uninitialized.

        Returns:
          Lists of variables to drop from the initialized/uninitialized
          and observational Datasets.
        """
        if init:
            init_vars = [var for var in self._datasets["initialized"].data_vars]
        else:
            init_vars = [var for var in self._datasets["uninitialized"].data_vars]
        obs_vars = [var for var in self._datasets["observations"].data_vars]
        # Make lists of variables to drop that aren't in common
        # with one another.
        init_vars_to_drop = list(set(init_vars) - set(obs_vars))
        obs_vars_to_drop = list(set(obs_vars) - set(init_vars))
        return init_vars_to_drop, obs_vars_to_drop

    @is_xarray(1)
    def add_observations(self, xobj):
        """Add verification data against which to verify the initialized ensemble.

        Args:
            xobj (xarray object): Dataset/DataArray to append to the
                ``HindcastEnsemble`` object.
        """
        if isinstance(xobj, xr.DataArray):
            xobj = xobj.to_dataset()
        match_initialized_dims(self._datasets["initialized"], xobj)
        match_initialized_vars(self._datasets["initialized"], xobj)
        # Check that time is int, cftime, or datetime; convert ints or cftime to
        # datetime.
        xobj = convert_time_index(xobj, "time", "xobj[init]")
        # Check that converted/original cftime calendar is the same as the
        # initialized calendar to avoid any alignment errors.
        match_calendars(self._datasets["initialized"], xobj)
        datasets = self._datasets.copy()
        datasets.update({"observations": xobj})
        return self._construct_direct(datasets, kind="hindcast")

    @is_xarray(1)
    def add_uninitialized(self, xobj):
        """Add a companion uninitialized ensemble for comparison to verification data.

        Args:
            xobj (xarray object): Dataset/DataArray of the uninitialzed
                                  ensemble.
        """
        if isinstance(xobj, xr.DataArray):
            xobj = xobj.to_dataset()
        match_initialized_dims(self._datasets["initialized"], xobj, uninitialized=True)
        match_initialized_vars(self._datasets["initialized"], xobj)
        # Check that init is int, cftime, or datetime; convert ints or cftime to
        # datetime.
        xobj = convert_time_index(xobj, "time", "xobj[init]")
        # Check that converted/original cftime calendar is the same as the
        # initialized calendar to avoid any alignment errors.
        match_calendars(self._datasets["initialized"], xobj, kind2="uninitialized")
        datasets = self._datasets.copy()
        datasets.update({"uninitialized": xobj})
        return self._construct_direct(datasets, kind="hindcast")

    def get_observations(self):
        """Returns xarray Datasets of the observations/verification data.

        Returns:
            ``xarray`` Dataset of observations.
        """
        return self._datasets["observations"]

    def verify(
        self,
        reference=None,
        metric=None,
        comparison=None,
        dim=None,
        alignment=None,
        **metric_kwargs,
    ):
        """Verifies the initialized ensemble against observations.

        .. note::
            This will automatically verify against all shared variables
            between the initialized ensemble and observations/verification data.

        Args:
            reference (str, list of str): Type of reference forecasts to also verify against the
                observations. Choose one or more of ['uninitialized', 'persistence', 'climatology'].
                Defaults to None.
            metric (str, :py:class:`~climpred.metrics.Metric`): Metric to apply for
                verification. see `metrics </metrics.html>`_.
            comparison (str, :py:class:`~climpred.comparisons.Comparison`): How to
                compare to the observations/verification data. See
                `comparisons </comparisons.html>`_.
            dim (str, list of str): Dimension(s) to apply metric over. ``dim`` is passed
                on to xskillscore.{metric} and includes xskillscore's ``member_dim``.
                ``dim`` should contain ``member`` when ``comparison`` is probabilistic
                but should not contain ``member`` when ``comparison=e2o``. Defaults to
                ``None`` meaning that all dimensions other than ``lead`` are reduced.
            alignment (str): which inits or verification times should be aligned?

                - 'maximize': maximize the degrees of freedom by slicing ``hind`` and
                  ``verif`` to a common time frame at each lead.

                - 'same_inits': slice to a common init frame prior to computing
                  metric. This philosophy follows the thought that each lead should be
                  based on the same set of initializations.

                - 'same_verif': slice to a common/consistent verification time frame
                  prior to computing metric. This philosophy follows the thought that
                  each lead should be based on the same set of verification dates.

            **metric_kwargs (optional): arguments passed to ``metric``.

        Returns:
            Dataset with dimension skill reduced by dim containing initialized and
            reference skill(s) if specified.

        Example:
            Root mean square error (``rmse``) comparing every member with the
            verification (``m2o``) over the same verification time (``same_verifs``)
            for all leads reducing dimensions ``init`` and ``member``:

            >>> HindcastEnsemble.verify(metric='rmse', comparison='m2o',
            ...     alignment='same_verifs', dim=['init','member'])
            <xarray.Dataset>
            Dimensions:  (lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
                skill    <U11 'initialized'
            Data variables:
                SST      (lead) float64 0.08516 0.09492 0.1041 ... 0.1525 0.1697 0.1785

            Pearson's Anomaly Correlation ('acc') comparing the ensemble mean with the
            verification (``e2o``) over the same initializations (``same_inits``) for
            all leads reducing dimension ``init`` while also calculating reference
            skill for the ``persistence``, ``climatology`` and ``uninitialized``
            forecast.

            >>> HindcastEnsemble.verify(metric='acc', comparison='e2o',
            ...     alignment='same_inits', dim='init',
            ...     reference=['persistence', 'climatology' ,'uninitialized'])
            <xarray.Dataset>
            Dimensions:  (skill: 4, lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
              * skill    (skill) <U13 'initialized' 'persistence' ... 'uninitialized'
            Data variables:
                SST      (skill, lead) float64 0.9023 0.8807 0.8955 ... 0.9078 0.9128 0.9159
        """
        # Have to do checks here since this doesn't call `compute_hindcast` directly.
        # Will be refactored when `climpred` migrates to inheritance-based.
        if dim is None:
            viable_dims = list(self.get_initialized().isel(lead=0).dims) + [[]]
            raise ValueError(
                "Designate a dimension to reduce over when applying the "
                f"metric. Got {dim}. Choose one or more of {viable_dims}"
            )
        if ("member" in dim) and comparison not in ["m2o", "m2r"]:
            raise ValueError(
                "Comparison must equal 'm2o' with dim='member'. "
                f"Got comparison {comparison}."
            )
        reference = _check_valid_reference(reference)

        def _verify(
            hind,
            verif,
            hist,
            reference,
            metric,
            comparison,
            alignment,
            dim,
            **metric_kwargs,
        ):
            """Interior verify func to be passed to apply func."""
            metric, comparison, dim = _get_metric_comparison_dim(
                hind, metric, comparison, dim, kind=self.kind
            )
            forecast, verif = comparison.function(hind, verif, metric=metric)
            if metric.name == "rps":  # modify metric_kwargs for rps
                metric_kwargs = broadcast_metric_kwargs_for_rps(
                    forecast, verif, metric_kwargs
                )
            forecast = forecast.rename({"init": "time"})
            inits, verif_dates = return_inits_and_verif_dates(
                forecast,
                verif,
                alignment,
                reference=reference,
                hist=hist,
            )
            metric_over_leads = [
                _apply_metric_at_given_lead(
                    verif,
                    verif_dates,
                    lead,
                    hind=forecast,
                    hist=hist,
                    inits=inits,
                    # Ensure apply metric function returns skill and not reference
                    # results.
                    reference=None,
                    metric=metric,
                    comparison=comparison,
                    dim=dim,
                    **metric_kwargs,
                )
                for lead in forecast["lead"].data
            ]
            result = xr.concat(metric_over_leads, dim="lead", **CONCAT_KWARGS)
            result["lead"] = forecast["lead"]

            if reference is not None:
                if "member" in verif.dims:  # if broadcasted before
                    verif = verif.isel(member=0)
                for r in reference:
                    metric_over_leads = [
                        _apply_metric_at_given_lead(
                            verif,
                            verif_dates,
                            lead,
                            hind=forecast,
                            hist=hist,
                            inits=inits,
                            reference=r,
                            metric=metric,
                            comparison=comparison,
                            dim=dim,
                            **metric_kwargs,
                        )
                        for lead in forecast["lead"].data
                    ]
                    ref = xr.concat(metric_over_leads, dim="lead", **CONCAT_KWARGS)
                    ref["lead"] = forecast["lead"]
                    # fix to get no member dim for uninitialized e2o skill #477
                    if (
                        r == "uninitialized"
                        and comparison.name == "e2o"
                        and "member" in ref.dims
                    ):
                        ref = ref.mean("member")
                    result = xr.concat([result, ref], dim="skill", **CONCAT_KWARGS)
            # rename back to 'init'
            if "time" in result.dims:
                result = result.rename({"time": "init"})
            # Add dimension/coordinate for different references.
            result = result.assign_coords(skill=["initialized"] + reference)
            return result.squeeze()

        has_dataset(
            self._datasets["observations"], "observational", "verify a forecast"
        )
        if "uninitialized" in reference:
            has_dataset(
                self._datasets["uninitialized"],
                "uninitialized",
                "compute an uninitialized reference forecast",
            )
            hist = self._datasets["uninitialized"]
        else:
            hist = None

        res = self._apply_climpred_function(
            _verify,
            init=True,
            metric=metric,
            comparison=comparison,
            alignment=alignment,
            dim=dim,
            hist=hist,
            reference=reference,
            **metric_kwargs,
        )
        if self._temporally_smoothed:
            res = _reset_temporal_axis(res, self._temporally_smoothed, dim="lead")
            res["lead"].attrs = self.get_initialized().lead.attrs
        return res

    def bootstrap(
        self,
        metric=None,
        comparison=None,
        dim=None,
        alignment=None,
        reference=None,
        iterations=None,
        sig=95,
        resample_dim="member",
        pers_sig=None,
        **metric_kwargs,
    ):
        """Bootstrap with replacement according to Goddard et al. 2013.

        Args:
            metric (str, :py:class:`~climpred.metrics.Metric`): Metric to apply for
                verification, see `metrics <metrics.html>`_.
            comparison (str, :py:class:`~climpred.comparisons.Comparison`): How to
                compare to the observations/verification data, see
                `comparisons </comparisons.html>`_.
            dim (str, list of str): dimension(s) to apply metric over. ``dim`` is passed
                on to xskillscore.{metric} and includes xskillscore's ``member_dim``.
                ``dim`` should contain ``member`` when ``comparison`` is probabilistic
                but should not contain ``member`` when ``comparison='e2o'``. Defaults to
                ``None`` meaning that all dimensions other than ``lead`` are reduced.
            reference (str, list of str): Type of reference forecasts with which to
                verify. One or more of ['uninitialized', 'persistence', 'climatology'].
                If None or empty, returns no p value.
            alignment (str): which inits or verification times should be aligned?

                - 'maximize': maximize the degrees of freedom by slicing ``init`` and
                  ``verif`` to a common time frame at each lead.

                - 'same_inits': slice to a common init frame prior to computing
                  metric. This philosophy follows the thought that each lead should be
                  based on the same set of initializations.

                - 'same_verif': slice to a common/consistent verification time frame
                  prior to computing metric. This philosophy follows the thought that
                  each lead should be based on the same set of verification dates.

            iterations (int): Number of resampling iterations for bootstrapping with
                replacement. Recommended >= 500.
            sig (int, default 95): Significance level in percent for deciding whether
                uninitialized and persistence beat initialized skill.
            resample_dim (str or list): dimension to resample from. default: 'member'.

                - 'member': select a different set of members from hind
                - 'init': select a different set of initializations from hind

            pers_sig (int, default None):
                If not None, the separate significance level for persistence.
            **metric_kwargs (optional): arguments passed to ``metric``.

        Returns:
            xr.Datasets: with dimensions ``results`` (holding ``skill``, ``p``,
            ``low_ci`` and ``high_ci``) and ``skill`` (holding ``initialized``,
            ``persistence`` and/or ``uninitialized``):
                * results='verify skill', skill='initialized':
                    mean initialized skill
                * results='high_ci', skill='initialized':
                    high confidence interval boundary for initialized skill
                * results='p', skill='uninitialized':
                    p value of the hypothesis that the
                    difference of skill between the initialized and
                    uninitialized simulations is smaller or equal to zero
                    based on bootstrapping with replacement.
                * results='p', skill='persistence':
                    p value of the hypothesis that the
                    difference of skill between the initialized and persistence
                    simulations is smaller or equal to zero based on
                    bootstrapping with replacement.

        Example:
            Calculate the Pearson's Anomaly Correlation ('acc') comparing the ensemble
            mean forecast to the verification (``e2o``) over the same verification
            times (``same_verifs``) for all leads reducing dimensions ``init`` 50
            times after resampling ``member`` dimension with replacement. Also
            calculate reference skill for the ``persistence``, ``climatology``
            and ``uninitialized`` forecast and compare whether initialized skill is
            better than reference skill: Returns verify skill, probability that
            reference forecast performs better than initialized and the lower and
            upper bound of the resample.

            >>> HindcastEnsemble.bootstrap(metric='acc', comparison='e2o',
            ...     dim='init', iterations=50, resample_dim='member',
            ...     alignment='same_verifs',
            ...     reference=['persistence', 'climatology' ,'uninitialized'])
            <xarray.Dataset>
            Dimensions:  (skill: 4, results: 4, lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
              * results  (results) <U12 'verify skill' 'p' 'low_ci' 'high_ci'
              * skill    (skill) <U13 'initialized' 'persistence' ... 'uninitialized'
            Data variables:
                SST      (skill, results, lead) float64 0.9313 0.9119 ... 0.8078 0.8078
            Attributes:
                prediction_skill:            calculated by climpred https://climpred.read...
                number_of_initializations:   61
                number_of_members:           10
                alignment:                   same_verifs
                metric:                      pearson_r
                comparison:                  e2o
                dim:                         ['init']
                units:                       None
                confidence_interval_levels:  0.975-0.025
                bootstrap_iterations:        50
                p:                           probability that reference performs better t...
        """
        if iterations is None:
            raise ValueError("Designate number of bootstrapping `iterations`.")
        # TODO: replace with more computationally efficient classes implementation
        reference = _check_valid_reference(reference)
        if "uninitialized" in reference and not isinstance(
            self.get_uninitialized(), xr.Dataset
        ):
            raise ValueError("reference uninitialized requires uninitialized.")
        return bootstrap_hindcast(
            self.get_initialized(),
            self.get_uninitialized()
            if isinstance(self.get_uninitialized(), xr.Dataset)
            else None,
            self.get_observations(),
            metric=metric,
            comparison=comparison,
            dim=dim,
            alignment=alignment,
            reference=reference,
            resample_dim=resample_dim,
            sig=sig,
            iterations=iterations,
            pers_sig=pers_sig,
            **metric_kwargs,
        )

    def remove_bias(
        self,
        alignment=None,
        how="additive_mean",
        train_test_split="unfair",
        train_init=None,
        train_time=None,
        cv=False,
        **metric_kwargs,
    ):
        """Calculate and remove bias from
        :py:class:`~climpred.classes.HindcastEnsemble`.
        Bias is grouped by ``seasonality`` set via :py:class:`~climpred.options.set_options`. When wrapping xclim.sbda.adjustment use ``group`` instead.

        Args:
            alignment (str): which inits or verification times should be aligned?

                - 'maximize': maximize the degrees of freedom by slicing ``hind`` and
                  ``verif`` to a common time frame at each lead.

                - 'same_inits': slice to a common init frame prior to computing
                  metric. This philosophy follows the thought that each lead should be
                  based on the same set of initializations.

                - 'same_verif': slice to a common/consistent verification time frame
                  prior to computing metric. This philosophy follows the thought that
                  each lead should be based on the same set of verification dates.

            how (str): what kind of bias removal to perform. Defaults to 'additive_mean'. Select from:

                - 'additive_mean': correcting the mean forecast additively
                - 'multiplicative_mean': correcting the mean forecast multiplicatively
                - 'multiplicative_std': correcting the standard deviation multiplicatively
                - 'modified_quantile': `Reference <https://www.sciencedirect.com/science/article/abs/pii/S0034425716302000?via%3Dihub>`_
                - 'basic_quantile': `Reference <https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/joc.2168>`_
                - 'gamma_mapping': `Reference <https://www.hydrol-earth-syst-sci.net/21/2649/2017/>`_
                - 'normal_mapping': `Reference <https://www.hydrol-earth-syst-sci.net/21/2649/2017/>`_
                - 'EmpiricalQuantileMapping': `Reference <https://xclim.readthedocs.io/en/stable/sdba_api.html#xclim.sdba.adjustment.EmpiricalQuantileMapping>`_
                - 'DetrendedQuantileMapping': `Reference <https://xclim.readthedocs.io/en/stable/sdba_api.html#xclim.sdba.adjustment.DetrendedQuantileMapping>`_
                - 'PrincipalComponents': `Reference <https://xclim.readthedocs.io/en/stable/sdba_api.html#xclim.sdba.adjustment.PrincipalComponents>`_
                - 'QuantileDeltaMapping': `Reference <https://xclim.readthedocs.io/en/stable/sdba_api.html#xclim.sdba.adjustment.QuantileDeltaMapping>`_
                - 'Scaling': `Reference <https://xclim.readthedocs.io/en/stable/sdba_api.html#xclim.sdba.adjustment.Scaling>`_
                - 'LOCI': `Reference <https://xclim.readthedocs.io/en/stable/sdba_api.html#xclim.sdba.adjustment.LOCI>`_

            train_test_split (str): How to separate train period to calculate the bias and test period to apply bias correction to? For a detailed description, see `Risbey et al. 2021 <http://www.nature.com/articles/s41467-021-23771-z>`_:

                - `fair`: no overlap between `train` and `test` (recommended).
                    Set either `train_init` or `train_time`.
                - `unfair`: completely overlapping `train` and `test`
                    (climpred default).
                - `unfair-cv`: overlapping `train` and `test` except for current
                    `init`, which is `left out <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation>`_
                    (set `cv='LOO'`).

            train_init (xr.DataArray, slice): Define initializations for training
                when ``alignment='same_inits/maximize'``.
            train_time (xr.DataArray, slice): Define time for training
                when ``alignment='same_verif'``.
            cv (bool or str): Only relevant when `train_test_split='unfair-cv'`. Defaults to False.

                - True/'LOO': Calculate bias by `leaving given initialization out <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation>`_
                - False: include all initializations in the calculation of bias, which
                    is much faster and but yields similar skill with a large N of
                    initializations.

            **metric_kwargs (dict): passed to ``xclim.sdba`` (including ``group``) or ``XBias_Correction``

        Returns:
            HindcastEnsemble: bias removed HindcastEnsemble.

        Example:

            Skill from raw model output without bias reduction:

            >>> HindcastEnsemble.verify(metric='rmse', comparison='e2o',
            ...     alignment='maximize', dim='init')
            <xarray.Dataset>
            Dimensions:  (lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
                skill    <U11 'initialized'
            Data variables:
                SST      (lead) float64 0.08359 0.08141 0.08362 ... 0.1361 0.1552 0.1664

            Note that this HindcastEnsemble is already bias reduced, therefore
            ``train_test_split='unfair'`` has hardly any effect. Use all
            initializations to calculate bias and verify skill:

            >>> HindcastEnsemble.remove_bias(alignment='maximize',
            ...     how='additive_mean', test_train_split='unfair'
            ... ).verify(metric='rmse', comparison='e2o', alignment='maximize',
            ... dim='init')
            <xarray.Dataset>
            Dimensions:  (lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
                skill    <U11 'initialized'
            Data variables:
                SST      (lead) float64 0.08349 0.08039 0.07522 ... 0.07305 0.08107 0.08255

            Separate initializations 1954 - 1980 to calculate bias. Note that
            this HindcastEnsemble is already bias reduced, therefore
            ``train_test_split='fair'`` worsens skill here. Generally,
            ``train_test_split='fair'`` is recommended to use for a fair
            comparison against real-time forecasts.

            >>> HindcastEnsemble.remove_bias(alignment='maximize',
            ...     how='additive_mean', train_test_split='fair',
            ...     train_init=slice('1954', '1980')).verify(metric='rmse',
            ...     comparison='e2o', alignment='maximize', dim='init')
            <xarray.Dataset>
            Dimensions:  (lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
                skill    <U11 'initialized'
            Data variables:
                SST      (lead) float64 0.132 0.1085 0.08722 ... 0.08209 0.08969 0.08732

            Wrapping methods ``how`` from `xclim <https://xclim.readthedocs.io/en/stable/sdba_api.html>`_ and providing ``group`` for ``groupby``:

            >>> HindcastEnsemble.remove_bias(alignment='same_init', group='init',
            ...     how='DetrendedQuantileMapping', train_test_split='unfair',
            ...     ).verify(metric='rmse',
            ...     comparison='e2o', alignment='maximize', dim='init')
            <xarray.Dataset>
            Dimensions:  (lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
                skill    <U11 'initialized'
            Data variables:
                SST      (lead) float64 0.09823 0.09747 0.08235 ... 0.07742 0.08115 0.08326

            Wrapping methods ``how`` from `bias_correction <https://github.com/pankajkarman/bias_correction/blob/master/bias_correction.py>`_:

            >>> HindcastEnsemble.remove_bias(alignment='same_init',
            ...     how='modified_quantile', train_test_split='unfair',
            ...     ).verify(metric='rmse',
            ...     comparison='e2o', alignment='maximize', dim='init')
            <xarray.Dataset>
            Dimensions:  (lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
                skill    <U11 'initialized'
            Data variables:
                SST      (lead) float64 0.07628 0.08293 0.08169 ... 0.1577 0.1821 0.2087
        """
        if train_test_split not in BIAS_CORRECTION_TRAIN_TEST_SPLIT_METHODS:
            raise NotImplementedError(
                f"train_test_split='{train_test_split}' not implemented. Please choose `train_test_split` from {BIAS_CORRECTION_TRAIN_TEST_SPLIT_METHODS}, see Risbey et al. 2021 http://www.nature.com/articles/s41467-021-23771-z for description and https://github.com/pangeo-data/climpred/issues/648 for implementation status."
            )

        alignment = _check_valud_alignment(alignment)

        if train_test_split in ["fair"]:
            if (
                (train_init is None)
                or not isinstance(train_init, (slice, xr.DataArray))
            ) and (alignment in ["same_inits", "maximize"]):
                raise ValueError(
                    f'When alignment="{alignment}", please provide `train_init` as xr.DataArray, e.g. `hindcast.coords["init"].slice(start, end)` or slice, e.g. `slice(start, end)`, got `train_init={train_init}`.'
                )
            if (
                (train_time is None)
                or not isinstance(train_time, (slice, xr.DataArray))
            ) and (alignment in ["same_verif"]):
                raise ValueError(
                    f'When alignment="{alignment}", please provide `train_time` as xr.DataArray, e.g. `hindcast.coords["time"].slice(start, end)` or slice, e.g. `slice(start, end)`, got `train_time={train_time}`'
                )

            if isinstance(train_init, slice):
                train_init = self.coords["init"].sel(init=train_init)
            if isinstance(train_time, slice):
                train_time = self.coords["time"].sel(time=train_time)

        if how == "mean":
            how = "additive_mean"  # backwards compatibility
        if how in ["additive_mean", "multiplicative_mean", "multiplicative_std"]:
            func = gaussian_bias_removal
        elif how in BIAS_CORRECTION_BIAS_CORRECTION_METHODS:
            func = bias_correction
        elif how in XCLIM_BIAS_CORRECTION_METHODS:
            func = xclim_sdba
        else:
            raise NotImplementedError(
                f"bias removal '{how}' is not implemented, please choose from {INTERNAL_BIAS_CORRECTION_METHODS+BIAS_CORRECTION_BIAS_CORRECTION_METHODS}."
            )

        if train_test_split in ["unfair-cv"]:
            if cv not in [True, "LOO"]:
                raise ValueError(
                    f"Please provide `cv='LOO'` when train_test_split='unfair-cv', found `cv='{cv}'`"
                )
            else:
                cv = "LOO"  # backward compatibility
            if cv not in CROSS_VALIDATE_METHODS:
                raise NotImplementedError(
                    f"cross validation method {cv} not implemented. Please choose cv from {CROSS_VALIDATE_METHODS}."
                )
            metric_kwargs["cv"] = cv

        self = func(
            self,
            alignment=alignment,
            how=how,
            train_test_split=train_test_split,
            train_init=train_init,
            train_time=train_time,
            **metric_kwargs,
        )
        return self
