"""Main module instantiating ``PerfectModelEnsemble`` and ``HindcastEnsemble."""

import warnings
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import cf_xarray  # noqa
import numpy as np
import xarray as xr
from IPython.display import display_html
from xarray.core.coordinates import DatasetCoordinates
from xarray.core.dataset import DataVariables
from xarray.core.formatting_html import dataset_repr
from xarray.core.options import OPTIONS as XR_OPTIONS
from xarray.core.utils import Frozen

from .alignment import return_inits_and_verif_dates
from .bias_removal import bias_correction, gaussian_bias_removal, xclim_sdba
from .bootstrap import (
    bootstrap_hindcast,
    bootstrap_perfect_model,
    bootstrap_uninit_pm_ensemble_from_control_cftime,
    resample_uninitialized_from_initialized,
)
from .checks import (
    _check_valid_alignment,
    _check_valid_reference,
    attach_long_names,
    attach_standard_names,
    has_dataset,
    has_dims,
    has_valid_lead_units,
    match_calendars,
    match_initialized_dims,
    match_initialized_vars,
    rename_to_climpred_dims,
)
from .comparisons import Comparison
from .constants import (
    BIAS_CORRECTION_BIAS_CORRECTION_METHODS,
    BIAS_CORRECTION_TRAIN_TEST_SPLIT_METHODS,
    CLIMPRED_DIMS,
    CONCAT_KWARGS,
    CROSS_VALIDATE_METHODS,
    INTERNAL_BIAS_CORRECTION_METHODS,
    XCLIM_BIAS_CORRECTION_METHODS,
)
from .exceptions import CoordinateError, DimensionError, VariableError
from .metrics import Metric
from .options import OPTIONS, set_options
from .prediction import (
    _apply_metric_at_given_lead,
    _get_metric_comparison_dim,
    compute_perfect_model,
)
from .reference import (
    compute_climatology,
    compute_persistence,
    compute_persistence_from_first_lead,
)
from .smoothing import (
    _reset_temporal_axis,
    smooth_goddard_2013,
    spatial_smoothing_xesmf,
    temporal_smoothing,
)
from .utils import (
    add_time_from_init_lead,
    assign_attrs,
    broadcast_metric_kwargs_for_rps,
    convert_time_index,
    convert_Timedelta_to_lead_units,
)

metricType = Union[str, Metric]
comparisonType = Union[str, Comparison]
dimType = Optional[Union[str, List[str]]]
alignmentType = str
referenceType = Union[List[str], str]
groupbyType = Optional[Union[str, xr.DataArray]]
metric_kwargsType = Optional[Any]

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    optionalaxisType = Optional[plt.Axes]
else:
    optionalaxisType = Optional[Any]


def _display_metadata(self) -> str:
    """
    Print the contents of the :py:class:`.PredictionEnsemble` as text.

    Example:
        >>> init = climpred.tutorial.load_dataset("CESM-DP-SST")
        >>> hindcast = climpred.HindcastEnsemble(init)
        >>> print(hindcast)
        <climpred.HindcastEnsemble>
        Initialized Ensemble:
            SST      (init, lead, member) float64 -0.2404 -0.2085 ... 0.7442 0.7384
        Observations:
            None
        Uninitialized:
            None

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


def _display_metadata_html(self) -> str:
    """Print contents of :py:class:`.PredictionEnsemble` as html."""
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
    The main object :py:class:`.PredictionEnsemble`.

    This is the super of both :py:class:`.PerfectModelEnsemble` and
    :py:class:`.HindcastEnsemble`. This cannot be called directly by
    a user, but should house functions that both ensemble types can use.

    Associated :py:class:`xarray.Dataset` are stored in:

        * ``PredictionEnsemble._datasets["initialized"]``
        * ``PredictionEnsemble._datasets["uninitialized"]``
        * ``PredictionEnsemble._datasets["control"]`` in
          :py:class:`.PerfectModelEnsemble`
        * ``PredictionEnsemble._datasets[observations"]`` in
          :py:class:`.HindcastEnsemble`

    """

    def __init__(self, initialized: Union[xr.DataArray, xr.Dataset]):
        """Create a :py:class:`.PredictionEnsemble` object."""
        if isinstance(initialized, xr.DataArray):
            # makes applying prediction functions easier, etc.
            initialized = initialized.to_dataset()
        initialized = rename_to_climpred_dims(initialized)
        has_dims(initialized, ["init", "lead"], "PredictionEnsemble")
        # Check that init is int, cftime, or datetime; convert ints or cftime to
        # datetime.
        initialized = convert_time_index(initialized, "init", "initialized[init]")
        # Put this after `convert_time_index` since it assigns 'years' attribute if the
        # `init` dimension is a `float` or `int`.
        initialized = convert_Timedelta_to_lead_units(initialized)
        has_valid_lead_units(initialized)
        initialized = add_time_from_init_lead(initialized)
        # add metadata
        initialized = attach_standard_names(initialized)
        initialized = attach_long_names(initialized)
        initialized = initialized.cf.add_canonical_attributes(
            verbose=False, override=True, skip="units"
        )
        del initialized.attrs["history"]  # better only delete xclim message or not?
        # Add initialized dictionary and reserve sub-dictionary for an uninitialized
        # run.
        self._datasets = {"initialized": initialized, "uninitialized": {}}
        self.kind = "prediction"
        self._temporally_smoothed: Optional[Dict[str, int]] = None
        self._is_annual_lead = None
        self._warn_if_chunked_along_init_member_time()

    def _groupby(self, call: str, groupby: Union[str, xr.DataArray], **kwargs: Any):
        """Help for verify/bootstrap(groupby="month")."""
        skill_group, group_label = [], []
        groupby_str = f"init.{groupby}" if isinstance(groupby, str) else groupby
        with set_options(warn_for_failed_PredictionEnsemble_xr_call=False):
            for group, hind_group in self.get_initialized().init.groupby(groupby_str):
                skill_group.append(
                    getattr(self.sel(init=hind_group), call)(
                        **kwargs,
                    )
                )
                group_label.append(group)
        new_dim_name = groupby if isinstance(groupby, str) else groupby_str.name
        skill_group = xr.concat(skill_group, new_dim_name).assign_coords(
            {new_dim_name: group_label}
        )
        skill_group[new_dim_name] = skill_group[new_dim_name].assign_attrs(  # type: ignore # noqa: E501
            {
                "description": "new dimension showing skill grouped by init.{groupby}"
                " created by .verify(groupby) or .bootstrap(groupby)"
            }
        )
        return skill_group

    @property
    def coords(self) -> DatasetCoordinates:
        """Return coordinates of :py:class:`.PredictionEnsemble`.

        Dictionary of :py:class:`xarray.DataArray` objects corresponding to coordinate
        variables available in all PredictionEnsemble._datasets.

        See also:
            :py:meth:`~xarray.Dataset.coords`
        """
        pe_coords = self.get_initialized().coords.to_dataset()
        for ds in self._datasets.values():
            if isinstance(ds, xr.Dataset):
                pe_coords.update(ds.coords.to_dataset())
        return pe_coords.coords

    @property
    def nbytes(self) -> int:
        """Bytes sizes of all PredictionEnsemble._datasets.

        See also:
            :py:meth:`~xarray.Dataset.nbytes`
        """
        return sum(
            [
                sum(v.nbytes for v in ds.variables.values())
                for ds in self._datasets.values()
                if isinstance(ds, xr.Dataset)
            ]
        )

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """
        Return sizes of :py:class:`.PredictionEnsemble`.

        Mapping from dimension names to lengths for all PredictionEnsemble._datasets.

        See also:
            :py:meth:`~xarray.Dataset.equals`
        """
        pe_dims = dict(self.get_initialized().dims)
        for ds in self._datasets.values():
            if isinstance(ds, xr.Dataset):
                pe_dims.update(dict(ds.dims))
        return pe_dims

    @property
    def dims(self) -> Mapping[Hashable, int]:
        """
        Return dimension of :py:class:`.PredictionEnsemble`.

        Mapping from dimension names to lengths all PredictionEnsemble._datasets.

        See also:
            :py:meth:`~xarray.Dataset.dims`
        """
        return Frozen(self.sizes)

    @property
    def chunks(self) -> Mapping[Hashable, Tuple[int, ...]]:
        """
        Return chunks of :py:class:`.PredictionEnsemble`.

        Mapping from chunks all PredictionEnsemble._datasets.

        See also:
            :py:meth:`~xarray.Dataset.chunks`
        """
        pe_chunks = dict(self.get_initialized().chunks)
        for ds in self._datasets.values():
            if isinstance(ds, xr.Dataset):
                for d in ds.chunks:
                    if d not in pe_chunks:
                        pe_chunks.update({d: ds.chunks[d]})
        return Frozen(pe_chunks)

    @property
    def chunksizes(self) -> Mapping[Hashable, Tuple[int, ...]]:
        """Return chunksizes of :py:class:`.PredictionEnsemble`.

        Mapping from dimension names to block lengths for this dataset's data, or
        None if the underlying data is not a dask array.
        Cannot be modified directly, but can be modified by calling ``.chunk()``.
        Same as :py:meth:`~xarray.Dataset.chunks`.

        See also:
            :py:meth:`~xarray.Dataset.chunksizes`
        """
        return self.chunks

    @property
    def data_vars(self) -> DataVariables:
        """
        Return data variables of :py:class:`.PredictionEnsemble`.

        Dictionary of DataArray objects corresponding to data variables available in
        all PredictionEnsemble._datasets.

        See also:
            :py:meth:`~xarray.Dataset.data_vars`
        """
        varset = set(self.get_initialized().data_vars)
        for ds in self._datasets.values():
            if isinstance(ds, xr.Dataset):
                # take union
                varset = varset & set(ds.data_vars)
        varlist = list(varset)
        return self.get_initialized()[varlist].data_vars

    # when you just print it interactively
    # https://stackoverflow.com/questions/1535327/how-to-print-objects-of-class-using-print
    def __repr__(self) -> str:
        """Return for print(PredictionEnsemble)."""
        if XR_OPTIONS["display_style"] == "html":
            return _display_metadata_html(self)
        else:
            return _display_metadata(self)

    def __len__(self) -> int:
        """Return number of all variables :py:class:`.PredictionEnsemble`."""
        return len(self.data_vars)

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over underlying :py:class:`xarray.Dataset`."""
        return iter(self._datasets.values())

    def __delitem__(self, key: Hashable) -> None:
        """Remove a variable from :py:class:`.PredictionEnsemble`."""
        del self._datasets["initialized"][key]
        for ds in self._datasets.values():
            if isinstance(ds, xr.Dataset):
                if key in ds.data_vars:
                    del ds[key]

    def __contains__(self, key: Hashable) -> bool:
        """Check variable in :py:class:`.PredictionEnsemble`.

        The ``"in"`` operator will return true or false depending on whether
        ``"key"`` is in any PredictionEnsemble._datasets.
        """
        contained = True
        for ds in self._datasets.values():
            if isinstance(ds, xr.Dataset):
                if key not in ds.data_vars:
                    contained = False
        return contained

    def equals(self, other: Union["PredictionEnsemble", Any]) -> bool:
        """Check if :py:class:`.PredictionEnsemble` is equal to other.

        Two :py:class:`.PredictionEnsemble` are equal if they have
        matching variables and coordinates, all of which are equal.
        ``PredictionEnsembles`` can still be equal (like pandas objects) if they have
        NaN values in the same locations.
        This method is necessary because `v1 == v2` for ``PredictionEnsembles``
        does element-wise comparisons (like numpy.ndarrays).

        See also:
            :py:meth:`~xarray.Dataset.equals`
        """
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

    def identical(self, other: Union["PredictionEnsemble", Any]) -> bool:
        """
        Check if :py:class:`.PredictionEnsemble` is identical to other.

        Like ``equals``, but also checks all dataset attributes and the
        attributes on all variables and coordinates.

        See also:
            :py:meth:`~xarray.Dataset.identical`
        """
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

    def plot(
        self,
        variable: Optional[str] = None,
        ax: optionalaxisType = None,
        show_members: bool = False,
        cmap: Optional[str] = None,
        x: str = "time",
    ) -> "plt.Axes":
        """Plot datasets from :py:class:`.PredictionEnsemble`.

        Wraps :py:func:`.climpred.graphics.plot_ensemble_perfect_model` or
        :py:func:`.climpred.graphics.plot_lead_timeseries_hindcast`.

        Args:
            variable: `variable` to show. Defaults to first in data_vars.
            ax: Axis to use in plotting. By default, creates a new axis.
            show_members: whether to display all members individually.
                Defaults to False.
            cmap: Name of matplotlib-recognized colorbar. Defaults to ``viridis``
                for :py:class:`.HindcastEnsemble`
                and ``tab10`` for :py:class:`.PerfectModelEnsemble`.
            x: Name of x-axis. Use ``time`` to show observations and
                hindcasts in real time. Use ``init`` to see hindcasts as
                initializations. For ``x=init`` only initialized is shown and only
                works for :py:class:`.HindcastEnsemble`.

        .. note::
            Alternatively inspect initialized datasets by
            ``PredictionEnsemble.get_initialized()[v].plot.line(x=time)``
            to see ``validtime`` on x-axis or
            ``PredictionEnsemble.get_initialized()[v].plot.line(x=init)``
            to see ``init`` on x-axis.

        Returns:
            ax: plt.axes

        """
        from .graphics import plot_ensemble_perfect_model, plot_lead_timeseries_hindcast

        if x == "time":
            x = "valid_time"
        assert x in ["valid_time", "init"]
        if isinstance(self, HindcastEnsemble):
            if cmap is None:
                cmap = "viridis"
            return plot_lead_timeseries_hindcast(
                self,
                variable=variable,
                ax=ax,
                show_members=show_members,
                cmap=cmap,
                x=x,
            )
        elif isinstance(self, PerfectModelEnsemble):
            if cmap is None:
                cmap = "tab10"
            return plot_ensemble_perfect_model(
                self, variable=variable, ax=ax, show_members=show_members, cmap=cmap
            )

    mathType = Union[int, float, np.ndarray, xr.DataArray, xr.Dataset]

    def _math(
        self,
        other: mathType,
        operator: str,
    ):
        """Help function for __add__, __sub__, __mul__, __truediv__.

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
            if not set(other.dims).issubset(self._datasets["initialized"].dims):  # type: ignore # noqa: E501
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

        _operator = eval(operator)

        if isinstance(other, PredictionEnsemble):
            # Create temporary copy to modify to avoid inplace operation.
            datasets = self._datasets.copy()
            for dataset in datasets:
                # Some pre-allocated entries might be empty, such as 'uninitialized'
                if isinstance(other._datasets[dataset], xr.Dataset) and isinstance(
                    self._datasets[dataset], xr.Dataset
                ):
                    datasets[dataset] = _operator(
                        datasets[dataset], other._datasets[dataset]
                    )
            return self._construct_direct(datasets, kind=self.kind)
        else:
            return self._apply_func(_operator, other)

    def __add__(self, other: mathType) -> "PredictionEnsemble":
        """Add."""
        return self._math(other, operator="add")

    def __sub__(self, other: mathType) -> "PredictionEnsemble":
        """Sub."""
        return self._math(other, operator="sub")

    def __mul__(self, other: mathType) -> "PredictionEnsemble":
        """Mul."""
        return self._math(other, operator="mul")

    def __truediv__(self, other: mathType) -> "PredictionEnsemble":
        """Div."""
        return self._math(other, operator="div")

    def __getitem__(self, varlist: Union[str, List[str]]) -> "PredictionEnsemble":
        """Allow subsetting variable(s) from

        Allow subsetting variable(s) from
        :py:class:`.PredictionEnsemble` as from
        :py:class:`xarray.Dataset`.

        Args:
            varlist: list of names or name of data variable(s) to subselect
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

    def __getattr__(
        self, name: str
    ) -> Callable:  # -> Callable[[VarArg(Any), KwArg(Any)], Any]
        """Allow for ``xarray`` methods to be applied to our prediction objects.

        Args:
            * name: str of xarray function, e.g., ``.isel()`` or ``.sum()``.
        """

        def wrapper(*args, **kwargs):
            """Apply arbitrary function to all datasets in ``PerfectModelEnsemble``.

            Got this from: https://stackoverflow.com/questions/41919499/
            how-to-call-undefined-methods-sequentially-in-python-class
            """

            def _apply_xr_func(v, name, *args, **kwargs):
                """Handle exceptions in our dictionary comprehension.

                In other words, this will skip applying the arbitrary function
                to a sub-dataset if a ValueError is thrown. This specifically
                targets cases where certain datasets don't have the given
                dim that's being called. E.g., ``.isel(lead=0)`` should only
                be applied to the initialized dataset.

                References:
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
                            msg = f"{func_name}({args[1:]}, {kwargs}) failed\n{error_type}: {e}"  # noqa: E501
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
                                    f"Error due to verification/control/uninitialized: {msg}"  # noqa: E501
                                )
                    else:
                        if OPTIONS["warn_for_failed_PredictionEnsemble_xr_call"]:
                            warnings.warn(msg)
                    return v

            return self._apply_func(_apply_xr_func, name, *args, **kwargs)

        return wrapper

    @classmethod
    def _construct_direct(cls, datasets, kind):
        """Shortcut around __init__ for internal use to avoid inplace operations.

        Pulled from xarrray Dataset class.
        https://github.com/pydata/xarray/blob/master/xarray/core/dataset.py
        """
        obj = object.__new__(cls)
        obj._datasets = datasets
        obj.kind = kind
        obj._warn_if_chunked_along_init_member_time()
        return obj

    def _apply_func(
        self, func: Callable[..., xr.Dataset], *args: Any, **kwargs: Any
    ) -> "PredictionEnsemble":
        """Apply a function to all datasets in a ``PerfectModelEnsemble``."""
        # Create temporary copy to modify to avoid inplace operation.
        # isnt that essentially the same as .map(func)?
        datasets = self._datasets.copy()

        # More explicit than nested dictionary comprehension.
        for key, ds in datasets.items():
            # If ds is xr.Dataset, apply the function directly to it
            # else, e.g. for {} ignore
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

    def get_initialized(self) -> xr.Dataset:
        """Return the :py:class:`xarray.Dataset` for the initialized ensemble."""
        return self._datasets["initialized"]

    def get_uninitialized(self) -> xr.Dataset:
        """Return the :py:class:`xarray.Dataset` for the uninitialized ensemble."""
        return self._datasets["uninitialized"]

    def smooth(
        self,
        smooth_kws: Optional[Union[str, Dict[str, int]]] = None,
        how: str = "mean",
        **xesmf_kwargs: str,
    ):
        """Smooth in space and/or aggregate in time in ``PredictionEnsemble``.

        Args:
            smooth_kws: Dictionary to specify the dims to
                smooth compatible with
                :py:func:`~climpred.smoothing.spatial_smoothing_xesmf` or
                :py:func:`~climpred.smoothing.temporal_smoothing`.
                Shortcut for :cite:t:`Goddard2013` ``goddard2013``.
                Defaults to ``None``.
            how: how to smooth temporally. From Choose from ``["mean", "sum"]``.
                Defaults to ``"mean"``.
            **xesmf_kwargs: kwargs passed to
                :py:func:`~climpred.smoothing.spatial_smoothing_xesmf`

        Examples:
            >>> PerfectModelEnsemble.get_initialized().lead.size
            20
            >>> PerfectModelEnsemble.smooth(
            ...     {"lead": 4}, how="sum"
            ... ).get_initialized().lead.size
            17

            >>> HindcastEnsemble_3D.smooth({"lon": 1, "lat": 1})
            <climpred.HindcastEnsemble>
            Initialized Ensemble:
                SST      (init, lead, lat, lon) float32 -0.3236 -0.3161 -0.3083 ... 0.0 0.0
            Observations:
                SST      (time, lat, lon) float32 0.002937 0.001561 0.002587 ... 0.0 0.0 0.0
            Uninitialized:
                None

            ``smooth`` simultaneously aggregates spatially listening to ``lon`` and
            ``lat`` and temporally listening to ``lead`` or ``time``.

            >>> HindcastEnsemble_3D.smooth(
            ...     {"lead": 2, "lat": 5, "lon": 4}
            ... ).get_initialized().coords
            Coordinates:
              * init        (init) object 1954-01-01 00:00:00 ... 2017-01-01 00:00:00
              * lead        (lead) int32 1 2 3 4 5 6 7 8 9
              * lon         (lon) float64 250.8 254.8 258.8 262.8
              * lat         (lat) float64 -9.75 -4.75
                valid_time  (lead, init) object 1955-01-01 00:00:00 ... 2026-01-01 00:00:00
            >>> HindcastEnsemble_3D.smooth("goddard2013").get_initialized().coords
            Coordinates:
              * init        (init) object 1954-01-01 00:00:00 ... 2017-01-01 00:00:00
              * lead        (lead) int32 1 2 3 4 5 6 7
              * lon         (lon) float64 250.8 255.8 260.8 265.8
              * lat         (lat) float64 -9.75 -4.75
                valid_time  (lead, init) object 1955-01-01 00:00:00 ... 2024-01-01 00:00:00


        """
        if not smooth_kws:
            return self
        tsmooth_kws: Optional[Union[str, Dict[str, int]]] = None
        d_lon_lat_kws: Optional[Union[str, Dict[str, int]]] = None
        smooth_fct: Callable[..., xr.Dataset]
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
            # recalc valid_time
            del self._datasets["initialized"].coords["valid_time"]
            self._datasets["initialized"] = add_time_from_init_lead(
                self._datasets["initialized"]
            )
        return self

    def remove_seasonality(
        self, seasonality: Union[None, str] = None
    ) -> "PredictionEnsemble":
        """Remove seasonal cycle from :py:class:`.PredictionEnsemble`.

        Args:
            seasonality: Seasonality to be removed. Choose from:
                ``["season", "month", "weekofyear", "dayofyear"]``.
                Defaults to ``OPTIONS["seasonality"]``.

        Examples:
            >>> HindcastEnsemble
            <climpred.HindcastEnsemble>
            Initialized Ensemble:
                SST      (init, lead, member) float64 -0.2392 -0.2203 ... 0.618 0.6136
            Observations:
                SST      (time) float32 -0.4015 -0.3524 -0.1851 ... 0.2481 0.346 0.4502
            Uninitialized:
                SST      (time, member) float64 -0.1969 -0.01221 -0.275 ... 0.4179 0.3974
            >>> # example already effectively without seasonal cycle
            >>> HindcastEnsemble.remove_seasonality(seasonality="month")
            <climpred.HindcastEnsemble>
            Initialized Ensemble:
                SST      (init, lead, member) float64 -0.2349 -0.216 ... 0.6476 0.6433
            Observations:
                SST      (time) float32 -0.3739 -0.3248 -0.1575 ... 0.2757 0.3736 0.4778
            Uninitialized:
                SST      (time, member) float64 -0.1789 0.005732 -0.257 ... 0.4359 0.4154
        """

        def _remove_seasonality(ds, initialized_dim="init", seasonality=None):
            """Remove the seasonal cycle from the data."""
            if ds is {}:
                return {}
            if seasonality is None:
                seasonality = OPTIONS["seasonality"]
            dim = initialized_dim if initialized_dim in ds.dims else "time"
            groupby = f"{dim}.{seasonality}"
            if "member" in ds.dims:
                clim = ds.mean("member").groupby(groupby).mean()
            else:
                clim = ds.groupby(groupby).mean()
            anom = ds.groupby(groupby) - clim
            return anom

        return self.map(
            _remove_seasonality,
            seasonality=seasonality,
        )

    def _warn_if_chunked_along_init_member_time(self) -> None:
        """
        Warn when ``CLIMPRED_DIMS`` except ``lead`` are wrongly chunked.

        When more than one chunk to show how to circumvent ``xskillscore`` chunking
        ``ValueError``.
        """
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
            msg = (
                f"{name} is chunked along dimensions {suggest_one_chunk} with more "
                f"than one chunk. `{name}.chunks={self.chunks}`.\nYou cannot call "
                f"`{name}.verify` or `{name}.bootstrap` in combination with any of "
                f" {suggest_one_chunk_time_to_init} passed as `dim`. In order to do "
                f"so, please rechunk {suggest_one_chunk} with `{name}.chunk("
                "{{dim:-1}}).verify(dim=dim).`\nIf you do not want to use dimensions "
                f" {suggest_one_chunk_time_to_init} in `{name}.verify(dim=dim)`, you "
                "can disregard this warning."
            )
            # chunk lead:1 in HindcastEnsemble
            if self.kind == "hindcast":
                msg += '\nIn `HindcastEnsemble` you may also create one chunk per "\
                " lead, as the `climpred` internally loops over lead, e.g. "\
                " `.chunk({{"lead": 1}}).verify().`'
            # chunk auto on non-climpred dims
            ndims = list(self.sizes)
            for d in CLIMPRED_DIMS:
                if d in ndims:
                    ndims.remove(d)
            if len(ndims) > 0:
                msg += (
                    f"\nConsider chunking embarassingly parallel dimensions such as "
                    f"{ndims} automatically, i.e. "
                    f'`{name}.chunk({ndims[0]}="auto").verify(...).'
                )
            warnings.warn(msg)


class PerfectModelEnsemble(PredictionEnsemble):
    """An object for "perfect model" prediction ensembles.

    :py:class:`.PerfectModelEnsemble` is a sub-class of
    :py:class:`.PredictionEnsemble`. It tracks
    the control run used to initialize the ensemble for easy computations,
    bootstrapping, etc.

    This object is built on ``xarray`` and thus requires the input object to
    be an :py:class:`xarray.Dataset` or :py:class:`xarray.DataArray`.
    """

    def __init__(self, initialized: Union[xr.DataArray, xr.Dataset]) -> None:
        """Create a :py:class:`.PerfectModelEnsemble` object.

        Args:
          initialized: prediction ensemble output.

        Attributes:
            control: datasets dictionary item of control simulation associated with the
                initialized ensemble.
            uninitialized: datasets dictionary item of uninitialized forecast.
        """
        super().__init__(initialized)
        # Reserve sub-dictionary for the control simulation.
        self._datasets.update({"control": {}})
        self.kind = "perfect"

    def _apply_climpred_function(
        self,
        func: Callable[..., xr.Dataset],
        input_dict: Dict[str, Any],
        **kwargs: Any,
    ) -> Union["PerfectModelEnsemble", xr.Dataset]:
        """Loop through observations and apply an arbitrary climpred function.

        Args:
            func: climpred function to apply to object.
            input_dict: dictionary with the following things:
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

    def _vars_to_drop(self, init: bool = True) -> Tuple[List[str], List[str]]:
        """Return list of variables to drop when comparing datasets.

        This is useful if the two products being compared do not share the same
        variables. I.e., if the control has ["SST"] and the initialized has
        ["SST", "SALT"], this will return a list with ["SALT"] to be dropped
        from the initialized.

        Args:
            init:
                If ``True``, check variables on the initialized.
                If ``False``, check variables on the uninitialized.

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

    def add_control(
        self, control: Union[xr.DataArray, xr.Dataset]
    ) -> "PerfectModelEnsemble":
        """Add the control run that initialized the prediction ensemble.

        Args:
            control: control run.
        """
        # NOTE: These should all be decorators.
        if isinstance(control, xr.DataArray):
            control = control.to_dataset()
        match_initialized_dims(self._datasets["initialized"], control)
        match_initialized_vars(self._datasets["initialized"], control)
        # Check that init is int, cftime, or datetime; convert ints or cftime to
        # datetime.
        control = convert_time_index(control, "time", "control[time]")
        # Check that converted/original cftime calendar is the same as the
        # initialized calendar to avoid any alignment errors.
        match_calendars(self._datasets["initialized"], control, kind2="control")
        datasets = self._datasets.copy()
        datasets.update({"control": control})
        return self._construct_direct(datasets, kind="perfect")

    def generate_uninitialized(self) -> "PerfectModelEnsemble":
        """Generate an uninitialized ensemble by resampling from the control simulation.

        Returns:
            ``uninitialzed`` resampled from ``control`` added
            to :py:class:`.PerfectModelEnsemble`
        """
        has_dataset(
            self._datasets["control"], "control", "generate an uninitialized ensemble."
        )

        uninit = bootstrap_uninit_pm_ensemble_from_control_cftime(
            self._datasets["initialized"], self._datasets["control"]
        )
        uninit.coords["valid_time"] = self.get_initialized().coords["valid_time"]
        datasets = self._datasets.copy()
        datasets.update({"uninitialized": uninit})
        return self._construct_direct(datasets, kind="perfect")

    def get_control(self) -> xr.Dataset:
        """Return the control as an :py:class:`xarray.Dataset`."""
        return self._datasets["control"]

    def verify(
        self,
        metric: metricType = None,
        comparison: comparisonType = None,
        dim: dimType = None,
        reference: referenceType = None,
        groupby: groupbyType = None,
        **metric_kwargs: metric_kwargsType,
    ) -> xr.Dataset:
        """Verify initialized predictions against a configuration of its members.

        .. note::
            The configuration of the other ensemble members is based off of the
            ``comparison`` keyword argument.

        Args:
            metric: Metric to apply for verification, see `metrics <../metrics.html>`_
            comparison: How to compare the initialized prediction ensemble with itself,
                see `comparisons <../comparisons.html>`_.
            dim: Dimension(s) over which to apply ``metric``.
                ``dim`` is passed on to xskillscore.{metric} and includes xskillscore's
                ``member_dim``. ``dim`` should contain ``member`` when ``comparison``
                is probabilistic but should not contain ``member`` when
                ``comparison=e2c``. Defaults to ``None`` meaning that all dimensions
                other than ``lead`` are reduced.
            reference: Type of reference forecasts with which to verify against.
                One or more of ``["uninitialized", "persistence", "climatology"]``.
                Defaults to ``None`` meaning no reference.
                For ``persistence``, choose between
                ``set_options(PerfectModel_persistence_from_initialized_lead_0)=False``
                (default) using :py:func:`~climpred.reference.compute_persistence` or
                ``set_options(PerfectModel_persistence_from_initialized_lead_0)=True``
                using
                :py:func:`~climpred.reference.compute_persistence_from_first_lead`.
            groupby: group ``init`` before passing ``initialized`` to ``verify``.
            **metric_kwargs: Arguments passed to ``metric``.

        Returns:
            ``initialized`` and ``reference`` forecast skill reduced by dimensions
            ``dim``

        Example:
            Root mean square error (``rmse``) comparing every member with the
            ensemble mean forecast (``m2e``) for all leads reducing dimensions
            ``init`` and ``member``:

            >>> PerfectModelEnsemble.verify(
            ...     metric="rmse", comparison="m2e", dim=["init", "member"]
            ... )
            <xarray.Dataset>
            Dimensions:  (lead: 20)
            Coordinates:
              * lead     (lead) int64 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
            Data variables:
                tos      (lead) float32 0.1028 0.1249 0.1443 0.1707 ... 0.2113 0.2452 0.2297
            Attributes:
                prediction_skill_software:     climpred https://climpred.readthedocs.io/
                skill_calculated_by_function:  PerfectModelEnsemble.verify()
                number_of_initializations:     12
                number_of_members:             10
                metric:                        rmse
                comparison:                    m2e
                dim:                           ['init', 'member']
                reference:                     []


            Continuous Ranked Probability Score (``"crps"``) comparing every member to every
            other member (``"m2m"``) reducing dimensions ``member`` and ``init`` while
            also calculating reference skill for the ``persistence``, ``climatology``
            and ``uninitialized`` forecast.

            >>> PerfectModelEnsemble.verify(
            ...     metric="crps",
            ...     comparison="m2m",
            ...     dim=["init", "member"],
            ...     reference=["persistence", "climatology", "uninitialized"],
            ... )
            <xarray.Dataset>
            Dimensions:  (skill: 4, lead: 20)
            Coordinates:
              * lead     (lead) int64 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
              * skill    (skill) <U13 'initialized' 'persistence' ... 'uninitialized'
            Data variables:
                tos      (skill, lead) float64 0.0621 0.07352 0.08678 ... 0.1188 0.09737
            Attributes:
                prediction_skill_software:                         climpred https://climp...
                skill_calculated_by_function:                      PerfectModelEnsemble.v...
                number_of_initializations:                         12
                number_of_members:                                 10
                metric:                                            crps
                comparison:                                        m2m
                dim:                                               ['init', 'member']
                reference:                                         ['persistence', 'clima...
                PerfectModel_persistence_from_initialized_lead_0:  False
        """
        if groupby is not None:
            return self._groupby(
                "verify",
                groupby,
                reference=reference,
                metric=metric,
                comparison=comparison,
                dim=dim,
                **metric_kwargs,
            )

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
                if (
                    not OPTIONS["PerfectModel_persistence_from_initialized_lead_0"]
                    and r != "persistence"
                ):
                    ref_compute_kwargs["comparison"] = comparison
                ref = getattr(self, f"_compute_{r}")(**ref_compute_kwargs)
                result = xr.concat([result, ref], dim="skill", **CONCAT_KWARGS)
            result = result.assign_coords(skill=["initialized"] + reference)
        result = assign_attrs(
            result,
            self.get_initialized(),
            function_name="PerfectModelEnsemble.verify()",
            metric=metric,
            comparison=comparison,
            dim=dim,
            reference=reference,
            **metric_kwargs,
        )
        return result.squeeze()

    def _compute_uninitialized(
        self,
        metric: metricType = None,
        comparison: comparisonType = None,
        dim: dimType = None,
        **metric_kwargs: metric_kwargsType,
    ) -> xr.Dataset:
        """Verify the bootstrapped uninitialized run against itself.

        .. note::
            The configuration of the other ensemble members is based off of the
            ``comparison`` keyword argument.

        Args:
            metric: Metric to apply for verification, see `metrics <../metrics.html>`_
            comparison: How to compare the uninitialized against itself, see
                `comparisons <../comparisons.html>`_.
            dim: Dimension(s) over which to apply metric.
                ``dim`` is passed on to xskillscore.{metric} and includes xskillscore's
                ``member_dim``. ``dim`` should contain ``member`` when ``comparison``
                is probabilistic but should not contain ``member`` when
                ``comparison=e2c``. Defaults to ``None``, meaning that all dimensions
                other than ``lead`` are reduced.
            **metric_kwargs: Arguments passed to ``metric``.

        Returns:
            ``initialized`` and ``reference`` forecast skill reduced by dimensions
            ``dim``
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

    def _compute_persistence(
        self,
        metric: metricType = None,
        dim: dimType = None,
        **metric_kwargs: metric_kwargsType,
    ):
        """Verify a simple persistence forecast of the control run against itself.

        Note: uses :py:func:`~climpred.reference.compute_persistence_from_first_lead`
        if ``set_options(PerfectModel_persistence_from_initialized_lead_0=True)`` else
        :py:func:`~climpred.reference.compute_persistence`.

        Args:
            metric: Metric to apply for verification, see `metrics <../metrics.html>`_
            comparison: How to compare the persistence against itself, see
                `comparisons <../comparisons.html>`_. Only valid if
                ``PerfectModel_persistence_from_initialized_lead_0=True``.
            dim: Dimension(s) over which to apply metric.
                ``dim`` is passed on to xskillscore.{metric} and includes xskillscore's
                ``member_dim``. ``dim`` should contain ``member`` when ``comparison``
                is probabilistic but should not contain ``member`` when
                ``comparison=e2c``. Defaults to ``None``, meaning that all dimensions
                other than ``lead`` are reduced.
            **metric_kwargs: Arguments passed to ``metric``.

        Returns:
            persistence forecast skill.

        References:
            * Chapter 8 (Short-Term Climate Prediction) in
              Van den Dool, Huug. Empirical methods in short-term climate
              prediction. Oxford University Press, 2007.
        """
        if dim is None:
            dim = list(self._datasets["initialized"].isel(lead=0).dims)
        compute_persistence_func: Callable[..., xr.Dataset]
        compute_persistence_func = compute_persistence_from_first_lead
        if OPTIONS["PerfectModel_persistence_from_initialized_lead_0"]:
            compute_persistence_func = compute_persistence_from_first_lead
            if self.get_initialized().lead[0] != 0:
                if OPTIONS["warn_for_failed_PredictionEnsemble_xr_call"]:
                    warnings.warn(
                        "Calculate persistence from "
                        f"lead={int(self.get_initialized().lead[0].values)} instead "
                        "of lead=0 (recommended)."
                    )
        else:
            compute_persistence_func = compute_persistence
            if self._datasets["control"] == {}:
                warnings.warn(
                    "You may also calculate persistence based on "
                    "``initialized.isel(lead=0)`` by changing "
                    " ``set_options(PerfectModel_persistence_from_initialized_lead_0=True)``."  # noqa: E501
                )
            has_dataset(
                self._datasets["control"], "control", "compute a persistence forecast"
            )
        input_dict = {
            "ensemble": self._datasets["initialized"],
            "control": self._datasets["control"],
            "init": True,
        }

        res = self._apply_climpred_function(
            compute_persistence_func,
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
        self,
        metric: metricType = None,
        comparison: comparisonType = None,
        dim: dimType = None,
        **metric_kwargs: metric_kwargsType,
    ) -> xr.Dataset:
        """Verify a climatology forecast.

        Args:
            metric: Metric to apply for verification, see `metrics <../metrics.html>`_
            comparison: How to compare the climatology against itself, see
                `comparisons <../comparisons.html>`_
            dim: Dimension(s) over which to apply metric.
                ``dim`` is passed on to xskillscore.{metric} and includes xskillscore's
                ``member_dim``. ``dim`` should contain ``member`` when ``comparison``
                is probabilistic but should not contain ``member`` when
                ``comparison=e2c``. Defaults to ``None``, meaning that all dimensions
                other than ``lead`` are reduced.
            **metric_kwargs: Arguments passed to ``metric``.

        Returns:
            climatology forecast skill

        References:
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
        metric: metricType = None,
        comparison: comparisonType = None,
        dim: dimType = None,
        reference: referenceType = None,
        groupby: groupbyType = None,
        iterations: Optional[int] = None,
        sig: int = 95,
        resample_dim: str = "member",
        pers_sig: Optional[int] = None,
        **metric_kwargs: metric_kwargsType,
    ) -> xr.Dataset:
        """Bootstrap with replacement according to :cite:t:`Goddard2013`.

        Args:
            metric: Metric to apply for verification, see `metrics <../metrics.html>`_
            comparison: How to compare the forecast against itself, see
                `comparisons <../comparisons.html>`_
            dim: Dimension(s) over which to apply metric.
                ``dim`` is passed on to xskillscore.{metric} and includes xskillscore's
                ``member_dim``. ``dim`` should contain ``member`` when ``comparison``
                is probabilistic but should not contain ``member`` when
                ``comparison=e2c``. Defaults to ``None`` meaning that all dimensions
                other than ``lead`` are reduced.
            reference: Type of reference forecasts with which to verify against.
                One or more of ``["uninitialized", "persistence", "climatology"]``.
                Defaults to ``None`` meaning no reference.
                If ``None`` or ``[]``, returns no p value.
                For ``persistence``, choose between
                ``set_options(PerfectModel_persistence_from_initialized_lead_0)=False``
                (default) using :py:func:`~climpred.reference.compute_persistence` or
                ``set_options(PerfectModel_persistence_from_initialized_lead_0)=True``
                using
                :py:func:`~climpred.reference.compute_persistence_from_first_lead`.
            iterations: Number of resampling iterations for bootstrapping with
                replacement. Recommended >= 500.
            resample_dim: dimension to resample from. Defaults to `"member"``.

                - "member": select a different set of members from forecast
                - "init': select a different set of initializations from forecast

            sig: Significance level in percent for deciding whether
                uninitialized and persistence beat initialized skill.
            pers_sig: If not ``None``, the separate significance level for
                persistence. Defaults to ``None``, or the same significance as ``sig``.
            groupby: group ``init`` before passing ``initialized`` to ``bootstrap``.
            **metric_kwargs: arguments passed to ``metric``.

        Returns:
            :py:class:`xarray.Dataset` with dimensions ``results`` (holding
            ``verify skill``, ``p``, ``low_ci`` and ``high_ci``) and ``skill``
            (holding ``initialized``, ``persistence`` and/or ``uninitialized``):
                * results="verify skill", skill="initialized":
                    mean initialized skill
                * results="high_ci", skill="initialized":
                    high confidence interval boundary for initialized skill
                * results="p", skill="uninitialized":
                    p value of the hypothesis that the
                    difference of skill between the initialized and
                    uninitialized simulations is smaller or equal to zero
                    based on bootstrapping with replacement.
                * results="p", skill="persistence":
                    p value of the hypothesis that the
                    difference of skill between the initialized and persistenceistence
                    simulations is smaller or equal to zero based on
                    bootstrapping with replacement.

        Reference:
            :cite:t:`Goddard2013`

        Example:
            Continuous Ranked Probability Score (``"crps"``) comparing every
            member to every other member (``"m2m"``) reducing dimensions ``member`` and
            ``init`` 50 times after resampling ``member`` dimension with replacement.
            Also calculate reference skill for the ``"persistence"``, ``"climatology"``
            and ``"uninitialized"`` forecast and compare whether initialized skill is
            better than reference skill: Returns verify skill, probability that
            reference forecast performs better than initialized and the lower and
            upper bound of the resample.

            >>> PerfectModelEnsemble.bootstrap(
            ...     metric="crps",
            ...     comparison="m2m",
            ...     dim=["init", "member"],
            ...     iterations=50,
            ...     resample_dim="member",
            ...     reference=["persistence", "climatology", "uninitialized"],
            ... )
            <xarray.Dataset>
            Dimensions:  (skill: 4, results: 4, lead: 20)
            Coordinates:
              * lead     (lead) int64 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
              * results  (results) <U12 'verify skill' 'p' 'low_ci' 'high_ci'
              * skill    (skill) <U13 'initialized' 'persistence' ... 'uninitialized'
            Data variables:
                tos      (skill, results, lead) float64 0.0621 0.07352 ... 0.1607 0.1439
            Attributes: (12/13)
                prediction_skill_software:                         climpred https://climp...
                skill_calculated_by_function:                      PerfectModelEnsemble.b...
                number_of_initializations:                         12
                number_of_members:                                 10
                metric:                                            crps
                comparison:                                        m2m
                ...                                                ...
                reference:                                         ['persistence', 'clima...
                PerfectModel_persistence_from_initialized_lead_0:  False
                resample_dim:                                      member
                sig:                                               95
                iterations:                                        50
                confidence_interval_levels:                        0.975-0.025

        """
        if groupby is not None:
            return self._groupby(
                "bootstrap",
                groupby,
                reference=reference,
                metric=metric,
                comparison=comparison,
                dim=dim,
                iterations=iterations,
                resample_dim=resample_dim,
                sig=sig,
                pers_sig=pers_sig,
                **metric_kwargs,
            )

        if iterations is None:
            raise ValueError("Designate number of bootstrapping `iterations`.")
        reference = _check_valid_reference(reference)
        has_dataset(self._datasets["control"], "control", "iteration")
        input_dict = {
            "ensemble": self._datasets["initialized"],
            "control": self._datasets["control"],
            "init": True,
        }
        bootstrapped_skill = self._apply_climpred_function(
            bootstrap_perfect_model,
            input_dict=input_dict,
            metric=metric,
            comparison=comparison,
            dim=dim,
            reference=reference,
            resample_dim=resample_dim,
            sig=sig,
            iterations=iterations,
            pers_sig=pers_sig,
            **metric_kwargs,
        )
        bootstrapped_skill = assign_attrs(
            bootstrapped_skill,
            self.get_initialized(),
            function_name="PerfectModelEnsemble.bootstrap()",
            metric=metric,
            comparison=comparison,
            dim=dim,
            reference=reference,
            resample_dim=resample_dim,
            sig=sig,
            iterations=iterations,
            pers_sig=pers_sig,
            **metric_kwargs,
        )
        return bootstrapped_skill


class HindcastEnsemble(PredictionEnsemble):
    """An object for initialized prediction ensembles.

    :py:class:`.HindcastEnsemble` is a sub-class of
    :py:class:`.PredictionEnsemble`. It tracks a
    verification dataset (i.e., observations) associated with the hindcast ensemble
    for easy computation across multiple variables.

    This object is built on :py:class:`xarray.Dataset` and thus requires the input
    object to be an :py:class:`xarray.Dataset` or :py:class:`xarray.DataArray`.
    """

    def __init__(self, initialized: Union[xr.DataArray, xr.Dataset]) -> None:
        """Create ``HindcastEnsemble`` from initialized prediction ensemble output.

        Args:
          initialized: initialized prediction ensemble output.

        Attributes:
          observations: datasets dictionary item of verification data to associate with
            the prediction ensemble.
          uninitialized: datasets dictionary item of uninitialized forecast.
        """
        super().__init__(initialized)
        self._datasets.update({"observations": {}})
        self.kind = "hindcast"

    def _apply_climpred_function(
        self, func: Callable[..., xr.Dataset], init: bool, **kwargs: Any
    ) -> Union["HindcastEnsemble", xr.Dataset]:
        """Loop through verification data and apply an arbitrary climpred function.

        Args:
            func: climpred function to apply to object.
            init: Whether or not it's the initialized ensemble.
        """
        # fixme: essentially the same as map?
        hind = self._datasets["initialized"]
        verif = self._datasets["observations"]
        drop_init, drop_obs = self._vars_to_drop(init=init)
        return func(hind.drop_vars(drop_init), verif.drop_vars(drop_obs), **kwargs)

    def _vars_to_drop(self, init: bool = True) -> Tuple[List[str], List[str]]:
        """Return list of variables to drop.

        When comparing initialized/uninitialized to observations.

        This is useful if the two products being compared do not share the same
        variables. I.e., if the observations have ["SST"] and the initialized has
        ["SST", "SALT"], this will return a list with ["SALT"] to be dropped
        from the initialized.

        Args:
          init:
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

    def add_observations(
        self, obs: Union[xr.DataArray, xr.Dataset]
    ) -> "HindcastEnsemble":
        """Add verification data against which to verify the initialized ensemble.

        Same as :py:meth:`.HindcastEnsemble.add_verification`.

        Args:
            obs: observations added to :py:class:`.HindcastEnsemble`.
        """
        if isinstance(obs, xr.DataArray):
            obs = obs.to_dataset()
        match_initialized_dims(self._datasets["initialized"], obs)
        match_initialized_vars(self._datasets["initialized"], obs)
        # Check that time is int, cftime, or datetime; convert ints or cftime to
        # datetime.
        obs = convert_time_index(obs, "time", "obs[time]")
        # Check that converted/original cftime calendar is the same as the
        # initialized calendar to avoid any alignment errors.
        match_calendars(self._datasets["initialized"], obs)
        datasets = self._datasets.copy()
        datasets.update({"observations": obs})
        return self._construct_direct(datasets, kind="hindcast")

    def add_verification(
        self, verif: Union[xr.DataArray, xr.Dataset]
    ) -> "HindcastEnsemble":
        """Add verification data against which to verify the initialized ensemble.

        Same as :py:meth:`.HindcastEnsemble.add_observations`.

        Args:
            verif: verification added to :py:class:`.HindcastEnsemble`.
        """
        return self.add_observations(verif)

    def add_uninitialized(
        self, uninit: Union[xr.DataArray, xr.Dataset]
    ) -> "HindcastEnsemble":
        """Add a companion uninitialized ensemble for comparison to verification data.

        Args:
            uninit: uninitialzed ensemble.
        """
        if isinstance(uninit, xr.DataArray):
            uninit = uninit.to_dataset()
        match_initialized_dims(
            self._datasets["initialized"], uninit, uninitialized=True
        )
        match_initialized_vars(self._datasets["initialized"], uninit)
        # Check that init is int, cftime, or datetime; convert ints or cftime to
        # datetime.
        uninit = convert_time_index(uninit, "time", "uninit[time]")
        # Check that converted/original cftime calendar is the same as the
        # initialized calendar to avoid any alignment errors.
        match_calendars(self._datasets["initialized"], uninit, kind2="uninitialized")
        datasets = self._datasets.copy()
        datasets.update({"uninitialized": uninit})
        return self._construct_direct(datasets, kind="hindcast")

    def get_observations(self) -> xr.Dataset:
        """Return the :py:class:`xarray.Dataset` of the observations/verification data.

        Returns:
            observations
        """
        return self._datasets["observations"]

    def generate_uninitialized(
        self, resample_dim: List[str] = ["init", "member"]
    ) -> "HindcastEnsemble":
        """Generate ``uninitialized`` by resampling from ``initialized``.

        Args:
            resample_dim: dimension to resample from. Must contain ``"init"``.

        Returns:
            resampled ``uninitialized`` ensemble added to
            :py:class:`.HindcastEnsemble`

        Example:
            >>> HindcastEnsemble  # uninitialized from historical simulations
            <climpred.HindcastEnsemble>
            Initialized Ensemble:
                SST      (init, lead, member) float64 -0.2392 -0.2203 ... 0.618 0.6136
            Observations:
                SST      (time) float32 -0.4015 -0.3524 -0.1851 ... 0.2481 0.346 0.4502
            Uninitialized:
                SST      (time, member) float64 -0.1969 -0.01221 -0.275 ... 0.4179 0.3974

            >>> HindcastEnsemble.generate_uninitialized()  # newly generated from initialized
            <climpred.HindcastEnsemble>
            Initialized Ensemble:
                SST      (init, lead, member) float64 -0.2392 -0.2203 ... 0.618 0.6136
            Observations:
                SST      (time) float32 -0.4015 -0.3524 -0.1851 ... 0.2481 0.346 0.4502
            Uninitialized:
                SST      (time, member) float64 0.04868 0.07173 0.09435 ... 0.4158 0.418
        """
        uninit = resample_uninitialized_from_initialized(
            self._datasets["initialized"], resample_dim=resample_dim
        )
        datasets = self._datasets.copy()
        datasets.update({"uninitialized": uninit})
        return self._construct_direct(datasets, kind="hindcast")

    def plot_alignment(
        self: "HindcastEnsemble",
        alignment: Optional[Union[str, List[str]]] = None,
        reference: Optional[referenceType] = None,
        date2num_units: str = "days since 1960-01-01",
        return_xr: bool = False,
        cmap: str = "viridis",
        edgecolors: str = "gray",
        **plot_kwargs: Any,
    ) -> Any:
        """
        Plot ``initialized`` ``valid_time`` where matching ``verification`` ``time``.

        Depends on ``alignment``. Plots ``days since reference date`` controlled by
        ``date2num_units``. ``NaN`` / white space shows where no verification is done.

        Args:
            alignment: which inits or verification times should be aligned?

                - ``"maximize"``: maximize the degrees of freedom by slicing
                  ``initialized`` and ``verif`` to a common time frame at each lead.
                - ``"same_inits"``: slice to a common ``init`` frame prior to computing
                  metric. This philosophy follows the thought that each lead should be
                  based on the same set of initializations.
                - ``"same_verif"``: slice to a common/consistent verification time frame
                  prior to computing metric. This philosophy follows the thought that
                  each lead should be based on the same set of verification dates.
                - ``None`` defaults to the three above.

            reference: Type of reference forecasts with which to verify against.
                One or more of ``["uninitialized", "persistence", "climatology"]``.
                Defaults to ``None`` meaning no reference.
            date2num_units: passed to ``cftime.date2num`` as units
            return_xr: if ``True`` return :py:class:`xarray.DataArray` else plot
            cmap: color palette
            edgecolors: color of the edges in the plot
            **plot_kwargs: arguments passed to ``plot``.

        Returns:
            :py:class:`xarray.DataArray` if ``return_xr`` else plot

        Example:
            >>> HindcastEnsemble.plot_alignment(alignment=None, return_xr=True)
            <xarray.DataArray 'valid_time' (alignment: 3, lead: 10, init: 61)>
            array([[[-1826., -1461., -1095., ...,    nan,    nan,    nan],
                    [-1461., -1095.,  -730., ...,    nan,    nan,    nan],
                    [-1095.,  -730.,  -365., ...,    nan,    nan,    nan],
                    ...,
                    [  731.,  1096.,  1461., ...,    nan,    nan,    nan],
                    [ 1096.,  1461.,  1827., ...,    nan,    nan,    nan],
                    [ 1461.,  1827.,  2192., ...,    nan,    nan,    nan]],
            <BLANKLINE>
                   [[   nan,    nan,    nan, ..., 19359., 19724., 20089.],
                    [   nan,    nan,    nan, ..., 19724., 20089.,    nan],
                    [   nan,    nan,    nan, ..., 20089.,    nan,    nan],
                    ...,
                    [   nan,    nan,  1461., ...,    nan,    nan,    nan],
                    [   nan,  1461.,  1827., ...,    nan,    nan,    nan],
                    [ 1461.,  1827.,  2192., ...,    nan,    nan,    nan]],
            <BLANKLINE>
                   [[-1826., -1461., -1095., ..., 19359., 19724., 20089.],
                    [-1461., -1095.,  -730., ..., 19724., 20089.,    nan],
                    [-1095.,  -730.,  -365., ..., 20089.,    nan,    nan],
                    ...,
                    [  731.,  1096.,  1461., ...,    nan,    nan,    nan],
                    [ 1096.,  1461.,  1827., ...,    nan,    nan,    nan],
                    [ 1461.,  1827.,  2192., ...,    nan,    nan,    nan]]])
            Coordinates:
              * init       (init) object 1954-01-01 00:00:00 ... 2014-01-01 00:00:00
              * lead       (lead) int32 1 2 3 4 5 6 7 8 9 10
              * alignment  (alignment) <U10 'same_init' 'same_verif' 'maximize'
            Attributes:
                units:    days since 1960-01-01

            >>> HindcastEnsemble.plot_alignment(
            ...     alignment="same_verifs"
            ... )  # doctest: +SKIP
            <matplotlib.collections.QuadMesh object at 0x1405c1520>

        See also:
            https://climpred.readthedocs.io/en/stable/alignment.html.
        """
        from .graphics import _verif_dates_xr

        if alignment is None or alignment == []:
            alignment = ["same_init", "same_verif", "maximize"]
        if isinstance(alignment, str):
            alignment = [alignment]

        alignment_dates = []
        alignments_success = []
        for a in alignment:
            try:
                alignment_dates.append(
                    _verif_dates_xr(self, a, reference, date2num_units)
                )
                alignments_success.append(a)
            except CoordinateError as e:
                warnings.warn(f"alignment='{a}' failed. CoordinateError: {e}")
        verif_dates_xr = (
            xr.concat(
                alignment_dates,
                "alignment",
            )
            .assign_coords(alignment=alignments_success)
            .squeeze()
        )
        if "alignment" in verif_dates_xr.dims:
            plot_kwargs["col"] = "alignment"

        if return_xr:
            return verif_dates_xr
        try:
            import nc_time_axis  # noqa:

            assert int(nc_time_axis.__version__.replace(".", "")) >= 140
            return verif_dates_xr.plot(cmap=cmap, edgecolors=edgecolors, **plot_kwargs)
        except ImportError:
            raise ValueError("nc_time_axis>1.4.0 required for plotting.")

    def verify(
        self,
        metric: metricType = None,
        comparison: comparisonType = None,
        dim: dimType = None,
        alignment: alignmentType = None,
        reference: referenceType = None,
        groupby: groupbyType = None,
        **metric_kwargs: metric_kwargsType,
    ) -> xr.Dataset:
        """Verify the initialized ensemble against observations.

        .. note::
            This will automatically verify against all shared variables
            between the initialized ensemble and observations/verification data.

        Args:
            metric: Metric to apply for verification, see `metrics <../metrics.html>`_
            comparison: How to compare to the observations/verification data.
                See `comparisons <../comparisons.html>`_.
            dim: Dimension(s) to apply metric over. ``dim`` is passed
                on to xskillscore.{metric} and includes xskillscore's ``member_dim``.
                ``dim`` should contain ``member`` when ``comparison`` is probabilistic
                but should not contain ``member`` when ``comparison=e2o``. Defaults to
                ``None`` meaning that all dimensions other than ``lead`` are reduced.
            alignment: which inits or verification times should be aligned?

                - ``"maximize"``: maximize the degrees of freedom by slicing
                  ``initialized`` and ``verif`` to a common time frame at each lead.
                - ``"same_inits"``: slice to a common ``init`` frame prior to computing
                  metric. This philosophy follows the thought that each lead should be
                  based on the same set of initializations.
                - ``"same_verif"``: slice to a common/consistent verification time frame
                  prior to computing metric. This philosophy follows the thought that
                  each lead should be based on the same set of verification dates.

            reference: Type of reference forecasts with which to verify against.
                One or more of ``["uninitialized", "persistence", "climatology"]``.
                Defaults to ``None`` meaning no reference.
            groupby: group ``init`` before passing ``initialized`` to ``verify``.
            **metric_kwargs: arguments passed to ``metric``.

        Returns:
            ``initialized`` and ``reference`` forecast skill reduced by dimensions
            ``dim``

        Example:
            Continuous Ranked Probability Score (``crps``) comparing every member with the
            verification (``m2o``) over the same verification time (``same_verifs``)
            for all leads reducing dimensions ``init`` and ``member``:

            >>> HindcastEnsemble.verify(
            ...     metric="rmse",
            ...     comparison="m2o",
            ...     alignment="same_verifs",
            ...     dim=["init", "member"],
            ... )
            <xarray.Dataset>
            Dimensions:  (lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
                skill    <U11 'initialized'
            Data variables:
                SST      (lead) float64 0.08516 0.09492 0.1041 ... 0.1525 0.1697 0.1785
            Attributes:
                prediction_skill_software:     climpred https://climpred.readthedocs.io/
                skill_calculated_by_function:  HindcastEnsemble.verify()
                number_of_initializations:     64
                number_of_members:             10
                alignment:                     same_verifs
                metric:                        rmse
                comparison:                    m2o
                dim:                           ['init', 'member']
                reference:                     []

            Root mean square error (``"rmse"``) comparing the ensemble mean with
            the verification (``"e2o"``) over the same initializations
            (``"same_inits"``) for all leads reducing dimension ``init`` while also
            calculating reference skill for the ``"persistence"``, ``"climatology"``
            and ``"uninitialized"`` forecast.

            >>> HindcastEnsemble.verify(
            ...     metric="rmse",
            ...     comparison="e2o",
            ...     alignment="same_inits",
            ...     dim="init",
            ...     reference=["persistence", "climatology", "uninitialized"],
            ... )
            <xarray.Dataset>
            Dimensions:  (skill: 4, lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
              * skill    (skill) <U13 'initialized' 'persistence' ... 'uninitialized'
            Data variables:
                SST      (skill, lead) float64 0.08135 0.08254 0.086 ... 0.07377 0.07409
            Attributes:
                prediction_skill_software:     climpred https://climpred.readthedocs.io/
                skill_calculated_by_function:  HindcastEnsemble.verify()
                number_of_initializations:     64
                number_of_members:             10
                alignment:                     same_inits
                metric:                        rmse
                comparison:                    e2o
                dim:                           init
                reference:                     ['persistence', 'climatology', 'uninitiali...
        """
        if groupby is not None:
            return self._groupby(
                "verify",
                groupby,
                reference=reference,
                metric=metric,
                comparison=comparison,
                dim=dim,
                alignment=alignment,
                **metric_kwargs,
            )

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
                    initialized=forecast,
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
            result = xr.concat(metric_over_leads, dim="lead")  # , **CONCAT_KWARGS)
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
                            initialized=forecast,
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
                    ref = xr.concat(metric_over_leads, dim="lead")  # , **CONCAT_KWARGS)
                    ref["lead"] = forecast["lead"]
                    # fix to get no member dim for uninitialized e2o skill #477
                    if (
                        r == "uninitialized"
                        and comparison.name == "e2o"
                        and "member" in ref.dims
                    ):
                        ref = ref.mean("member")
                        if "time" in ref.dims and "time" not in result.dims:
                            ref = ref.rename({"time": "init"})
                    result = xr.concat([result, ref], dim="skill", **CONCAT_KWARGS)
            # rename back to 'init'
            if "time" in result.dims:
                result = result.swap_dims({"time": "init"})
            if "time" in result.coords:
                if "init" in result.coords["time"].dims:
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

        res = assign_attrs(
            res,
            self.get_initialized(),
            function_name="HindcastEnsemble.verify()",
            metric=metric,
            comparison=comparison,
            dim=dim,
            alignment=alignment,
            reference=reference,
            **metric_kwargs,
        )
        return res

    def bootstrap(
        self,
        metric: metricType = None,
        comparison: comparisonType = None,
        dim: dimType = None,
        alignment: alignmentType = None,
        reference: referenceType = None,
        groupby: groupbyType = None,
        iterations: Optional[int] = None,
        sig: int = 95,
        resample_dim: str = "member",
        pers_sig: Optional[int] = None,
        **metric_kwargs: metric_kwargsType,
    ) -> xr.Dataset:
        """Bootstrap with replacement according to :cite:t:`Goddard2013`.

        Args:
            metric: Metric to apply for verification, see `metrics <../metrics.html>`_
            comparison: How to compare to the observations/verification data.
                See `comparisons <../comparisons.html>`_.
            dim: Dimension(s) to apply metric over. ``dim`` is passed
                on to xskillscore.{metric} and includes xskillscore's ``member_dim``.
                ``dim`` should contain ``member`` when ``comparison`` is probabilistic
                but should not contain ``member`` when ``comparison="e2o"``. Defaults to
                ``None`` meaning that all dimensions other than ``lead`` are reduced.
            reference: Type of reference forecasts with which to verify against.
                One or more of ``["uninitialized", "persistence", "climatology"]``.
                Defaults to ``None`` meaning no reference.
                If ``None`` or ``[]``, returns no p value.
            alignment: which inits or verification times should be aligned?

                - ""maximize: maximize the degrees of freedom by slicing ``init`` and
                  ``verif`` to a common time frame at each lead.
                - ``"same_inits"``: slice to a common ``init`` frame prior to computing
                  metric. This philosophy follows the thought that each lead should be
                  based on the same set of initializations.
                - ``"same_verif"``: slice to a common/consistent verification time frame
                  prior to computing metric. This philosophy follows the thought that
                  each lead should be based on the same set of verification dates.

            iterations: Number of resampling iterations for bootstrapping with
                replacement. Recommended >= 500.
            sig: Significance level in percent for deciding whether
                uninitialized and persistence beat initialized skill.
            resample_dim: dimension to resample from. Default: ``"member"``.

                - ``"member"``: select a different set of members from hind
                - ``"init"``: select a different set of initializations from hind

            pers_sig: If not ``None``, the separate significance level for persistence.
            groupby: group ``init`` before passing ``initialized`` to ``bootstrap``.
            **metric_kwargs: arguments passed to ``metric``.

        Returns:
            :py:class:`xarray.Dataset` with dimensions ``results`` (holding ``skill``,
            ``p``, ``low_ci`` and ``high_ci``) and ``skill`` (holding ``initialized``,
            ``persistence`` and/or ``uninitialized``):
                * results="verify skill", skill="initialized":
                    mean initialized skill
                * results="high_ci", skill="initialized":
                    high confidence interval boundary for initialized skill
                * results="p", skill="uninitialized":
                    p value of the hypothesis that the
                    difference of skill between the initialized and
                    uninitialized simulations is smaller or equal to zero
                    based on bootstrapping with replacement.
                * results="p", skill="persistence":
                    p value of the hypothesis that the
                    difference of skill between the initialized and persistence
                    simulations is smaller or equal to zero based on
                    bootstrapping with replacement.

        References:
            :cite:t:`Goddard2013`

        Example:
            Continuous Ranked Probability Score (``"crps"``) comparing every member
            forecast to the verification (``"m2o"``) over the same initializations
            (``"same_inits"``) for all leads reducing dimension ``member`` 50 times
            after resampling ``member`` dimension with replacement. Note that dimension
            ``init`` remains.
            Also calculate reference skill for the ``"persistence"``, ``"climatology"``
            and ``"uninitialized"`` forecast and compare whether initialized skill is
            better than reference skill: Returns verify skill, probability that
            reference forecast performs better than initialized and the lower and
            upper bound of the resample.

            >>> HindcastEnsemble.bootstrap(
            ...     metric="crps",
            ...     comparison="m2o",
            ...     dim="member",
            ...     iterations=50,
            ...     resample_dim="member",
            ...     alignment="same_inits",
            ...     reference=["persistence", "climatology", "uninitialized"],
            ... )
            <xarray.Dataset>
            Dimensions:     (skill: 4, results: 4, lead: 10, init: 51)
            Coordinates:
              * init        (init) object 1955-01-01 00:00:00 ... 2005-01-01 00:00:00
              * lead        (lead) int32 1 2 3 4 5 6 7 8 9 10
                valid_time  (lead, init) object 1956-01-01 00:00:00 ... 2015-01-01 00:00:00
              * results     (results) <U12 'verify skill' 'p' 'low_ci' 'high_ci'
              * skill       (skill) <U13 'initialized' 'persistence' ... 'uninitialized'
            Data variables:
                SST         (skill, results, lead, init) float64 0.1202 0.01764 ... 0.1033
            Attributes:
                prediction_skill_software:     climpred https://climpred.readthedocs.io/
                skill_calculated_by_function:  HindcastEnsemble.bootstrap()
                number_of_members:             10
                alignment:                     same_inits
                metric:                        crps
                comparison:                    m2o
                dim:                           member
                reference:                     ['persistence', 'climatology', 'uninitiali...
                resample_dim:                  member
                sig:                           95
                iterations:                    50
                confidence_interval_levels:    0.975-0.025

        """
        if groupby is not None:
            return self._groupby(
                "bootstrap",
                groupby,
                reference=reference,
                metric=metric,
                comparison=comparison,
                dim=dim,
                iterations=iterations,
                alignment=alignment,
                resample_dim=resample_dim,
                sig=sig,
                pers_sig=pers_sig,
                **metric_kwargs,
            )

        if iterations is None:
            raise ValueError("Designate number of bootstrapping `iterations`.")
        # TODO: replace with more computationally efficient classes implementation
        # https://github.com/pangeo-data/climpred/issues/375
        reference = _check_valid_reference(reference)
        if "uninitialized" in reference and not isinstance(
            self.get_uninitialized(), xr.Dataset
        ):
            raise ValueError(
                "`reference='uninitialized'` requires `uninitialized` dataset."
                "Use `HindcastEnsemble.add_uninitialized(uninitialized_ds)``."
            )
        bootstrapped_skill = bootstrap_hindcast(
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
        bootstrapped_skill = assign_attrs(
            bootstrapped_skill,
            self.get_initialized(),
            function_name="HindcastEnsemble.bootstrap()",
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
        return bootstrapped_skill

    def remove_bias(
        self,
        alignment: alignmentType = None,
        how: str = "additive_mean",
        train_test_split: str = "unfair",
        train_init: Optional[Union[xr.DataArray, slice]] = None,
        train_time: Optional[Union[xr.DataArray, slice]] = None,
        cv: Union[bool, str] = False,
        **metric_kwargs: metric_kwargsType,
    ) -> "HindcastEnsemble":
        """Remove bias from :py:class:`.HindcastEnsemble`.

        Bias is grouped by ``seasonality`` set via
        :py:class:`~climpred.options.set_options`. When wrapping
        :py:class:`xclim.sdba.adjustment.TrainAdjust` use ``group`` instead.

        Args:
            alignment: which inits or verification times should be aligned?

                - ``""maximize``: maximize the degrees of freedom by slicing
                  ``initialized`` and ``verif`` to a common time frame at each lead.
                - ``"same_inits"``: slice to a common ``init`` frame prior to computing
                  metric. This philosophy follows the thought that each lead should be
                  based on the same set of initializations.
                - ``"same_verif"``: slice to a common/consistent verification time frame
                  prior to computing metric. This philosophy follows the thought that
                  each lead should be based on the same set of verification dates.

            how: what kind of bias removal to perform.
                Defaults to ``"additive_mean"``. Select from:

                - ``"additive_mean"``: correcting the mean forecast additively
                - ``"multiplicative_mean"``: correcting the mean forecast
                  multiplicatively
                - ``"multiplicative_std"``: correcting the standard deviation
                  multiplicatively
                - ``"modified_quantile"``: `Reference <https://www.sciencedirect.com/science/article/abs/pii/S0034425716302000?via%3Dihub>`_
                - ``"basic_quantile"``: `Reference <https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/joc.2168>`_
                - ``"gamma_mapping"``: `Reference <https://www.hydrol-earth-syst-sci.net/21/2649/2017/>`_
                - ``"normal_mapping"``: `Reference <https://www.hydrol-earth-syst-sci.net/21/2649/2017/>`_
                - :py:class:`xclim.sdba.adjustment.EmpiricalQuantileMapping`
                - :py:class:`xclim.sdba.adjustment.DetrendedQuantileMapping`
                - :py:class:`xclim.sdba.adjustment.PrincipalComponents`
                - :py:class:`xclim.sdba.adjustment.QuantileDeltaMapping`
                - :py:class:`xclim.sdba.adjustment.Scaling`
                - :py:class:`xclim.sdba.adjustment.LOCI`

            train_test_split: How to separate train period to calculate the bias
                and test period to apply bias correction to? For a detailed
                description, see `Risbey et al. 2021 <http://www.nature.com/articles/s41467-021-23771-z>`_:

                - ``"fair"```: no overlap between ``train`` and ``test`` (recommended).
                  Set either ``train_init`` or ``train_time``.
                - ``"unfair"``: completely overlapping ``train`` and ``test`` (default).
                - ``"unfair-cv"```: overlapping ``train`` and ``test`` except for
                  current `init`, which is
                  `left out <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation>`_
                  (set ``cv="LOO"``).

            train_init: Define initializations for training
              when ``alignment="same_inits/maximize"``.
            train_time: Define time for training when ``alignment="same_verif"``.
            cv: Only relevant when ``train_test_split="unfair-cv"``.
              Defaults to ``False``.

                - ``True/"LOO"``: Calculate bias by `leaving given initialization out <https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Leave-one-out_cross-validation>`_
                - ``False``: include all initializations in the calculation of bias,
                  which is much faster and but yields similar skill with a large N of
                  initializations.

            **metric_kwargs: passed to ``xclim.sdba`` (including ``group``)
                or ``XBias_Correction``

        Returns:
            bias removed :py:class:`.HindcastEnsemble`.

        Example:

            Skill from raw model output without bias reduction:

            >>> HindcastEnsemble.verify(
            ...     metric="rmse", comparison="e2o", alignment="maximize", dim="init"
            ... )
            <xarray.Dataset>
            Dimensions:  (lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
                skill    <U11 'initialized'
            Data variables:
                SST      (lead) float64 0.08359 0.08141 0.08362 ... 0.1361 0.1552 0.1664
            Attributes:
                prediction_skill_software:     climpred https://climpred.readthedocs.io/
                skill_calculated_by_function:  HindcastEnsemble.verify()
                number_of_initializations:     64
                number_of_members:             10
                alignment:                     maximize
                metric:                        rmse
                comparison:                    e2o
                dim:                           init
                reference:                     []

            Note that this HindcastEnsemble is already bias reduced, therefore
            ``train_test_split="unfair"`` has hardly any effect. Use all
            initializations to calculate bias and verify skill:

            >>> HindcastEnsemble.remove_bias(
            ...     alignment="maximize", how="additive_mean", test_train_split="unfair"
            ... ).verify(
            ...     metric="rmse", comparison="e2o", alignment="maximize", dim="init"
            ... )
            <xarray.Dataset>
            Dimensions:  (lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
                skill    <U11 'initialized'
            Data variables:
                SST      (lead) float64 0.08349 0.08039 0.07522 ... 0.07305 0.08107 0.08255
            Attributes:
                prediction_skill_software:     climpred https://climpred.readthedocs.io/
                skill_calculated_by_function:  HindcastEnsemble.verify()
                number_of_initializations:     64
                number_of_members:             10
                alignment:                     maximize
                metric:                        rmse
                comparison:                    e2o
                dim:                           init
                reference:                     []

            Separate initializations 1954 - 1980 to calculate bias. Note that
            this HindcastEnsemble is already bias reduced, therefore
            ``train_test_split="fair"`` worsens skill here. Generally,
            ``train_test_split="fair"`` is recommended to use for a fair
            comparison against real-time forecasts.

            >>> HindcastEnsemble.remove_bias(
            ...     alignment="maximize",
            ...     how="additive_mean",
            ...     train_test_split="fair",
            ...     train_init=slice("1954", "1980"),
            ... ).verify(
            ...     metric="rmse", comparison="e2o", alignment="maximize", dim="init"
            ... )
            <xarray.Dataset>
            Dimensions:  (lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
                skill    <U11 'initialized'
            Data variables:
                SST      (lead) float64 0.132 0.1085 0.08722 ... 0.08209 0.08969 0.08732
            Attributes:
                prediction_skill_software:     climpred https://climpred.readthedocs.io/
                skill_calculated_by_function:  HindcastEnsemble.verify()
                number_of_initializations:     37
                number_of_members:             10
                alignment:                     maximize
                metric:                        rmse
                comparison:                    e2o
                dim:                           init
                reference:                     []

            Wrapping methods ``how`` from
            `xclim <https://xclim.readthedocs.io/en/stable/sdba_api.html>`_ and
            providing ``group`` for ``groupby``:

            >>> HindcastEnsemble.remove_bias(
            ...     alignment="same_init",
            ...     group="init",
            ...     how="DetrendedQuantileMapping",
            ...     train_test_split="unfair",
            ... ).verify(
            ...     metric="rmse", comparison="e2o", alignment="maximize", dim="init"
            ... )
            <xarray.Dataset>
            Dimensions:  (lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
                skill    <U11 'initialized'
            Data variables:
                SST      (lead) float64 0.09841 0.09758 0.08238 ... 0.0771 0.08119 0.08322
            Attributes:
                prediction_skill_software:     climpred https://climpred.readthedocs.io/
                skill_calculated_by_function:  HindcastEnsemble.verify()
                number_of_initializations:     52
                number_of_members:             10
                alignment:                     maximize
                metric:                        rmse
                comparison:                    e2o
                dim:                           init
                reference:                     []

            Wrapping methods ``how`` from `bias_correction <https://github.com/pankajkarman/bias_correction/blob/master/bias_correction.py>`_:

            >>> HindcastEnsemble.remove_bias(
            ...     alignment="same_init",
            ...     how="modified_quantile",
            ...     train_test_split="unfair",
            ... ).verify(
            ...     metric="rmse", comparison="e2o", alignment="maximize", dim="init"
            ... )
            <xarray.Dataset>
            Dimensions:  (lead: 10)
            Coordinates:
              * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
                skill    <U11 'initialized'
            Data variables:
                SST      (lead) float64 0.07628 0.08293 0.08169 ... 0.1577 0.1821 0.2087
            Attributes:
                prediction_skill_software:     climpred https://climpred.readthedocs.io/
                skill_calculated_by_function:  HindcastEnsemble.verify()
                number_of_initializations:     52
                number_of_members:             10
                alignment:                     maximize
                metric:                        rmse
                comparison:                    e2o
                dim:                           init
                reference:                     []
        """  # noqa: E501
        if train_test_split not in BIAS_CORRECTION_TRAIN_TEST_SPLIT_METHODS:
            raise NotImplementedError(
                f"train_test_split='{train_test_split}' not implemented. Please choose "
                f"`train_test_split` from {BIAS_CORRECTION_TRAIN_TEST_SPLIT_METHODS}, "
                "see Risbey et al. 2021 "
                "http://www.nature.com/articles/s41467-021-23771-z for description and "
                "https://github.com/pangeo-data/climpred/issues/648 for implementation "
                "status."
            )

        alignment = _check_valid_alignment(alignment)

        if train_test_split in ["fair"]:
            if (
                (train_init is None)
                or not isinstance(train_init, (slice, xr.DataArray))
            ) and (alignment in ["same_inits", "maximize"]):
                raise ValueError(
                    f'When alignment="{alignment}", please provide `train_init` as '
                    f"`xr.DataArray`, e.g. "
                    '`HindcastEnsemble.coords["init"].slice(start, end)` '
                    "or slice, e.g. `slice(start, end)`, got `train_init={train_init}`."
                )
            if (
                (train_time is None)
                or not isinstance(train_time, (slice, xr.DataArray))
            ) and (alignment in ["same_verif"]):
                raise ValueError(
                    f'When alignment="{alignment}", please provide `train_time` as '
                    "`xr.DataArray`, e.g. "
                    '`HindcastEnsemble.coords["time"].slice(start, end)` '
                    "or slice, e.g. `slice(start, end)`, got `train_time={train_time}`."
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
                f"bias removal '{how}' is not implemented, please choose from "
                f" {INTERNAL_BIAS_CORRECTION_METHODS+BIAS_CORRECTION_BIAS_CORRECTION_METHODS}."  # noqa: E501
            )

        if train_test_split in ["unfair-cv"]:
            if cv not in [True, "LOO"]:
                raise ValueError(
                    f"Please provide cross-validation keyword `cv='LOO'` when using "
                    f"`train_test_split='unfair-cv'`, found `cv='{cv}'`."
                )
            else:
                cv = "LOO"  # backward compatibility
            if cv not in CROSS_VALIDATE_METHODS:
                raise NotImplementedError(
                    f"Cross validation method {cv} not implemented. "
                    f"Please choose cv from {CROSS_VALIDATE_METHODS}."
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
        # TODO: find better location to prevent valid_time with member dims
        if "valid_time" in self._datasets["initialized"].coords:
            if "member" in self._datasets["initialized"].coords["valid_time"].dims:
                self._datasets["initialized"].coords["valid_time"] = (
                    self._datasets["initialized"]
                    .coords["valid_time"]
                    .isel(member=0, drop=True)
                )
        return self
