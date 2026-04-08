"""Main module instantiating ``PerfectModelEnsemble`` and ``HindcastEnsemble."""

import importlib.util as _util
import logging
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
from xml.etree import ElementTree

import cf_xarray  # noqa
import numpy as np
import xarray as xr
from packaging.version import Version
from xarray.core.coordinates import DatasetCoordinates
from xarray.core.dataset import DataVariables
from xarray.core.formatting_html import dataset_repr
from xarray.core.options import OPTIONS as XR_OPTIONS
from xarray.core.utils import Frozen

from .alignment import return_inits_and_verif_dates
from .bias_removal import bias_correction, gaussian_bias_removal, xclim_sdba
from .bootstrap import (
    _distribution_to_ci,
    _p_ci_from_sig,
    _pvalue_from_distributions,
    bootstrap_uninit_pm_ensemble_from_control_cftime,
    resample_skill_exclude_resample_dim_from_dim,
    resample_skill_loop,
    resample_skill_resample_before,
    resample_uninitialized_from_initialized,
    warn_if_chunking_would_increase_performance,
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
from .exceptions import CoordinateError, DimensionError, KeywordError, VariableError
from .metrics import PEARSON_R_CONTAINING_METRICS, Metric
from .options import OPTIONS, set_options
from .prediction import (
    _apply_metric_at_given_lead,
    _get_metric_comparison_dim,
    _sanitize_to_list,
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
