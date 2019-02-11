import xarray as xr
from .prediction import (compute_reference, compute_persistence,
                         compute_perfect_model, bootstrap_perfect_model)
# TODO: add horizon functionality
# TODO: add perfect model functionality
# TODO: add relative entropy functionality
# TODO: add various `get` and `set` decorators
# TODO: refactor other modules, e.g. move all metrics to `metrics.py`


# --------------
# VARIOUS CHECKS
# --------------
def _check_prediction_ensemble_dimensions(xobj):
    """
    Checks that at the minimum, the dple object has dimensions initialization
    and time (a time series with lead times).
    """
    cond = all(dims in xobj.dims for dims in ['initialization', 'time'])
    if not cond:
        # create custom error here.
        raise ValueError("""Your decadal prediction object must contain the
            dimensions `time` and `initialization` at the minimum.""")


def _check_reference_dimensions(init, ref):
    """Checks that the reference matches all initialized dimensions except
    for 'time'"""
    init_dims = list(init.dims)
    if 'time' in init_dims:
        init_dims.remove('time')
    if 'member' in init_dims:
        init_dims.remove('member')
    if not (set(ref.dims) == set(init_dims)):
        raise ValueError("""Reference dimensions must match initialized
            prediction ensemble dimensions (excluding `time`.)""")


# TODO: Don't force same variables for every reference. Allow for some
# references to only have a subset of the initialized prediction variables.
# This means skill, etc. will be computed for whatever variables
# are available.
def _check_reference_vars_match_initialized(init, ref):
    """
    Checks that a new reference (or control) dataset has identical variables
    to the initialized dataset. This ensures that they can be compared
    pairwise.

    ref: new addition
    init: dp.initialized
    """
    init_list = [var for var in init.data_vars]
    ref_list = [var for var in ref.data_vars]
    if set(init_list) != set(ref_list):
        raise ValueError("""Please provide a dataset with matching variables
        (and variable names) to the initialied prediction ensemble.""")


def _check_xarray(xobj):
    if not isinstance(xobj, (xr.Dataset, xr.DataArray)):
        raise ValueError("""You must input an xarray Dataset or DataArray.""")


# ----------
# Aesthetics
# ----------
def _display_metadata(self):
    header = f'<climpred.{type(self).__name__}>'
    summary = header + '\nInitialized Ensemble:\n'
    summary += '    ' + str(self.initialized.data_vars)[18:].strip()
    if isinstance(self, ReferenceEnsemble):
        # TODO: convert to decorator
        if any(self.reference):
            for key in self.reference:
                summary += f'\n{key}:'
                N = len(self.reference[key].data_vars)
                for i in range(1, N+1):
                    summary += '\n    ' + \
                               str(self.reference[key].data_vars) \
                               .split('\n')[i].strip()
        else:
            summary += '\nReferences:\n'
            summary += '    None'
    elif isinstance(self, PerfectModelEnsemble):
        summary += '\nControl:\n'
        # TODO: convert to decorator
        if any(self.control):
            N = len(self.control.data_vars)
            for i in range(1, N+1):
                summary += '    ' + \
                           str(self.control.data_vars) \
                           .split('\n')[i].strip() + '\n'
        else:
            summary += '    None'
    return summary


# -----------------
# CLASS DEFINITIONS
# -----------------
class PredictionEnsemble:
    def __init__(self, xobj):
        _check_xarray(xobj)
        if isinstance(xobj, xr.DataArray):
            # makes applying prediction functions easier, etc.
            xobj = xobj.to_dataset()
        _check_prediction_ensemble_dimensions(xobj)
        self.initialized = xobj

    # when you just print it interactively
    # https://stackoverflow.com/questions/1535327/how-to-print-objects-of-class-using-print
    def __repr__(self):
        return _display_metadata(self)


class PerfectModelEnsemble(PredictionEnsemble):
    def __init__(self, xobj):
        super().__init__(xobj)
        # for consistency with ReferenceEnsemble
        self.control = {}

    def add_control(self, xobj):
        """
        Special to PerfectModelEnsemble. Ensures that there's a control
        to do PM computations with.
        """
        _check_xarray(xobj)
        if isinstance(xobj, xr.DataArray):
            xobj = xobj.to_dataset()
        _check_reference_dimensions(self.initialized, xobj)
        _check_reference_vars_match_initialized(self.initialized, xobj)
        self.control = xobj

    def compute_skill(self, metric='pearson_r', comparison='m2m',
                      running=None, reference_period=None):
        if len(self.control) == 0:
            raise ValueError("""You need to add a control dataset before
            attempting to compute predictability.""")
        else:
            return compute_perfect_model(self.initialized,
                                         self.control,
                                         metric=metric,
                                         comparison=comparison,
                                         running=running,
                                         reference_period=reference_period)

    def compute_persistence(self, nlags=None, metric='pearson_r'):
        if len(self.control) == 0:
            raise ValueError("""You need to add a control dataset before
            attempting to compute a persistence forecast.""")
        if nlags is None:
            nlags = self.initialized.time.size
        return compute_persistence(self.control,
                                   nlags=nlags,
                                   metric=metric)

    def bootstrap(self, metric='rmse', comparison='m2m', reference_period='MK',
                  sig=95, bootstrap=30):
        if len(self.control) == 0:
            raise ValueError("""You need to add a control dataset before
            attempting to bootstrap.""")
        else:
            return bootstrap_perfect_model(self.initialized, self.control,
                                           metric=metric,
                                           comparison=comparison,
                                           reference_period=reference_period,
                                           sig=sig, bootstrap=bootstrap)


class ReferenceEnsemble(PredictionEnsemble):
    def __init__(self, xobj):
        super().__init__(xobj)
        self.reference = {}

    def add_reference(self, xobj, name=None):
        """
        uninitialized things should all have 'initialization' dimension,
        so can stack them here.
        """
        _check_xarray(xobj)
        if isinstance(xobj, xr.DataArray):
            xobj = xobj.to_dataset()
        # TODO: Should just check that at least *one* variable matches.
        # In a case where you have SST references for some products, etc.
        _check_reference_dimensions(self.initialized, xobj)
        _check_reference_vars_match_initialized(self.initialized, xobj)
        self.reference[name] = xobj

    def compute_skill(self, refname=None, metric='pearson_r', comparison='e2r',
                      nlags=None, return_p=False):
        if len(self.reference) == 0:
            raise ValueError("""You need to add a reference dataset before
                attempting to compute predictability.""")
        if refname is not None:
            return compute_reference(self.initialized,
                                     self.reference[refname],
                                     metric=metric,
                                     comparison=comparison,
                                     nlags=nlags,
                                     return_p=return_p)
        else:
            if len(self.reference) == 1:
                refname = list(self.reference.keys())[0]
                return compute_reference(self.initialized,
                                         self.reference[refname],
                                         metric=metric,
                                         comparison=comparison,
                                         nlags=nlags,
                                         return_p=return_p)
            else:
                skill = {}
                for key in self.reference:
                    skill[key] = compute_reference(self.initialized,
                                                   self.reference[key],
                                                   metric=metric,
                                                   comparison=comparison,
                                                   nlags=nlags,
                                                   return_p=return_p)
                return skill

    def compute_persistence(self, refname=None, nlags=None,
                            metric='pearson_r'):
        # TODO: Make this into a _ function
        if len(self.reference) == 0:
            raise ValueError("""You need to add a reference dataset before
            attempting to compute persistence forecasts.""")
        if nlags is None:
            nlags = self.initialized.time.size
        if refname is not None:
            return compute_persistence(self.reference[refname],
                                       nlags=nlags,
                                       metric=metric)
        else:
            persistence = {}
            for key in self.reference:
                persistence[key] = compute_persistence(self.reference[key],
                                                       nlags=nlags,
                                                       metric=metric)
        return persistence
