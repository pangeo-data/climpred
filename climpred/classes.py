import xarray as xr
from .prediction import (compute_reference, compute_persistence,
                         compute_perfect_model, compute_persistence_pm,
                         compute_uninitialized)
from .bootstrap import bootstrap_perfect_model
# Both:
# TODO: add horizon functionality
# TODO: add various `get` and `set` decorators
# TODO: add checks for our package naming conventions. I.e., should
# have 'member', 'initialization', etc. Can do this after updating the
# terminology.
# TODO: allow user to only compute things for one variable. I.e., if the
# PredictionEnsemble has multiple variables, maybe you only want to compute
# for one.
# TODO: For attributes, don't want them spit out for every `print(dp)` call.
# Maybe have a decorator under PredictionEnsemble that is .get_attr()
# TODO: Add attributes to the PredictionEnsemble that will change behavior
# for some functions. E.g.:
# temporal_resolution = 'annual'
# TODO: Add attributes to returned objects. E.g., 'skill' should come back
# with attribute explaining what two things were compared.

# PerfectModel:
# TODO: add relative entropy functionality

# Reference
# TODO: make sure that comparison 'm2r' works (i.e., allow for the ensemble
# members to be on DPLE and not just the mean)

# PerfectModel:
# TODO: add relative entropy functionality

# Reference
# TODO: make sure that comparison 'm2r' works (i.e., allow for the ensemble
# members to be on DPLE and not just the mean)


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
    for 'time' and 'member'"""
    init_dims = list(init.dims)
    if 'time' in init_dims:
        init_dims.remove('time')
    if 'member' in init_dims:
        init_dims.remove('member')
    if not (set(ref.dims) == set(init_dims)):
        raise ValueError("""Reference dimensions must match initialized
            prediction ensemble dimensions (excluding `time` and `member`.)""")


def _check_control_dimensions(init, control):
    """Checks that the control matches all initialized prediction ensemble
    dimensions except for `initialization` and `member`.

    NOTE: This needs to be merged with `_check_reference_dimensions` following
    refactoring. The dimension language is confusing, since control expects
    'time' and reference expects 'initialization'."""
    init_dims = list(init.dims)
    if 'initialization' in init_dims:
        init_dims.remove('initialization')
    if 'member' in init_dims:
        init_dims.remove('member')
    if not (set(control.dims) == set(init_dims)):
        raise ValueError("""Control dimensions must match initialized
            prediction ensemble dimensions (excluding `initialization` and
            `member`.)""")


def _check_reference_vars_match_initialized(init, ref):
    """
    Checks that a new reference (or control) dataset has at least one variable
    in common with the initialized dataset. This ensures that they can be
    compared pairwise.
    ref: new addition
    init: dp.initialized
    """
    init_list = [var for var in init.data_vars]
    ref_list = [var for var in ref.data_vars]
    # https://stackoverflow.com/questions/10668282/
    # one-liner-to-check-if-at-least-one-item-in-list-exists-in-another-list
    if set(init_list).isdisjoint(ref_list):
        raise ValueError("""Please provide a Dataset/DataArray with at least
        one matching variable to the initialized prediction ensemble.""")


def _check_xarray(xobj):
    if not isinstance(xobj, (xr.Dataset, xr.DataArray)):
        raise ValueError("""You must input an xarray Dataset or DataArray.""")


# ----------
# Aesthetics
# ----------
def _display_metadata(self):
    """
    This is called in the following case:

    ```
    dp = cp.ReferenceEnsemble(dple)
    print(dp)
    ```
    """
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
        if any(self.uninitialized):
            summary += '\nUninitialized:\n'
            summary += '    ' + str(self.uninitialized.data_vars)[18:].strip()
        else:
            summary += '\nUninitialized:\n'
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
    """
    The main object. This is the super of both `PerfectModelEnsemble` and
    `ReferenceEnsemble`. This cannot be called directly by a user, but
    should house functions that both ensemble types can use.
    """
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
        _check_control_dimensions(self.initialized, xobj)
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
        return compute_persistence_pm(self.initialized,
                                      self.control,
                                      nlags=nlags,
                                      metric=metric)

    def bootstrap(self, metric='rmse', comparison='m2m', reference_period='MK',
                  sig=95, bootstrap=30):
        """
        NOTE: This was written for an old bootstrap function. Needs to be
        updated with the newer one.
        """
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
    """An object for climate prediction ensembles initialized by a data-like
    product.

    `ReferenceEnsemble` is a sub-class of `PredictionEnsemble`. It tracks all
    simulations/observations associated with the prediction ensemble for easy
    computation across multiple variables and products.

    This object is built on `xarray` and thus requires the input object to
    be an `xarray` Dataset or DataArray.
    """
    def __init__(self, xobj):
        """Create a `ReferenceEnsemble` object by inputting output from a
        prediction ensemble in `xarray` format.

        Args:
          xobj (xarray object):
            decadal prediction ensemble output.

        Attributes:
          reference: Dictionary of various reference observations/simulations
                     to associate with the decadal prediction ensemble.
          uninitialized: Dictionary of companion (or bootstrapped)
                         uninitialized ensemble run.
        """
        super().__init__(xobj)
        self.reference = {}
        self.uninitialized = {}

    def _vars_to_drop(self, ref, init=True):
        """Returns list of variables to drop when comparing
        initialized/uninitialized to a reference.

        This is useful if the two products being compared do not share the same
        variables. I.e., if the reference has ['SST'] and the initialized has
        ['SST', 'SALT'], this will return a list with ['SALT'] to be dropped
        from the initialized.

        Args:
          ref (str):
            Name of reference being compared to.
          init (bool, default True):
            If `True`, check variables on the initialized.
            If `False`, check variables on the uninitialized.

        Returns:
          Lists of variables to drop from the initialized/uninitialized
          and reference Datasets.
        """
        if init:
            init_vars = [var for var in self.initialized.data_vars]
        else:
            init_vars = [var for var in self.uninitialized.data_vars]
        ref_vars = [var for var in self.reference[ref].data_vars]
        # find what variable they have in common.
        intersect = set(ref_vars).intersection(init_vars)
        # perhaps could be done cleaner than this.
        for var in intersect:
            # generates a list of variables to drop from each product being
            # compared.
            idx = init_vars.index(var)
            init_vars.pop(idx)
            idx = ref_vars.index(var)
            ref_vars.pop(idx)
        return init_vars, ref_vars

    def add_reference(self, xobj, name):
        """Add a reference product for comparison to the initialized ensemble.

        NOTE: There is currently no check to ensure that these objects cover
        the same time frame.

        Args:
            xobj (xarray object): Dataset/DataArray being appended to the
                                  `ReferenceEnsemble` object.
            name (str): Name of this object (e.g., "reconstruction")
        """
        _check_xarray(xobj)
        if isinstance(xobj, xr.DataArray):
            xobj = xobj.to_dataset()
        # TODO: Make sure everything is the same length. Can add keyword
        # to autotrim to the common timeframe?
        _check_reference_dimensions(self.initialized, xobj)
        _check_reference_vars_match_initialized(self.initialized, xobj)
        self.reference[name] = xobj

    def add_uninitialized(self, xobj):
        """
        This will be a special case for a complimentary uninitialized
        simulation, like LENS for DPLE.

        There should be complimentary functions for uninitialized skill.
        """
        _check_xarray(xobj)
        if isinstance(xobj, xr.DataArray):
            xobj = xobj.to_dataset()
        _check_reference_dimensions(self.initialized, xobj)
        _check_reference_vars_match_initialized(self.initialized, xobj)
        self.uninitialized = xobj

    def compute_skill(self, refname=None, metric='pearson_r', comparison='e2r',
                      nlags=None, return_p=False):
        """
        Add docstring here.
        """
        # TODO: Check that p-value return is easy on the user.
        if len(self.reference) == 0:
            raise ValueError("""You need to add a reference dataset before
                attempting to compute predictability.""")
        if refname is not None:
            drop_init, drop_ref = self._vars_to_drop(refname)
            return compute_reference(self.initialized.drop(drop_init),
                                     self.reference[refname].drop(drop_ref),
                                     metric=metric,
                                     comparison=comparison,
                                     nlags=nlags,
                                     return_p=return_p)
        else:
            if len(self.reference) == 1:
                refname = list(self.reference.keys())[0]
                drop_init, drop_ref = self._vars_to_drop(refname)
                return compute_reference(self.initialized.drop(drop_init),
                                         self.reference[refname]
                                             .drop(drop_ref),
                                         metric=metric,
                                         comparison=comparison,
                                         nlags=nlags,
                                         return_p=return_p)
            else:
                skill = {}
                for key in self.reference:
                    drop_init, drop_ref = self._vars_to_drop(key)
                    skill[key] = compute_reference(self.initialized
                                                       .drop(drop_init),
                                                   self.reference[key]
                                                       .drop(drop_ref),
                                                   metric=metric,
                                                   comparison=comparison,
                                                   nlags=nlags,
                                                   return_p=return_p)
                return skill

    def compute_uninitialized(self, refname=None, nlags=None,
                              metric='pearson_r', comparison='e2r',
                              return_p=False):
        # TODO: Check that p-value return is easy on the user.
        if len(self.uninitialized) == 0:
            raise ValueError("""You need to add an uninitialized ensemble
                before attempting to compute its skill.""")
        if refname is not None:
            drop_un, drop_ref = self._vars_to_drop(refname, init=False)
            return compute_uninitialized(self.uninitialized.drop(drop_un),
                                         self.reference[refname]
                                             .drop(drop_ref),
                                         metric=metric,
                                         comparison=comparison,
                                         return_p=return_p,
                                         dim='initialization')
        else:
            if len(self.reference) == 1:
                refname = list(self.reference.keys())[0]
                drop_un, drop_ref = self._vars_to_drop(refname,
                                                       init=False)
                return compute_uninitialized(self.uninitialized
                                                 .drop(drop_un),
                                             self.reference[refname]
                                                 .drop(drop_ref),
                                             metric=metric,
                                             comparison=comparison,
                                             return_p=return_p,
                                             dim='initialization')
            else:
                u = {}
                for key in self.reference:
                    drop_un, drop_ref = self._vars_to_drop(key,
                                                           init=False)
                    u[key] = compute_uninitialized(self.uninitialized
                                                       .drop(drop_un),
                                                   self.reference[key]
                                                       .drop(drop_ref),
                                                   metric=metric,
                                                   comparison=comparison,
                                                   return_p=return_p,
                                                   dim='initialization')
                return u

    def compute_persistence(self, refname=None, nlags=None,
                            metric='pearson_r'):
        if len(self.reference) == 0:
            raise ValueError("""You need to add a reference dataset before
            attempting to compute persistence forecasts.""")
        if nlags is None:
            nlags = self.initialized.time.size
        if refname is not None:
            return compute_persistence(self.initialized,
                                       self.reference[refname],
                                       nlags=nlags,
                                       metric=metric)
        else:
            persistence = {}
            for key in self.reference:
                persistence[key] = compute_persistence(self.initialized,
                                                       self.reference[key],
                                                       nlags=nlags,
                                                       metric=metric)
            return persistence

    def compute_horizon(self, refname=None,):
        """
        Method to compute the predictability horizon.
        """
        raise NotImplementedError("""Predictability horizons are not yet fully
            implemented and tested.""")
