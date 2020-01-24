import warnings

import xarray as xr

from .bootstrap import (
    bootstrap_perfect_model,
    bootstrap_uninit_pm_ensemble_from_control,
)
from .checks import (
    has_dataset,
    has_dims,
    has_valid_lead_units,
    is_xarray,
    match_initialized_dims,
    match_initialized_vars,
)
from .exceptions import DimensionError
from .prediction import (
    compute_hindcast,
    compute_perfect_model,
    compute_persistence,
    compute_uninitialized,
)
from .smoothing import (
    smooth_goddard_2013,
    spatial_smoothing_xesmf,
    spatial_smoothing_xrcoarsen,
    temporal_smoothing,
)
from .utils import convert_time_index


# ----------
# Aesthetics
# ----------
def _display_metadata(self):
    """
    This is called in the following case:

    ```
    dp = cp.HindcastEnsemble(dple)
    print(dp)
    ```
    """
    SPACE = '    '
    header = f'<climpred.{type(self).__name__}>'
    summary = header + '\nInitialized Ensemble:\n'
    summary += SPACE + str(self._datasets['initialized'].data_vars)[18:].strip() + '\n'
    if isinstance(self, HindcastEnsemble):
        # Prints out verification data names and associated variables if they exist.
        # If not, just write "None".
        if any(self._datasets['observations']):
            for key in self._datasets['observations']:
                summary += f'{key}:\n'
                num_obs = len(self._datasets['observations'][key].data_vars)
                for i in range(1, num_obs + 1):
                    summary += (
                        SPACE
                        + str(self._datasets['observations'][key].data_vars)
                        .split('\n')[i]
                        .strip()
                        + '\n'
                    )
        else:
            summary += 'Verification Data:\n'
            summary += SPACE + 'None\n'
    elif isinstance(self, PerfectModelEnsemble):
        summary += 'Control:\n'
        # Prints out control variables if a control is appended. If not,
        # just write "None".
        if any(self._datasets['control']):
            num_ctrl = len(self._datasets['control'].data_vars)
            for i in range(1, num_ctrl + 1):
                summary += (
                    SPACE
                    + str(self._datasets['control'].data_vars).split('\n')[i].strip()
                    + '\n'
                )
        else:
            summary += SPACE + 'None\n'
    if any(self._datasets['uninitialized']):
        summary += 'Uninitialized:\n'
        summary += SPACE + str(self._datasets['uninitialized'].data_vars)[18:].strip()
    else:
        summary += 'Uninitialized:\n'
        summary += SPACE + 'None'
    return summary


# -----------------
# CLASS DEFINITIONS
# -----------------
class PredictionEnsemble:
    """
    The main object. This is the super of both `PerfectModelEnsemble` and
    `HindcastEnsemble`. This cannot be called directly by a user, but
    should house functions that both ensemble types can use.
    """

    # -------------
    # Magic Methods
    # -------------
    @is_xarray(1)
    def __init__(self, xobj):
        if isinstance(xobj, xr.DataArray):
            # makes applying prediction functions easier, etc.
            xobj = xobj.to_dataset()
        has_dims(xobj, ['init', 'lead'], 'PredictionEnsemble')
        # Check that init is int, cftime, or datetime; convert ints or cftime to
        # datetime.
        xobj = convert_time_index(xobj, 'init', 'xobj[init]')
        # Put this after `convert_time_index` since it assigns 'years' attribute if the
        # `init` dimension is a `float` or `int`.
        has_valid_lead_units(xobj)
        # Add initialized dictionary and reserve sub-dictionary for an uninitialized
        # run.
        self._datasets = {'initialized': xobj, 'uninitialized': {}}

    # when you just print it interactively
    # https://stackoverflow.com/questions/1535327/how-to-print-objects-of-class-using-print
    def __repr__(self):
        return _display_metadata(self)

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

            def _apply_func(v, name, *args, **kwargs):
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
                # it to the given dataset if the dimension doesn't exist.
                #
                # DimensionError: This accounts for our custom error when applying
                # some stats functions.
                except (ValueError, DimensionError):
                    return v

            # Create temporary copy to modify to avoid inplace operation.
            datasets = self._datasets.copy()

            # More explicit than nested dictionary comprehension.
            for outer_k, outer_v in datasets.items():
                # If initialized, control, uninitialized and just a singular
                # dataset, apply the function directly to it.
                if isinstance(outer_v, xr.Dataset):
                    datasets.update(
                        {outer_k: _apply_func(outer_v, name, *args, **kwargs)}
                    )
                else:
                    # If a nested dictionary is encountered (i.e., a set of
                    # observations) apply to each individually.
                    #
                    # Similar to the ``add_observations`` method, this only seems to
                    # avoid inplace operations by copying the nested dictionary
                    # separately and then updating the main dictionary.
                    temporary_dataset = self._datasets[outer_k].copy()
                    for inner_k, inner_v in temporary_dataset.items():
                        temporary_dataset.update(
                            {inner_k: _apply_func(inner_v, name, *args, **kwargs)}
                        )
                    datasets.update({outer_k: temporary_dataset})
            # Instantiates new object with the modified datasets.
            return self._construct_direct(datasets)

        return wrapper

    # ----------------
    # Helper Functions
    # ----------------
    @classmethod
    def _construct_direct(cls, datasets):
        """Shortcut around __init__ for internal use to avoid inplace
        operations.

        Pulled from xarrray Dataset class.
        https://github.com/pydata/xarray/blob/master/xarray/core/dataset.py
        """
        obj = object.__new__(cls)
        obj._datasets = datasets
        return obj

    # -----------------
    # Getters & Setters
    # -----------------
    def get_initialized(self):
        """Returns the xarray dataset for the initialized ensemble."""
        return self._datasets['initialized']

    def get_uninitialized(self):
        """Returns the xarray dataset for the uninitialized ensemble."""
        return self._datasets['uninitialized']

    # ------------------
    # Analysis Functions
    # ------------------
    def smooth(self, smooth_kws='goddard2013'):
        """Smooth all entries of PredictionEnsemble in the same manner to be
        able to still calculate prediction skill afterwards.

        Args:
          xobj (xarray object):
            decadal prediction ensemble output.

        Attributes:
            smooth_kws (dict or str): Dictionary to specify the dims to
                smooth compatible with `spatial_smoothing_xesmf`,
                `temporal_smoothing` or `spatial_smoothing_xrcoarsen`.
                Shortcut for Goddard et al. 2013 recommendations:
                'goddard2013'

        Example:
        >>> PredictionEnsemble.smooth(smooth_kws={'time': 2,
            'lat': 5, 'lon': 4'})
        >>> PredictionEnsemble.smooth(smooth_kws='goddard2013')
        """
        # get proper smoothing function based on smooth args
        if isinstance(smooth_kws, str):
            if 'goddard' in smooth_kws:
                smooth_fct = smooth_goddard_2013
                smooth_kws = {'lead': 4}  # default
            else:
                raise ValueError(
                    'Please provide from list of available smoothings: \
                     ["goddard2013"]'
                )
        elif isinstance(smooth_kws, dict):
            non_time_dims = [
                dim for dim in smooth_kws.keys() if dim not in ['time', 'lead']
            ]
            if len(non_time_dims) > 0:
                non_time_dims = non_time_dims[0]
            # goddard when time_dim and lon/lat given
            if ('lon' in smooth_kws or 'lat' in smooth_kws) and (
                'lead' in smooth_kws or 'time' in smooth_kws
            ):
                smooth_fct = smooth_goddard_2013
            # fail goddard and fall back to xrcoarsen when
            # coarsen dim and time_dim provided
            elif (
                (non_time_dims is not [])
                and (non_time_dims in list(self._datasets['initialized'].dims))
                and ('lead' in smooth_kws or 'time' in smooth_kws)
            ):
                smooth_fct = smooth_goddard_2013
            # else only one smoothing operation
            elif 'lon' in smooth_kws or 'lat' in smooth_kws:
                smooth_fct = spatial_smoothing_xesmf
            elif 'lead' in smooth_kws:
                smooth_fct = temporal_smoothing
            elif non_time_dims in list(self._datasets['initialized'].dims):
                smooth_fct = spatial_smoothing_xrcoarsen
            else:
                raise ValueError(
                    'Please provide kwargs to fulfill functions: \
                     ["spatial_smoothing_xesmf", "temporal_smoothing", \
                     "spatial_smoothing_xrcoarsen"].'
                )
        else:
            raise ValueError(
                'Please provide kwargs as str or dict and not', type(smooth_kws)
            )
        # Apply throughout the dataset
        # TODO: Parallelize
        datasets = self._datasets.copy()
        datasets['initialized'] = smooth_fct(self._datasets['initialized'], smooth_kws)
        # Apply if uninitialized, control, observations exist.
        if self._datasets['uninitialized']:
            datasets['uninitialized'] = smooth_fct(
                self._datasets['uninitialized'], smooth_kws
            )
        if isinstance(self, PerfectModelEnsemble):
            if self._datasets['control']:
                datasets['control'] = smooth_fct(self._datasets['control'], smooth_kws)
        # if type(self).__name__ == 'HindcastEnsemble':
        return self._construct_direct(datasets)


class PerfectModelEnsemble(PredictionEnsemble):
    """An object for "perfect model" climate prediction ensembles.

    `PerfectModelEnsemble` is a sub-class of `PredictionEnsemble`. It tracks
    the control run used to initialize the ensemble for easy computations,
    bootstrapping, etc.

    This object is built on `xarray` and thus requires the input object to
    be an `xarray` Dataset or DataArray.
    """

    # -------------
    # Magic Methods
    # -------------
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
        self._datasets.update({'control': {}})

    # ----------------
    # Helper Functions
    # ----------------
    def _apply_climpred_function(self, func, input_dict=None, **kwargs):
        """Helper function to loop through verification data and apply an arbitrary climpred
        function.

        Args:
            func (function): climpred function to apply to object.
            input_dict (dict): dictionary with the following things:
                * ensemble: initialized or uninitialized ensemble.
                * control: control dictionary from HindcastEnsemble.
                * init (bool): True if the initialized ensemble, False if uninitialized.
        """
        ensemble = input_dict['ensemble']
        control = input_dict['control']
        init = input_dict['init']
        init_vars, ctrl_vars = self._vars_to_drop(init=init)
        ensemble = ensemble.drop_vars(init_vars)
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
        init_str = 'initialized' if init else 'uninitialized'
        init_vars = list(self._datasets[init_str])
        ctrl_vars = list(self._datasets['control'])
        # Make lists of variables to drop that aren't in common
        # with one another.
        init_vars_to_drop = list(set(init_vars) - set(ctrl_vars))
        ctrl_vars_to_drop = list(set(ctrl_vars) - set(init_vars))
        return init_vars_to_drop, ctrl_vars_to_drop

    # ---------------
    # Object Builders
    # ---------------
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
        match_initialized_dims(self._datasets['initialized'], xobj)
        match_initialized_vars(self._datasets['initialized'], xobj)
        # Check that init is int, cftime, or datetime; convert ints or cftime to
        # datetime.
        xobj = convert_time_index(xobj, 'time', 'xobj[init]')
        datasets = self._datasets.copy()
        datasets.update({'control': xobj})
        return self._construct_direct(datasets)

    def generate_uninitialized(self):
        """Generate an uninitialized ensemble by bootstrapping the
        initialized prediction ensemble.

        Returns:
            Bootstrapped (uninitialized) ensemble as a Dataset.
        """
        has_dataset(
            self._datasets['control'], 'control', 'generate an uninitialized ensemble.'
        )

        uninit = bootstrap_uninit_pm_ensemble_from_control(
            self._datasets['initialized'], self._datasets['control']
        )
        datasets = self._datasets.copy()
        datasets.update({'uninitialized': uninit})
        return self._construct_direct(datasets)

    # -----------------
    # Getters & Setters
    # -----------------
    def get_control(self):
        """Returns the control as an xarray dataset."""
        return self._datasets['control']

    # ------------------
    # Analysis Functions
    # ------------------
    def compute_metric(self, metric='pearson_r', comparison='m2m'):
        """Compares the initialized ensemble to the control run.

        Args:
            metric (str, default 'pearson_r'):
              Metric to apply in the comparison.
            comparison (str, default 'm2m'):
              How to compare the climate prediction ensemble to the control.

        Returns:
            Result of the comparison as a Dataset.
        """
        has_dataset(self._datasets['control'], 'control', 'compute a metric')
        input_dict = {
            'ensemble': self._datasets['initialized'],
            'control': self._datasets['control'],
            'init': True,
        }
        return self._apply_climpred_function(
            compute_perfect_model,
            input_dict=input_dict,
            metric=metric,
            comparison=comparison,
        )

    def compute_uninitialized(self, metric='pearson_r', comparison='m2e'):
        """Compares the bootstrapped uninitialized run to the control run.

        Args:
            metric (str, default 'pearson_r'):
              Metric to apply in the comparison.
            comparison (str, default 'm2m'):
              How to compare to the control run.
            running (int, default None):
              Size of the running window for variance smoothing.

        Returns:
            Result of the comparison as a Dataset.
        """
        has_dataset(
            self._datasets['uninitialized'],
            'uninitialized',
            'compute an uninitialized metric',
        )
        input_dict = {
            'ensemble': self._datasets['uninitialized'],
            'control': self._datasets['control'],
            'init': False,
        }
        return self._apply_climpred_function(
            compute_perfect_model,
            input_dict=input_dict,
            metric=metric,
            comparison=comparison,
        )

    def compute_persistence(self, metric='pearson_r'):
        """Compute a simple persistence forecast for the control run.

        Args:
            metric (str, default 'pearson_r'):
                Metric to apply to the persistence forecast.

        Returns:
            Dataset of persistence forecast results (if ``refname`` is declared),
            or dictionary of Datasets with keys corresponding to verification data
            name.

        Reference:
            * Chapter 8 (Short-Term Climate Prediction) in
              Van den Dool, Huug. Empirical methods in short-term climate
              prediction. Oxford University Press, 2007.
        """
        has_dataset(
            self._datasets['control'], 'control', 'compute a persistence forecast'
        )
        input_dict = {
            'ensemble': self._datasets['initialized'],
            'control': self._datasets['control'],
            'init': True,
        }
        return self._apply_climpred_function(
            compute_persistence, input_dict=input_dict, metric=metric
        )

    def bootstrap(
        self, metric='pearson_r', comparison='m2e', sig=95, bootstrap=500, pers_sig=None
    ):
        """Bootstrap ensemble simulations with replacement.

        Args:
            metric (str, default 'pearson_r'):
                Metric to apply for bootstrapping.
            comparison (str, default 'm2e'):
                Comparison style for bootstrapping.
            sig (int, default 95):
                Significance level for uninitialized and initialized
                comparison.
            bootstrap (int, default 500): Number of resampling iterations for
                bootstrapping with replacement.
            pers_sig (int, default None):
                If not None, the separate significance level for persistence.

        Returns:
            Dictionary of Datasets for each variable applied to with the
            following variables:
                * init_ci: confidence levels of init_skill.
                * uninit_ci: confidence levels of uninit_skill.
                * pers_ci: confidence levels of pers_skill.
                * p_uninit_over_init: p value of the hypothesis that the
                    difference of skill between the initialized and
                    uninitialized simulations is smaller or equal to zero
                    based on bootstrapping with replacement.
                * p_pers_over_init: p value of the hypothesis that the
                    difference of skill between the initialized and persistence
                    simulations is smaller or equal to zero based on
                    bootstrapping with replacement.

        Reference:
            * Goddard, L., A. Kumar, A. Solomon, D. Smith, G. Boer, P.
              Gonzalez, V. Kharin, et al. “A Verification Framework for
              Interannual-to-Decadal Predictions Experiments.” Climate
              Dynamics 40, no. 1–2 (January 1, 2013): 245–72.
              https://doi.org/10/f4jjvf.

        """
        has_dataset(self._datasets['control'], 'control', 'bootstrap')
        input_dict = {
            'ensemble': self._datasets['initialized'],
            'control': self._datasets['control'],
            'init': True,
        }
        return self._apply_climpred_function(
            bootstrap_perfect_model,
            input_dict=input_dict,
            metric=metric,
            comparison=comparison,
            sig=sig,
            bootstrap=bootstrap,
            pers_sig=pers_sig,
        )


class HindcastEnsemble(PredictionEnsemble):
    """An object for climate prediction ensembles initialized by a data-like
    product.

    `HindcastEnsemble` is a sub-class of `PredictionEnsemble`. It tracks all
    verification data associated with the prediction ensemble for easy
    computation across multiple variables and products.

    This object is built on `xarray` and thus requires the input object to
    be an `xarray` Dataset or DataArray.
    """

    # -------------
    # Magic Methods
    # -------------
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
        self._datasets.update({'observations': {}})

    # ----------------
    # Helper Functions
    # ----------------
    def _apply_climpred_function(self, func, input_dict=None, **kwargs):
        """Helper function to loop through verification data and apply an arbitrary
        climpred function.

        Args:
            func (function): climpred function to apply to object.
            input_dict (dict): dictionary with the following things:
                * ensemble: initialized or uninitialized ensemble.
                * observations: Dictionary of verification data from
                    ``HindcastEnsemble``.
                * name: name of verification data to target.
                * init: bool of whether or not it's the initialized ensemble.
        """
        ensemble = input_dict['ensemble']
        observations = input_dict['observations']
        name = input_dict['name']
        init = input_dict['init']

        # Apply only to specific observations.
        if name is not None:
            drop_init, drop_obs = self._vars_to_drop(name, init=init)
            ensemble = ensemble.drop_vars(drop_init)
            observations = observations[name].drop_vars(drop_obs)
            return func(ensemble, observations, **kwargs)
        else:
            # If only one observational product, just apply to that one.
            if len(observations) == 1:
                name = list(observations.keys())[0]
                drop_init, drop_obs = self._vars_to_drop(name, init=init)
                return func(ensemble, observations[name], **kwargs)
            # Loop through observations, apply function, and store in dictionary.
            # TODO: Parallelize this process.
            else:
                result = {}
                for name in observations.keys():
                    drop_init, drop_obs = self._vars_to_drop(name, init=init)
                    result[name] = func(ensemble, observations[name], **kwargs)
                return result

    def _vars_to_drop(self, name, init=True):
        """Returns list of variables to drop when comparing
        initialized/uninitialized to observations.

        This is useful if the two products being compared do not share the same
        variables. I.e., if the observations have ['SST'] and the initialized has
        ['SST', 'SALT'], this will return a list with ['SALT'] to be dropped
        from the initialized.

        Args:
          name (str): Short name of observations being compared to.
          init (bool, default True):
            If ``True``, check variables on the initialized.
            If ``False``, check variables on the uninitialized.

        Returns:
          Lists of variables to drop from the initialized/uninitialized
          and observational Datasets.
        """
        if init:
            init_vars = [var for var in self._datasets['initialized'].data_vars]
        else:
            init_vars = [var for var in self._datasets['uninitialized'].data_vars]
        obs_vars = [var for var in self._datasets['observations'][name].data_vars]
        # Make lists of variables to drop that aren't in common
        # with one another.
        init_vars_to_drop = list(set(init_vars) - set(obs_vars))
        obs_vars_to_drop = list(set(obs_vars) - set(init_vars))
        return init_vars_to_drop, obs_vars_to_drop

    # ---------------
    # Object Builders
    # ---------------
    @is_xarray(1)
    def add_observations(self, xobj, name):
        """Add a verification data with which to verify the initialized ensemble.

        Args:
            xobj (xarray object): Dataset/DataArray to append to the
                ``HindcastEnsemble`` object.
            name (str): Short name for referencing the verification data.
        """
        if isinstance(xobj, xr.DataArray):
            xobj = xobj.to_dataset()
        match_initialized_dims(self._datasets['initialized'], xobj)
        match_initialized_vars(self._datasets['initialized'], xobj)
        # Check that time is int, cftime, or datetime; convert ints or cftime to
        # datetime.
        xobj = convert_time_index(xobj, 'time', 'xobj[init]')
        # For some reason, I could only get the non-inplace method to work
        # by updating the nested dictionaries separately.
        datasets_obs = self._datasets['observations'].copy()
        datasets = self._datasets.copy()
        datasets_obs.update({name: xobj})
        datasets.update({'observations': datasets_obs})
        return self._construct_direct(datasets)

    @is_xarray(1)
    def add_reference(self, xobj, name):
        """Add an observational product with which to verify the initialized ensemble.

        Args:
            xobj (xarray object): Dataset/DataArray to append to the
                                  `HindcastEnsemble` object.
            name (str): Short name for referencing the verification data.
        """
        warnings.warn(
            '`HindcastEnsemble.add_reference()` will be deprecated. '
            'Using `HindcastEnsemble.add_observations()` is encouraged.',
            PendingDeprecationWarning,
        )
        return self.add_observations(xobj, name)

    @is_xarray(1)
    def add_uninitialized(self, xobj):
        """Add a companion uninitialized ensemble for comparison to verification data.

        Args:
            xobj (xarray object): Dataset/DataArray of the uninitialzed
                                  ensemble.
        """
        if isinstance(xobj, xr.DataArray):
            xobj = xobj.to_dataset()
        match_initialized_dims(self._datasets['initialized'], xobj, uninitialized=True)
        match_initialized_vars(self._datasets['initialized'], xobj)
        # Check that init is int, cftime, or datetime; convert ints or cftime to
        # datetime.
        xobj = convert_time_index(xobj, 'time', 'xobj[init]')
        datasets = self._datasets.copy()
        datasets.update({'uninitialized': xobj})
        return self._construct_direct(datasets)

    # -----------------
    # Getters & Setters
    # -----------------
    def get_observations(self, name=None):
        """Returns xarray Datasets of the observations/verification data.

        Args:
            name (str, optional): Name of the observations/verification data to return.
                If ``None``, return dictionary of all observations/verification data.

        Returns:
            Dictionary of ``xarray`` Datasets (if ``name`` is ``None``) or single
            ``xarray`` Dataset.
        """
        if name is None:
            if len(self._datasets['observations']) == 1:
                key = list(self._datasets['observations'].keys())[0]
                return self._datasets['observations'][key]
            else:
                return self._datasets['observations']
        else:
            return self._datasets['observations'][name]

    def get_reference(self, name=None):
        """Returns xarray Datasets of the observations/verification data.

        Args:
            name (str, optional): Name of the observations/verification data to return.
                If ``None``, return dictionary of all observations/verification data.

        Returns:
            Dictionary of ``xarray`` Datasets (if ``name`` is ``None``) or single
            ``xarray`` Dataset.
        """
        warnings.warn(
            '`HindcastEnsemble.get_reference()` will be deprecated. '
            'Using `HindcastEnsemble.get_observations()` is encouraged.',
            PendingDeprecationWarning,
        )
        return self.get_observations(name=name)

    # ------------------
    # Analysis Functions
    # ------------------
    def verify(self, name=None, metric='pearson_r', comparison='e2o', max_dof=False):
        """Verifies the initialized ensemble against observations/verification data.

        This will automatically verify against all shared variables
        between the initialized ensemble and observations/verification data.

        Args:
            name (str): Short name of observations/verification data to compare to.
                If ``None``, compare to all observations/verification data.
            metric (str, default 'pearson_r'): Metric to apply for verification.
            comparison (str, default 'e2o'): How to compare to the
                observations/verification data. ('e2o' for ensemble mean to
                observations/verification data. 'm2o' for each individual member to
                observations/verification data).
            max_dof (bool, default False): If ``True``, maximize the degrees of freedom
                for each lag calculation.

        Returns:
            Dataset of comparison results (if comparing to one observational product),
            or dictionary of Datasets with keys corresponding to
            observations/verification data short name.
        """
        has_dataset(
            self._datasets['observations'], 'observational', 'verify a forecast'
        )
        input_dict = {
            'ensemble': self._datasets['initialized'],
            'observations': self._datasets['observations'],
            'name': name,
            'init': True,
        }
        return self._apply_climpred_function(
            compute_hindcast,
            input_dict=input_dict,
            metric=metric,
            comparison=comparison,
            max_dof=max_dof,
        )

    def compute_metric(
        self, name=None, metric='pearson_r', comparison='e2o', max_dof=False
    ):
        """Verifies the initialized ensemble against observations/verification data.

        This will automatically verify against all shared variables
        between the initialized ensemble and observations/verification data.

        Args:
            name (str): Short name of observations/verification data to compare to.
                If ``None``, compare to all observations/verification data.
            metric (str, default 'pearson_r'): Metric to apply for verification.
            comparison (str, default 'e2o'): How to compare to the
                observations/verification data. ('e2o'
                for ensemble mean to observations/verification data.
                'm2o' for each individual member to observations/verification data).
            max_dof (bool, default False): If ``True``, maximize the degrees of freedom
                for each lag calculation.

        Returns:
            Dataset of comparison results (if comparing to one observational product),
            or dictionary of Datasets with keys corresponding to observations short
            name.
        """
        warnings.warn(
            '`HindcastEnsemble.compute_metric()` will be deprecated. '
            'Using `HindcastEnsemble.verify()` is encouraged.',
            PendingDeprecationWarning,
        )
        return self.verify(
            name=name, metric=metric, comparison=comparison, max_dof=max_dof
        )

    def compute_uninitialized(self, name=None, metric='pearson_r', comparison='e2o'):
        """Verifies the uninitialized ensemble against observations/verification data.

        This will automatically verify against all shared variables
        between the uninitialized ensemble and observations/verification data.

        Args:
            name (str): Short name of observations/verification data to compare to.
                If ``None``, compare to all observations/verification data.
            metric (str, default 'pearson_r'): Metric to apply for verification.
            comparison (str, default 'e2o'): How to compare to the
                observations/verification data. ('e2o' for ensemble mean to
                observations/verification data. 'm2o' for each individual member to
                observations/verification data).
            max_dof (bool, default False): If ``True``, maximize the degrees of freedom
                for each lag calculation.

        Returns:
            Dataset of comparison results (if comparing to one observational product),
            or dictionary of Datasets with keys corresponding to
            observations/verification data short name.
        """
        has_dataset(
            self._datasets['uninitialized'],
            'uninitialized',
            'compute an uninitialized metric',
        )
        input_dict = {
            'ensemble': self._datasets['uninitialized'],
            'observations': self._datasets['observations'],
            'name': name,
            'init': False,
        }
        return self._apply_climpred_function(
            compute_uninitialized,
            input_dict=input_dict,
            metric=metric,
            comparison=comparison,
        )

    def compute_persistence(self, name=None, metric='pearson_r', max_dof=False):
        """Verify against a persistence forecast of the observations/verification data.

        This simply applies some metric between the observational product and itself out
        to some lag (e.g., an ACF in the case of 'pearson_r').

        .. note::

            The persistence forecast is computed starting from the initialization date
            and moving forward one time step. Some protocols suggest that the "lead one"
            persistence forecast is actually from the time step prior to initialization.
            This will be implemented as an option in a future version of ``climpred``.

        Args:
            name (str, default None): Short name of observations/verification data
                with which to compute the persistence forecast. If ``None``, compute
                for all observations/verification data.
            metric (str, default 'pearson_r'): Metric to apply for verification.
            max_dof (bool, default False): If ``True``, maximize the degrees of freedom
                for each lag calculation.

        Returns:
            Dataset of persistence forecast results (if ``name`` is not ``None``),
            or dictionary of Datasets with keys corresponding to
            observations/verification data short name.

        Reference:
            * Chapter 8 (Short-Term Climate Prediction) in
              Van den Dool, Huug. Empirical methods in short-term climate
              prediction. Oxford University Press, 2007.
        """
        has_dataset(
            self._datasets['observations'],
            'observational',
            'compute a persistence forecast',
        )
        input_dict = {
            'ensemble': self._datasets['initialized'],
            'observations': self._datasets['observations'],
            'name': name,
            'init': True,
        }
        return self._apply_climpred_function(
            compute_persistence, input_dict=input_dict, metric=metric, max_dof=max_dof
        )
