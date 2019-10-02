import xarray as xr

from .bootstrap import (
    bootstrap_perfect_model,
    bootstrap_uninit_pm_ensemble_from_control,
)
from .checks import (
    has_dims,
    is_initialized,
    is_xarray,
    match_initialized_dims,
    match_initialized_vars,
)
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
    header = f'<climpred.{type(self).__name__}>'
    summary = header + '\nInitialized Ensemble:\n'
    summary += '    ' + str(self.initialized.data_vars)[18:].strip() + '\n'
    if isinstance(self, HindcastEnsemble):
        if any(self.reference):
            for key in self.reference:
                summary += f'{key}:\n'
                N = len(self.reference[key].data_vars)
                for i in range(1, N + 1):
                    summary += (
                        '    '
                        + str(self.reference[key].data_vars).split('\n')[i].strip()
                        + '\n'
                    )
        else:
            summary += 'References:\n'
            summary += '    None\n'
    elif isinstance(self, PerfectModelEnsemble):
        summary += 'Control:\n'
        if any(self.control):
            N = len(self.control.data_vars)
            for i in range(1, N + 1):
                summary += (
                    '    ' + str(self.control.data_vars).split('\n')[i].strip() + '\n'
                )
        else:
            summary += '    None\n'
    if any(self.uninitialized):
        summary += 'Uninitialized:\n'
        summary += '    ' + str(self.uninitialized.data_vars)[18:].strip()
    else:
        summary += 'Uninitialized:\n'
        summary += '    None'
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

    @is_xarray(1)
    def __init__(self, xobj):
        if isinstance(xobj, xr.DataArray):
            # makes applying prediction functions easier, etc.
            xobj = xobj.to_dataset()
        has_dims(xobj, ['init', 'lead'], 'PredictionEnsemble')
        self._datasets = {'initialized': xobj}
        # Reserve sub-dictionary for an uninitialized run.
        self._datasets.update({'uninitialized': {}})

    # when you just print it interactively
    # https://stackoverflow.com/questions/1535327/how-to-print-objects-of-class-using-print
    def __repr__(self):
        # REINSTATE THIS.
        return 'Printing is temporarily disabled.'
        # return _display_metadata(self)

    def __getattr__(self, name):
        """Allows for xarray methods to be applied to our prediction objects.

        Args:
            * name: Function, e.g., .isel() or .sum().
        """
        # Temporarily registers attribute with the object.
        setattr(self, name, self)

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

                Ref: https://stackoverflow.com/questions/1528237/
                how-to-handle-exceptions-in-a-list-comprehensions
                """
                try:
                    return getattr(v, name)(*args, **kwargs)
                except ValueError:
                    return v

            # Create temporary copy to modify to avoid inplace operation.
            datasets = self._datasets.copy()
            # Apply this arbitrary function to our nested dictionary,
            # based on https://stackoverflow.com/questions/17915117/
            # nested-dictionary-comprehension-python
            datasets = {
                outer_k: {
                    inner_k: _apply_func(inner_v, name, *args, **kwargs)
                    for inner_k, inner_v in outer_v.items()
                }
                for outer_k, outer_v in self._datasets.items()
            }
            # Instantiates new object with the modified datasets.
            return self._construct_direct(datasets)

        # Remove registered attribute.
        delattr(self, name)
        return wrapper

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

    def get_initialized(self):
        """Returns the xarray dataset for the initialized ensemble."""
        return self._datasets['initialized']

    def get_uninitialized(self):
        """Returns the xarray dataset for the uninitialized ensemble."""
        return self._datasets['uninitialized']

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
            print(non_time_dims, 'non_time_dims')
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
        # Apply if uninitialized, control, reference exist.
        if self._datasets['uninitialized']:
            datasets['uninitialized'] = smooth_fct(
                self._datasets['uninitialized'], smooth_kws
            )
        if type(self).__name__ == 'PerfectModelEnsemble':
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

    # ---------------
    # Magic Functions
    # ---------------
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
        """Helper function to loop through references and apply an arbitrary climpred function.

        Args:
            func (function): climpred function to apply to object.
            input_dict (dict): dictionary with the following things:
                * ensemble: initialized or uninitialized ensemble.
                * control: control dictionary from HindcastEnsemble.
                * var: name of variable to target.
        """
        ensemble = input_dict['ensemble']
        control = input_dict['control']
        var = input_dict['var']

        # Compute for single variable.
        if var is not None:
            return func(ensemble[var], control[var], **kwargs)
        # Compute for all variables in control.
        else:
            return func(ensemble, control, **kwargs)

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
        self._datasets.update({'control': xobj})

    def generate_uninitialized(self, var=None):
        """Generate an uninitialized ensemble by bootstrapping the
        initialized prediction ensemble.

        Args:
            var (str, default None):
              Name of variable to be bootstrapped.

        Returns:
            Bootstrapped (uninitialized) ensemble as a Dataset.
        """
        if var is not None:
            uninit = bootstrap_uninit_pm_ensemble_from_control(
                self._datasets['initialized'][var], self._datasets['control'][var]
            ).to_dataset()
        else:
            uninit = bootstrap_uninit_pm_ensemble_from_control(
                self._datasets['initialized'], self._datasets['control']
            )
        self._datasets.update({'uninitialized': uninit})

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
        is_initialized(self._datasets['control'], 'control', 'predictability')
        return compute_perfect_model(
            self._datasets['initialized'],
            self._datasets['control'],
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
        is_initialized(
            self._datasets['uninitialized'],
            'uninitialized',
            'an uninitialized comparison',
        )
        return compute_perfect_model(
            self._datasets['uninitialized'],
            self._datasets['control'],
            metric=metric,
            comparison=comparison,
        )

    def compute_persistence(self, metric='pearson_r'):
        """Compute a simple persistence forecast for the control run.

        Args:
            metric (str, default 'pearson_r'):
              Metric to apply to the persistence forecast.

        Returns:
            Dataset of persistence forecast results (if refname is declared),
            or dictionary of Datasets with keys corresponding to reference
            name.

        Reference:
            * Chapter 8 (Short-Term Climate Prediction) in
              Van den Dool, Huug. Empirical methods in short-term climate
              prediction. Oxford University Press, 2007.
        """
        is_initialized(self._datasets['control'], 'control', 'a persistence forecast')
        return compute_persistence(
            self._datasets['initialized'], self._datasets['control'], metric=metric
        )

    def bootstrap(
        self,
        var=None,
        metric='pearson_r',
        comparison='m2e',
        sig=95,
        bootstrap=500,
        pers_sig=None,
    ):
        """Bootstrap ensemble simulations with replacement.

        Args:
            var (str, default None):
                Variable to apply bootstrapping to.
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
                * p_uninit_over_init: p-value of the hypothesis that the
                    difference of skill between the initialized and
                    uninitialized simulations is smaller or equal to zero
                    based on bootstrapping with replacement.
                * p_pers_over_init: p-value of the hypothesis that the
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
        is_initialized(self._datasets['control'], 'control', 'a bootstrap')
        input_dict = {
            'ensemble': self._datasets['initialized'],
            'control': self._datasets['control'],
            'var': var,
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
    simulations/observations associated with the prediction ensemble for easy
    computation across multiple variables and products.

    This object is built on `xarray` and thus requires the input object to
    be an `xarray` Dataset or DataArray.
    """

    # ---------------
    # Magic Functions
    # ---------------
    def __init__(self, xobj):
        """Create a `HindcastEnsemble` object by inputting output from a
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
        self._datasets.update({'reference': {}})

    # ----------------
    # Helper Functions
    # ----------------
    def _apply_climpred_function(self, func, input_dict=None, **kwargs):
        """Helper function to loop through references and apply an arbitrary climpred function.

        Args:
            func (function): climpred function to apply to object.
            input_dict (dict): dictionary with the following things:
                * ensemble: initialized or uninitialized ensemble.
                * reference: reference dictionary from HindcastEnsemble.
                * refname: name of reference to target.
                * init: bool of whether or not it's the initialized ensemble.
        """
        ensemble = input_dict['ensemble']
        reference = input_dict['reference']
        refname = input_dict['refname']
        init = input_dict['init']

        # Apply only to specific reference.
        if refname is not None:
            drop_init, drop_ref = self._vars_to_drop(refname, init=init)
            ensemble = ensemble.drop(drop_init)
            reference = reference[refname].drop(drop_ref)
            return func(ensemble, reference, **kwargs)
        else:
            # If only one reference, just apply to that one.
            if len(reference) == 1:
                refname = list(reference.keys())[0]
                drop_init, drop_ref = self._vars_to_drop(refname, init=init)
                return func(ensemble, reference[refname], **kwargs)
            # Loop through references, apply function, and store in dictionary.
            # TODO: Parallelize this process.
            else:
                result = {}
                for refname, _ in reference.items():
                    drop_init, drop_ref = self._vars_to_drop(refname, init=init)
                    result[refname] = func(ensemble, reference[refname], **kwargs)
                return result

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
            init_vars = [var for var in self._datasets['initialized'].data_vars]
        else:
            init_vars = [var for var in self._datasets['uninitialized'].data_vars]
        ref_vars = [var for var in self._datasets['reference'][ref].data_vars]
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

    # ---------------
    # Object Builders
    # ---------------
    @is_xarray(1)
    def add_reference(self, xobj, name):
        """Add a reference product for comparison to the initialized ensemble.

        NOTE: There is currently no check to ensure that these objects cover
        the same time frame.

        Args:
            xobj (xarray object): Dataset/DataArray being appended to the
                                  `HindcastEnsemble` object.
            name (str): Name of this object (e.g., "reconstruction")
        """
        if isinstance(xobj, xr.DataArray):
            xobj = xobj.to_dataset()
        match_initialized_dims(self._datasets['initialized'], xobj)
        match_initialized_vars(self._datasets['initialized'], xobj)

        def wrapper(xobj, name):
            datasets = self._datasets.copy()
            datasets['reference'].update({name: xobj})
            return self._construct_direct(datasets)

        return wrapper(xobj, name)

    @is_xarray(1)
    def add_uninitialized(self, xobj):
        """Add a companion uninitialized ensemble for comparison to references.

        NOTE: There is currently no check to ensure that these objects cover
        the same time frame as the initialized ensemble.

        Args:
            xobj (xarray object): Dataset/DataArray of the uninitialzed
                                  ensemble.
        """
        if isinstance(xobj, xr.DataArray):
            xobj = xobj.to_dataset()
        match_initialized_dims(self._datasets['initialized'], xobj, uninitialized=True)
        match_initialized_vars(self._datasets['initialized'], xobj)
        self._datasets.update({'uninitialized': xobj})

    # -----------------
    # Getters & Setters
    # -----------------
    def get_reference(self, name=None):
        """Returns the given reference(s).

        Args:
            name (str): Name of the reference to return (optional)

        Returns:
            Dictionary of xarray datasets (if name is ``None``) or single xarray
            dataset.
        """
        if name is None:
            return self._datasets['reference']
        else:
            return self._datasets['reference'][name]

    # ------------------
    # Analysis Functions
    # ------------------
    def compute_metric(
        self, refname=None, metric='pearson_r', comparison='e2r', max_dof=False
    ):
        """Compares the initialized ensemble to a given reference.

        This will automatically run the comparison against all shared variables
        between the initialized ensemble and reference.

        Args:
            refname (str):
              Name of reference to compare to. If `None`, compare to all
              references.
            metric (str, default 'pearson_r'):
              Metric to apply in the comparison.
            comparison (str, default 'e2r'):
              How to compare to the reference. ('e2r' for ensemble mean to
              reference. 'm2r' for each individual member to reference)
            max_dof (bool, default False):
              If True, maximize the degrees of freedom for each lag calculation.

        Returns:
            Dataset of comparison results (if comparing to one reference),
            or dictionary of Datasets with keys corresponding to reference
            name.
        """
        is_initialized(self._datasets['reference'], 'reference', 'predictability')
        input_dict = {
            'ensemble': self._datasets['initialized'],
            'reference': self._datasets['reference'],
            'refname': refname,
            'init': True,
        }
        return self._apply_climpred_function(
            compute_hindcast,
            input_dict=input_dict,
            metric=metric,
            comparison=comparison,
            max_dof=max_dof,
        )

    def compute_uninitialized(self, refname=None, metric='pearson_r', comparison='e2r'):
        """Compares the uninitialized ensemble to a given reference.

        This will automatically run the comparison against all shared variables
        between the initialized ensemble and reference.

        Args:
            refname (str):
              Name of reference to compare to. If `None`, compare to all
              references.
            metric (str, default 'pearson_r'):
              Metric to apply in the comparison.
            comparison (str, default 'e2r'):
              How to compare to the reference. ('e2r' for ensemble mean to
              reference. 'm2r' for each individual member to reference)

        Returns:
            Dataset of comparison results (if comparing to one reference),
            or dictionary of Datasets with keys corresponding to reference
            name.
        """
        is_initialized(
            self._datasets['uninitialized'],
            'uninitialized',
            'an uninitialized comparison',
        )
        input_dict = {
            'ensemble': self._datasets['uninitialized'],
            'reference': self._datasets['reference'],
            'refname': refname,
            'init': False,
        }
        return self._apply_climpred_function(
            compute_uninitialized,
            input_dict=input_dict,
            metric=metric,
            comparison=comparison,
        )

    def compute_persistence(self, refname=None, metric='pearson_r', max_dof=False):
        """Compute a simple persistence forecast for a reference.

        This simply applies some metric between the reference and itself out
        to some lag (i.e., an ACF in the case of pearson r).

        Args:
            refname (str, default None):
              Name of reference to compute the persistence forecast for. If
              `None`, compute for all references.
            metric (str, default 'pearson_r'):
              Metric to apply to the persistence forecast.
            max_dof (bool, default False):
              If True, maximize the degrees of freedom for each lag calculation.

        Returns:
            Dataset of persistence forecast results (if refname is declared),
            or dictionary of Datasets with keys corresponding to reference
            name.

        Reference:
            * Chapter 8 (Short-Term Climate Prediction) in
              Van den Dool, Huug. Empirical methods in short-term climate
              prediction. Oxford University Press, 2007.
        """
        is_initialized(
            self._datasets['reference'], 'reference', 'a persistence forecast'
        )
        input_dict = {
            'ensemble': self._datasets['initialized'],
            'reference': self._datasets['reference'],
            'refname': refname,
            'init': True,
        }
        return self._apply_climpred_function(
            compute_persistence, input_dict=input_dict, metric=metric, max_dof=max_dof
        )
