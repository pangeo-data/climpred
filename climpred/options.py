from .constants import GROUPBY_SEASONALITIES

OPTIONS = {
    "seasonality": "month",
    "PerfectModel_persistence_from_initialized_lead_0": False,
    "warn_for_failed_PredictionEnsemble_xr_call": True,
    "warn_for_rename_to_climpred_dims": True,
    "warn_for_init_coords_int_to_annual": True,
    "climpred_warnings": True,
}  # defaults

_SEASONALITY_OPTIONS = frozenset(GROUPBY_SEASONALITIES)

_VALIDATORS = {
    "seasonality": _SEASONALITY_OPTIONS.__contains__,
    "PerfectModel_persistence_from_initialized_lead_0": lambda choice: choice
    in [True, False, "default"],
    "warn_for_failed_PredictionEnsemble_xr_call": lambda choice: choice
    in [True, False, "default"],
    "warn_for_rename_to_climpred_dims": lambda choice: choice
    in [True, False, "default"],
    "warn_for_init_coords_int_to_annual": lambda choice: choice
    in [True, False, "default"],
    "climpred_warnings": lambda choice: choice in [True, False, "default"],
}


class set_options:
    """
    Set options for ``climpred`` in a controlled context.

    Analogous to
    :py:class:`~xarray.options.set_options`.

    Args:
        ``seasonality`` : {``"dayofyear"``, ``"weekofyear"``, ``"month"``, ``"season"``}, default: ``"month"`` # noqa: E501
            Attribute to group dimension ``groupby(f"{dim}.{seasonality}"")``.
            Used in ``reference=climatology`` and
            :py:meth:`.HindcastEnsemble.remove_bias`.
        ``PerfectModel_persistence_from_initialized_lead_0`` : {``True``, ``False``}, default ``False`` # noqa: E501
            Which persistence function to use in
            ``PerfectModelEnsemble.verify/bootstrap(reference="persistence")``.
            If ``False`` use :py:func:`~climpred.reference.compute_persistence`.
            If ``True`` use
            :py:func:`~climpred.reference.compute_persistence_from_first_lead`.
        ``warn_for_failed_PredictionEnsemble_xr_call`` : {``True``, ``False``}, default ``True``
            Raise ``UserWarning`` when ``PredictionEnsemble.xr_call``, e.g.
            ``.sel(lead=[1])`` fails on one of the datasets.
        ``warn_for_rename_to_climpred_dims`` : {``True``, ``False``}, default ``True``
            Raise ``UserWarning`` when dimensions are renamed to ``CLIMPRED_DIMS`` when
            :py:class:`.PredictionEnsemble` is instantiated.
        ``warn_for_init_coords_int_to_annual`` : {``True``, ``False``}, default ``True``
            Raise ``UserWarning`` when ``init`` coordinate is of type integer and gets
            converted to annual cftime_range when
            :py:class:`.PredictionEnsemble` is instantiated.
        ``climpred_warnings`` : {``True``, ``False``}, default ``True``
            Overwrites all options containing ``"*warn*"``.

    Examples:

        You can use ``set_options`` either as a context manager:

        >>> kw = dict(
        ...     metric="mse",
        ...     comparison="e2o",
        ...     dim="init",
        ...     alignment="same_verifs",
        ...     reference="climatology",
        ... )
        >>> with climpred.set_options(seasonality="month"):
        ...     HindcastEnsemble.verify(**kw).SST.sel(skill="climatology")
        ...
        <xarray.DataArray 'SST' (lead: 10)>
        array([0.03712573, 0.03712573, 0.03712573, 0.03712573, 0.03712573,
               0.03712573, 0.03712573, 0.03712573, 0.03712573, 0.03712573])
        Coordinates:
          * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
            skill    <U11 'climatology'
        Attributes:
            units:    (C)^2

        Or to set global options:

        >>> climpred.set_options(seasonality="month")  # doctest: +ELLIPSIS
        <climpred.options.set_options object at 0x...>
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    "argument name %r is not in the set of valid options %r"
                    % (k, set(OPTIONS))
                )
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                if k == "seasonality":
                    expected = f"Expected one of {_SEASONALITY_OPTIONS!r}"
                else:
                    expected = ""
                raise ValueError(
                    f"option {k!r} given an invalid value: {v!r}. " + expected
                )
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        if (
            "climpred_warnings" in options_dict
        ):  # climpred_warnings == False overwrites all warnings options
            if not options_dict["climpred_warnings"]:
                for k in [o for o in OPTIONS.keys() if "warn" in o]:
                    options_dict[k] = False
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)
