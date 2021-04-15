OPTIONS = {"seasonality": "dayofyear"}  # defaults

_SEASONALITY_OPTIONS = frozenset(["dayofyear", "weekofyear", "month"])

_VALIDATORS = {
    "seasonality": _SEASONALITY_OPTIONS.__contains__,
}


class set_options:
    """Set options for climpred in a controlled context. Analogous to
    `xarrayset_options(**option) <http://xarray.pydata.org/en/stable/generated/xarray.set_options.html>`_.

    Currently supported options:

        * ``seasonality``: Attribute to group dimension ``groupby(f"{dim}.{seasonality}"")``.
            Used in ``reference=climatology`` and :py:meth:`~climpred.classes.HindcastEnsemble.remove_bias`.
                - Allowed: ["dayofyear", "weekofyear", "month"]
                - Default: ``dayofyear``.

    Examples:
        You can use ``set_options`` either as a context manager:

        >>> kw = dict(metric='mse', comparison='e2o', dim='init', alignment='same_verifs',
        ...           reference='climatology')
        >>> with climpred.set_options(seasonality='month'):
        ...     HindcastEnsemble.verify(**kw).SST.sel(skill='climatology')
        <xarray.DataArray 'SST' (lead: 10)>
        array([0.03712573, 0.03712573, 0.03712573, 0.03712573, 0.03712573,
               0.03712573, 0.03712573, 0.03712573, 0.03712573, 0.03712573])
        Coordinates:
          * lead     (lead) int32 1 2 3 4 5 6 7 8 9 10
            skill    <U11 'climatology'
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
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)
