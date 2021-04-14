OPTIONS = {"seasonality": "dayofyear"}  # defaults

_SEASONALITY_OPTIONS = frozenset(["dayofyear", "weekofyear", "month"])

_VALIDATORS = {
    "seasonality": _SEASONALITY_OPTIONS.__contains__,
}


class set_options:
    """Set options for climpred in a controlled context. Similar to xr.set_otions.

    Currently supported options:
    - ``seasonality``: maximum display width for ``repr`` on xarray objects.
      Default: ``dayofyear``.

    You can use ``set_options`` either as a context manager:
    >>> ds = xr.Dataset({"x": np.arange(1000)})
    >>> with climpred.set_options(seasonality='monthofyear'):
    ...     HindcastEnsemble.verify(metric='mse', comparison='e2o', dim='init', alignment='same_verifs',reference='climatology')
    >>> with climpred.set_options(seasonality='dayofyear'):
    ...     HindcastEnsemble.verify(metric='mse', comparison='e2o', dim='init', alignment='same_verifs',reference='climatology')
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
