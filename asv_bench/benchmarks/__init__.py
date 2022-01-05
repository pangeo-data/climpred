# https://github.com/pydata/xarray/blob/master/asv_bench/benchmarks/__init__.py
import itertools
import os

import dask
import numpy as np

_counter = itertools.count()


def parameterized(names, params):
    def decorator(func):
        func.param_names = names
        func.params = params
        return func

    return decorator


def requires_dask():
    try:
        import dask  # noqa
    except ImportError:
        raise NotImplementedError


def randn(shape, frac_nan=None, chunks=None, seed=0):
    rng = np.random.RandomState(seed)
    if chunks is None:
        x = rng.standard_normal(shape)
    else:
        import dask.array as da

        rng = da.random.RandomState(seed)
        x = rng.standard_normal(shape, chunks=chunks)

    if frac_nan is not None:
        inds = rng.choice(range(x.size), int(x.size * frac_nan))
        x.flat[inds] = np.nan

    return x


def randint(low, high=None, size=None, frac_minus=None, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.randint(low, high, size)
    if frac_minus is not None:
        inds = rng.choice(range(x.size), int(x.size * frac_minus))
        x.flat[inds] = -1

    return x


def ensure_loaded(res):
    """Compute no lazy results."""
    if dask.is_dask_collection(res):
        res = res.compute()
    return res


def _skip_slow():
    """
    Use this function to skip slow or highly demanding tests.
    Use it as a `Class.setup` method or a `function.setup` attribute.
    Examples
    --------
    >>> from . import _skip_slow
    >>> def time_something_slow():
    ...     pass
    ...
    >>> time_something.setup = _skip_slow
    """
    if os.environ.get("ASV_SKIP_SLOW", "0") == "1":
        raise NotImplementedError("Skipping this test...")
