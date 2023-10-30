import importlib

import pytest
from packaging import version


def _importorskip(modname, minversion=None):
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            if version.parse(mod.__version__) < version.parse(minversion):
                raise ImportError("Minimum version not satisfied")
    except ImportError:
        has = False
    func = pytest.mark.skipif(not has, reason=f"requires {modname}")
    return has, func


has_matplotlib, requires_matplotlib = _importorskip("matplotlib")
has_nc_time_axis, requires_nc_time_axis = _importorskip("nc_time_axis", "1.4.0")
has_xclim, requires_xclim = _importorskip("xclim", "0.31")
has_bias_correction, requires_bias_correction = _importorskip("bias_correction")
has_xesmf, requires_xesmf = _importorskip("xesmf")
has_xrft, requires_xrft = _importorskip("xrft")
has_eofs, requires_eofs = _importorskip("eofs")
