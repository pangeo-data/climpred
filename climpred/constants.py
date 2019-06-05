from .metrics import (
    _pearson_r, _pearson_r_p_value, _rmse, _mse, _mae, _msss_murphy,
    _conditional_bias, _bias, _std_ratio, _bias_slope, _crps, _crpss,
    _less, _nmae, _nrmse, _nmse, _ppp, _uacc)
from .comparisons import _e2r, _m2r, _m2c, _e2c, _m2m, _m2e

INIT = 'init'
TIME = 'time'
LEAD = 'lead'

ALL_HINDCAST_METRICS_DICT = {
    'pearson_r': _pearson_r,
    'pr': _pearson_r,
    'acc': _pearson_r,
    'pearson_r_p_value': _pearson_r_p_value,
    'rmse': _rmse,
    'mse': _mse,
    'mae': _mae,
    'msss_murphy': _msss_murphy,
    'conditional_bias': _conditional_bias,
    'c_b': _conditional_bias,
    'unconditional_bias': _bias,
    'u_b': _bias,
    'bias': _bias,
    'std_ratio': _std_ratio,
    'bias_slope': _bias_slope,
    'crps': _crps,
    'crpss': _crpss,
    'less': _less,
    'nmae': _nmae,
    'nrmse': _nrmse,
    'nmse': _nmse,
    'nev': _nmse,
    'ppp': _ppp,
    'msss': _ppp,
    'uacc': _uacc,
}

ALL_PM_METRICS_DICT = ALL_HINDCAST_METRICS_DICT.copy()
del ALL_PM_METRICS_DICT['less']

# more positive skill is better than more negative
POSITIVELY_ORIENTED_METRICS = [
    'pearson_r',
    'msss_murphy',
    'ppp',
    'msss',
    'crpss',
    'uacc',
    'msss',
]

ALL_HINDCAST_COMPARISONS_DICT = {'e2r': _e2r, 'm2r': _m2r}

ALL_PM_COMPARISONS_DICT = {'m2c': _m2c, 'e2c': _e2c, 'm2m': _m2m, 'm2e': _m2e}

ALL_COMPARISONS_DICT = {**ALL_HINDCAST_COMPARISONS_DICT, **ALL_PM_COMPARISONS_DICT}
