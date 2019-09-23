# metrics to be used in compute_hindcast
DETERMINISTIC_HINDCAST_METRICS = [
    'pearson_r',
    'pearson_r_p_value',
    'rmse',
    'mse',
    'mae',
    'msss_murphy',
    'conditional_bias',
    'bias',
    'std_ratio',
    'bias_slope',
    'nmae',
    'nrmse',
    'nmse',
    'ppp',
    'uacc',
]

# metrics to be used in compute_perfect_model
DETERMINISTIC_PM_METRICS = DETERMINISTIC_HINDCAST_METRICS.copy()

# to match a metric for multiple keywords
METRIC_ALIASES = {
    'pr': 'pearson_r',
    'acc': 'pearson_r',
    'pval': 'pearson_r_p_value',
    'pvalue': 'pearson_r_p_value',
    'c_b': 'conditional_bias',
    'unconditional_bias': 'bias',
    'u_b': 'bias',
    'nev': 'nmse',
    'msss': 'ppp',
    'brier': 'brier_score',
    'bs': 'brier_score',
    'tbs': 'threshold_brier_score',
}

# more positive skill is better than more negative
# needed to decide which skill is better in bootstrapping confidence levels
POSITIVELY_ORIENTED_METRICS = [
    'pearson_r',
    'msss_murphy',
    'ppp',
    'msss',
    'crpss',
    'uacc',
    'msss',
]

# needed to set attrs['units'] to None
DIMENSIONLESS_METRICS = [
    'pearson_r',
    'pearson_r_p_value',
    'crpss',
    'msss_murphy',
    'std_ratio',
    'bias_slope',
    'conditional_bias',
    'ppp',
    'nrmse',
    'nmse',
    'nmae',
    'uacc',
    'threshold_brier_score',
]

# to decide different logic in compute functions
PROBABILISTIC_METRICS = [
    'crpss_es',
    'threshold_brier_score',
    'crps',
    'crpss',
    'brier_score',
]

# combined allowed metrics for compute_hindcast and compute_perfect_model
HINDCAST_METRICS = DETERMINISTIC_HINDCAST_METRICS + PROBABILISTIC_METRICS
PM_METRICS = DETERMINISTIC_PM_METRICS + PROBABILISTIC_METRICS

# which comparisons work with which set of metrics
HINDCAST_COMPARISONS = ['e2r', 'm2r']
PM_COMPARISONS = ['m2c', 'e2c', 'm2m', 'm2e']

PROBABILISTIC_PM_COMPARISONS = ['m2c', 'm2m']
PROBABILISTIC_HINDCAST_COMPARISONS = ['m2r']

# for general checks of climpred-required dimensions

CLIMPRED_ENSEMBLE_DIMS = ['init', 'member', 'lead']
CLIMPRED_DIMS = CLIMPRED_ENSEMBLE_DIMS + ['time']
