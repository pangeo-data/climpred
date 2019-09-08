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

DETERMINISTIC_PM_METRICS = DETERMINISTIC_HINDCAST_METRICS.copy()

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
}

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
    'less',
    'threshold_brier_score',
]

PROBABILISTIC_METRICS = [
    'threshold_brier_score',
    'crps',
    'crpss',
    'brier_score',
    'less',
]

HINDCAST_METRICS = DETERMINISTIC_HINDCAST_METRICS + PROBABILISTIC_METRICS
PM_METRICS = DETERMINISTIC_PM_METRICS + PROBABILISTIC_METRICS

HINDCAST_COMPARISONS = ['e2r', 'm2r']
PM_COMPARISONS = ['m2c', 'e2c', 'm2m', 'm2e']

PROBABILISTIC_PM_COMPARISONS = ['m2c', 'm2m']
PROBABILISTIC_HINDCAST_COMPARISONS = ['m2r']
