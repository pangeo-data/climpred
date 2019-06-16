HINDCAST_METRICS = [
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
    'crps',
    'crpss',
    'less',
    'nmae',
    'nrmse',
    'nmse',
    'ppp',
    'uacc',
]

PM_METRICS = HINDCAST_METRICS.copy()
PM_METRICS.remove('less')

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

HINDCAST_COMPARISONS = ['e2r', 'm2r']
PM_COMPARISONS = ['m2c', 'e2c', 'm2m', 'm2e']
