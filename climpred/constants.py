from .metrics import __all_metrics__ as all_metrics

# TODO: delete when all metrics as class Metric
METRIC_ALIASES = {
    'pr': 'pearson_r',
    'acc': 'pearson_r',
    'pacc': 'pearson_r',
    'p_pval': 'pearson_r_p_value',
    'pvalue': 'pearson_r_p_value',
    'sacc': 'spearman_r',
    's_pval' 'spearman_r_p_value' 'c_b': 'conditional_bias',
    'unconditional_bias': 'bias',
    'u_b': 'bias',
    'nev': 'nmse',
    'msss': 'ppp',
    'brier': 'brier_score',
    'bs': 'brier_score',
    'tbs': 'threshold_brier_score',
}

# to match a metric for (multiple) keywords
METRIC_ALIASES = dict()
for m in all_metrics:
    if m.aliases is not None:
        for a in m.aliases:
            METRIC_ALIASES[a] = m.name


DETERMINISTIC_METRICS = [m.name for m in all_metrics if not m.is_probabilistic]

# TODO: remove hindcast_PM_metric thing
# metrics to be used in compute_hindcast
DETERMINISTIC_HINDCAST_METRICS = DETERMINISTIC_METRICS
# metrics to be used in compute_perfect_model
DETERMINISTIC_PM_METRICS = DETERMINISTIC_HINDCAST_METRICS.copy()


# more positive skill is better than more negative
# needed to decide which skill is better in bootstrapping confidence levels
POSITIVELY_ORIENTED_METRICS = [m.name for m in all_metrics if m.is_positive]

# needed to set attrs['units'] to None
DIMENSIONLESS_METRICS = [m.name for m in all_metrics if m.unit_power == 1]

# to decide different logic in compute functions
PROBABILISTIC_METRICS = [m.name for m in all_metrics if m.is_probabilistic]

# combined allowed metrics for compute_hindcast and compute_perfect_model
HINDCAST_METRICS = DETERMINISTIC_HINDCAST_METRICS + PROBABILISTIC_METRICS
PM_METRICS = DETERMINISTIC_PM_METRICS + PROBABILISTIC_METRICS

ALL_METRICS = (
    DETERMINISTIC_HINDCAST_METRICS + DETERMINISTIC_PM_METRICS + PROBABILISTIC_METRICS
)


# which comparisons work with which set of metrics
HINDCAST_COMPARISONS = ['e2r', 'm2r']
PM_COMPARISONS = ['m2c', 'e2c', 'm2m', 'm2e']

PROBABILISTIC_PM_COMPARISONS = ['m2c', 'm2m']
PROBABILISTIC_HINDCAST_COMPARISONS = ['m2r']

ALL_COMPARISONS = HINDCAST_COMPARISONS + PM_COMPARISONS


# for general checks of climpred-required dimensions

CLIMPRED_ENSEMBLE_DIMS = ['init', 'member', 'lead']
CLIMPRED_DIMS = CLIMPRED_ENSEMBLE_DIMS + ['time']
