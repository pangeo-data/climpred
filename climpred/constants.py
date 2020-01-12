from .comparisons import __ALL_COMPARISONS__ as all_comparisons
from .metrics import __ALL_METRICS__ as all_metrics

# To match a metric for (multiple) keywords.
METRIC_ALIASES = dict()
for m in all_metrics:
    if m.aliases is not None:
        for a in m.aliases:
            METRIC_ALIASES[a] = m.name

DETERMINISTIC_METRICS = [m.name for m in all_metrics if not m.probabilistic]
DETERMINISTIC_HINDCAST_METRICS = DETERMINISTIC_METRICS
# Metrics to be used in compute_perfect_model.
DETERMINISTIC_PM_METRICS = DETERMINISTIC_HINDCAST_METRICS.copy()
# Used to set attrs['units'] to None.
DIMENSIONLESS_METRICS = [m.name for m in all_metrics if m.unit_power == 1]
# More positive skill is better than more negative.
POSITIVELY_ORIENTED_METRICS = [m.name for m in all_metrics if m.positive]
PROBABILISTIC_METRICS = [m.name for m in all_metrics if m.probabilistic]

# Combined allowed metrics for compute_hindcast and compute_perfect_model
HINDCAST_METRICS = DETERMINISTIC_HINDCAST_METRICS + PROBABILISTIC_METRICS
PM_METRICS = DETERMINISTIC_PM_METRICS + PROBABILISTIC_METRICS
ALL_METRICS = (
    DETERMINISTIC_HINDCAST_METRICS + DETERMINISTIC_PM_METRICS + PROBABILISTIC_METRICS
)

# Which comparisons work with which set of metrics.
# ['e2r', 'm2r']
HINDCAST_COMPARISONS = [c.name for c in all_comparisons if c.hindcast]
# ['m2c', 'e2c', 'm2m', 'm2e']
PM_COMPARISONS = [c.name for c in all_comparisons if not c.hindcast]
ALL_COMPARISONS = HINDCAST_COMPARISONS + PM_COMPARISONS
# ['m2c', 'm2m']
PROBABILISTIC_PM_COMPARISONS = [
    c.name for c in all_comparisons if (not c.hindcast and c.probabilistic)
]
# ['m2r']
PROBABILISTIC_HINDCAST_COMPARISONS = [
    c.name for c in all_comparisons if (c.hindcast and c.probabilistic)
]
PROBABILISTIC_COMPARISONS = (
    PROBABILISTIC_HINDCAST_COMPARISONS + PROBABILISTIC_PM_COMPARISONS
)

# for general checks of climpred-required dimensions
CLIMPRED_ENSEMBLE_DIMS = ['init', 'member', 'lead']
CLIMPRED_DIMS = CLIMPRED_ENSEMBLE_DIMS + ['time']

# Valid units for lead dimension
VALID_LEAD_UNITS = ['years', 'seasons', 'months', 'weeks', 'pentads', 'days']
