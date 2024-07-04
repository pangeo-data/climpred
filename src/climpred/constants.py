# for general checks of climpred-required dimensions
CLIMPRED_ENSEMBLE_DIMS = ["init", "member", "lead"]
# corresponding CF-complying standard_names from http://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html to rename from  # noqa: E501
CF_STANDARD_NAMES = {
    "init": "forecast_reference_time",
    "member": "realization",
    "lead": "forecast_period",
}
CF_LONG_NAMES = {
    "init": "Initialization",
    "member": "Member",
    "lead": "Lead",
}
CLIMPRED_DIMS = CLIMPRED_ENSEMBLE_DIMS + ["time"]

# List of frequencies to check to infer time series stride
FREQ_LIST_TO_INFER_STRIDE = ["day", "month", "year"]

# calendar type for perfect-model (PM) (needed for bootstrapping_uninit)
# Leap also works, but changing Leap,NoLeap fails
PM_CALENDAR_STR = "DatetimeNoLeap"
# standard calendar for hindcast experiments
HINDCAST_CALENDAR_STR = "DatetimeProlepticGregorian"

# default kwargs when using concat
# data_vars='minimal' could be added but needs to check that xr.Dataset
CONCAT_KWARGS = {"coords": "minimal", "compat": "override"}

# name for additional dimension in m2m comparison
M2M_MEMBER_DIM = "forecast_member"

# Valid keywords for aligning inits and verification dates.
VALID_ALIGNMENTS = ["same_inits", "same_init", "same_verifs", "same_verif", "maximize"]

# Valid units for lead dimension
VALID_LEAD_UNITS = [
    "years",
    "seasons",
    "months",
    "weeks",
    "pentads",
    "days",
    "minutes",
    "hours",
    "seconds",
]

# Valid keywords for reference forecast
VALID_REFERENCES = ["uninitialized", "persistence", "climatology"]

# https://github.com/pankajkarman/bias_correction/blob/master/bias_correction.py
BIAS_CORRECTION_BIAS_CORRECTION_METHODS = [
    "modified_quantile",
    "gamma_mapping",
    "basic_quantile",
    "normal_mapping",
]
XCLIM_BIAS_CORRECTION_METHODS = [
    "DetrendedQuantileMapping",
    "LOCI",
    "EmpiricalQuantileMapping",
    # 'ExtremeValues',
    # 'NpdfTransform',
    "PrincipalComponents",
    "QuantileDeltaMapping",
    "Scaling",
]
INTERNAL_BIAS_CORRECTION_METHODS = [
    "additive_mean",
    "multiplicative_mean",
    "multiplicative_std",
]
BIAS_CORRECTION_METHODS = (
    BIAS_CORRECTION_BIAS_CORRECTION_METHODS
    + INTERNAL_BIAS_CORRECTION_METHODS
    + XCLIM_BIAS_CORRECTION_METHODS
)
BIAS_CORRECTION_TRAIN_TEST_SPLIT_METHODS = ["unfair", "unfair-cv", "fair"]
CROSS_VALIDATE_METHODS = ["LOO", False, True]

# seasonality: climpred.set_options(seasonality='...')
GROUPBY_SEASONALITIES = ["dayofyear", "weekofyear", "month", "season"]
