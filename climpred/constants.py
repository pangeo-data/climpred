# for general checks of climpred-required dimensions
CLIMPRED_ENSEMBLE_DIMS = ['init', 'member', 'lead']
CLIMPRED_DIMS = CLIMPRED_ENSEMBLE_DIMS + ['time']

# default kwargs when using concat
# data_vars='minimal' could be added but needs to check that not xr.Dataset
CONCAT_KWARGS = {'coords': 'minimal', 'compat': 'override'}

# name for additional dimension in m2m comparison
M2M_MEMBER_DIM = 'forecast_member'

# Valid keywords for aligning inits and verification dates.
VALID_ALIGNMENTS = ['same_inits', 'same_init', 'same_verifs', 'same_verif', 'maximize']

# Valid units for lead dimension
VALID_LEAD_UNITS = ['years', 'seasons', 'months', 'weeks', 'pentads', 'days']
