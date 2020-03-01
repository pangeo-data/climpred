# name for additional dimension in m2m comparison
M2M_MEMBER_DIM = 'forecast_member'

# for general checks of climpred-required dimensions
CLIMPRED_ENSEMBLE_DIMS = ['init', 'member', 'lead']
CLIMPRED_DIMS = CLIMPRED_ENSEMBLE_DIMS + ['time']

# Valid units for lead dimension
VALID_LEAD_UNITS = ['years', 'seasons', 'months', 'weeks', 'pentads', 'days']

# List of frequencies to check a dimension has different coords
FREQ_LIST = ['day', 'month', 'year']

# calendar type for PM (needed for bootstrapping_uninit)
# Leap also works, but changing Leap,NoLeap fails
PM_CALENDAR_STR = 'DatetimeNoLeap'
