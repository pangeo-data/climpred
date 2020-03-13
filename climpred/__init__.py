from pkg_resources import DistributionNotFound, get_distribution

from .classes import HindcastEnsemble, PerfectModelEnsemble
from .preprocessing import mpi, shared
from .versioning.print_versions import show_versions

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
