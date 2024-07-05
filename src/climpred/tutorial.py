"""Implement climpred.tutorial.load_dataset to load analysis ready datasets."""

import hashlib
import os as _os
import urllib
from typing import Dict, Optional
from urllib.request import urlretrieve as _urlretrieve

import xarray as xr
from xarray.backends.api import open_dataset as _open_dataset

_default_cache_dir: str = _os.sep.join(("~", ".climpred_data"))

aliases = [
    "MPI-control-1D",
    "MPI-control-3D",
    "MPI-PM-DP-1D",
    "MPI-PM-DP-3D",
    "CESM-DP-SST",
    "CESM-DP-SSS",
    "CESM-DP-SST-3D",
    "CESM-LE",
    "MPIESM_miklip_baseline1-hind-SST-global",
    "MPIESM_miklip_baseline1-hist-SST-global",
    "MPIESM_miklip_baseline1-assim-SST-global",
    "ERSST",
    "FOSI-SST",
    "FOSI-SSS",
    "FOSI-SST-3D",
    "GMAO-GEOS-RMM1",
    "RMM-INTERANN-OBS",
    "ECMWF_S2S_Germany",
    "Observations_Germany",
    "NMME_hindcast_Nino34_sst",
    "NMME_OIv2_Nino34_sst",
]
true_file_names = [
    "PM_MPI-ESM-LR_control",
    "PM_MPI-ESM-LR_control3d",
    "PM_MPI-ESM-LR_ds",
    "PM_MPI-ESM-LR_ds3d",
    "CESM-DP-LE.SST.global",
    "CESM-DP-LE.SSS.global",
    "CESM-DP-LE.SST.eastern_pacific",
    "CESM-LE.global_mean.SST.1955-2015",
    "MPIESM_miklip_baseline1-hind-SST-global",
    "MPIESM_miklip_baseline1-hist-SST-global",
    "MPIESM_miklip_baseline1-assim-SST-global",
    "ERSSTv4.global.mean",
    "FOSI.SST.global",
    "FOSI.SSS.global",
    "FOSI.SST.eastern_pacific",
    "GMAO-GEOS-V2p1.RMM1",
    "RMM1.observed.interannual.1974-06.2017-07",
    "ECMWF_S2S_Germany",
    "Observations_Germany",
    "NMME_hindcast_Nino34_sst",
    "NMME_Reyn_SmithOIv2_Nino34_sst",
]
file_descriptions = [
    "area averages for the MPI control run of SST/SSS.",
    "lat/lon/time for the MPI control run of SST/SSS.",
    "perfect model decadal prediction ensemble area averages of SST/SSS/AMO.",
    "perfect model decadal prediction ensemble lat/lon/time of SST/SSS/AMO.",
    "hindcast decadal prediction ensemble of global mean SSTs.",
    "hindcast decadal prediction ensemble of global mean SSS.",
    "hindcast decadal prediction ensemble of eastern Pacific SSTs.",
    "uninitialized ensemble of global mean SSTs.",
    "hindcast initialized ensemble of global mean SSTs",
    "uninitialized ensemble of global mean SSTs",
    "assimilation in MPI-ESM of global mean SSTs",
    "observations of global mean SSTs.",
    "reconstruction of global mean SSTs.",
    "reconstruction of global mean SSS.",
    "reconstruction of eastern Pacific SSTs",
    "daily RMM1 from the GMAO-GEOS-V2p1 model for SubX",
    "observed RMM with interannual variablity included",
    "S2S ECMWF on-the-fly hindcasts from the S2S Project for Germany",
    "CPC/ERA5 observations for S2S forecasts over Germany",
    "monthly multi-member hindcasts of sea-surface temperature averaged over the Nino3.4 region from the NMME project from IRIDL",  # noqa: E501
    "monthly Reyn_SmithOIv2 sea-surface temperature observations averaged over the Nino3.4 region",  # noqa: E501
]

FILE_ALIAS_DICT = dict(zip(aliases, true_file_names))
FILE_DESCRIPTIONS = dict(zip(aliases, file_descriptions))


def _file_md5_checksum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


def _get_datasets():
    """Print out available datasets for the user to load if no args are given."""
    for key in FILE_DESCRIPTIONS.keys():
        print(f"'{key}': {FILE_DESCRIPTIONS[key]}")


def _cache_all():
    """Cache all datasets for pytest -n 4 woth pytest-xdist."""
    for d in aliases:
        load_dataset(d)


def _initialize_proxy(proxy_dict):
    """Open a proxy for firewalled servers so that the downloads can go through.

    Args:
        proxy_dict (dictionary): Keys are either 'http' or 'https' and
            values are the proxy server.

    Ref: https://stackoverflow.com/questions/22967084/
         urllib-request-urlretrieve-with-proxy
    """
    proxy = urllib.request.ProxyHandler(proxy_dict)
    opener = urllib.request.build_opener(proxy)
    urllib.request.install_opener(opener)


def load_dataset(
    name: Optional[str] = None,
    cache: bool = True,
    cache_dir: str = _default_cache_dir,
    github_url: str = "https://github.com/pangeo-data/climpred-data",
    branch: str = "master",
    extension: Optional[str] = None,
    proxy_dict: Optional[Dict[str, str]] = None,
    **kws,
) -> xr.Dataset:
    """Load example data or a mask from an online repository.

    Args:
        name: Name of the netcdf file containing the
              dataset, without the ``.nc`` extension. If ``None``, this function
              prints out the available datasets to import.
        cache: If ``True``, cache data locally for use on later calls.
        cache_dir: The directory in which to search for and cache the data.
        github_url: Github repository where the data is stored.
        branch: The git branch to download from.
        extension: Subfolder within the repository where the data is stored.
        proxy_dict: Dictionary with keys as either "http" or "https" and values as the
            proxy server. This is useful if you are on a work computer behind a
            firewall and need to use a proxy out to download data.
        kws: Keywords passed to :py:meth:`~xarray.open_dataset`.

    Returns:
        The desired :py:class:`xarray.Dataset`

    Examples:
        >>> from climpred.tutorial import load_dataset
        >>> proxy_dict = {"http": "127.0.0.1"}
        >>> ds = load_dataset("FOSI-SST", cache=False, proxy_dict=proxy_dict)
    """
    if name is None:
        return _get_datasets()

    if proxy_dict is not None:
        _initialize_proxy(proxy_dict)

    # https://stackoverflow.com/questions/541390/extracting-extension-from-
    # filename-in-python
    # Allows for generalized file extensions.
    name, ext = _os.path.splitext(name)
    if not ext.endswith(".nc"):
        ext += ".nc"

    # use aliases
    if name in FILE_ALIAS_DICT.keys():
        name = FILE_ALIAS_DICT[name]
    longdir = _os.path.expanduser(cache_dir)
    fullname = name + ext
    localfile = _os.sep.join((longdir, fullname))
    md5name = name + ".md5"
    md5file = _os.sep.join((longdir, md5name))

    if not _os.path.exists(localfile):
        # This will always leave this directory on disk.
        # May want to add an option to remove it.
        if not _os.path.isdir(longdir):
            _os.mkdir(longdir)

        if extension is not None:
            url = "/".join((github_url, "raw", branch, extension, fullname))
            _urlretrieve(url, localfile)
            url = "/".join((github_url, "raw", branch, extension, md5name))
            _urlretrieve(url, md5file)
        else:
            url = "/".join((github_url, "raw", branch, fullname))
            _urlretrieve(url, localfile)
            url = "/".join((github_url, "raw", branch, md5name))
            _urlretrieve(url, md5file)

        localmd5 = _file_md5_checksum(localfile)
        with open(md5file, "r") as f:
            remotemd5 = f.read()
        if localmd5 != remotemd5:
            _os.remove(localfile)
            msg = """
            Try downloading the file again. There was a confliction between
            your local .md5 file compared to the one in the remote repository,
            so the local copy has been removed to resolve the issue.
            """
            raise IOError(msg)

    ds = _open_dataset(localfile, **kws)

    if not cache:
        ds = ds.load()
        _os.remove(localfile)
    return ds
