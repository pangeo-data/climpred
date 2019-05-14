"""Contains functions for loading sample datasets for decadal prediction.

This is nearly a replication of the `loadutils` in `esmtools` but it is
important for `climpred` users to have explicitly within-package. The
main function is `open_dataset` which by default points to the folder
containing example datasets relevant to decadal prediction.

Aliases have been made for the files to make it more clear what the user is
loading. See the README.md under climpred/sample_data/prediction on Github
for more details on the files themselves.

Perfect-Model:
* "MPI-PM-DP-1D": decadal prediction ensemble area averages of SST/SSS/AMO.
* "MPI-PM-DP-3D": decadal prediction ensemble lat/lon/time of SST/SSS/AMO.
* "MPI-control-1D": area averages for the control run of SST/SSS.
* "MPI-control-3D": lat/lon/time for the control run of SST/SSS.

Reference-based:
* "CESM-DP-SST": decadal prediction ensemble of global mean SSTs.
* "CESM-DP-SSS": decadal prediction ensemble of global mean SSS.
* "CESM-LE": uninitialized ensemble of global mean SSTs.
* "ERSST": observations of global mean SSTs.
* "FOSI-SST": reconstruction of global mean SSTs.
* "FOSI-SSS": reconstruction of global mean SSS.
"""

import hashlib
import os as _os
from urllib.request import urlretrieve as _urlretrieve

from xarray.backends.api import open_dataset as _open_dataset

_default_cache_dir = _os.sep.join(('~', '.climpred_data'))

file_alias_dict = {'MPI-control-1D': 'PM_MPI-ESM-LR_control',
                   'MPI-control-3D': 'PM_MPI-ESM-LR_control3d',
                   'MPI-PM-DP-1D': 'PM_MPI-ESM-LR_ds',
                   'MPI-PM-DP-3D': 'PM_MPI-ESM-LR_ds3d',
                   'CESM-DP-SST': 'CESM-DP-LE.SST.global',
                   'CESM-DP-SSS': 'CESM-DP-LE.SSS.global',
                   'CESM-LE': 'CESM-LE.global_mean.SST.1955-2015',
                   'MPIESM_miklip_baseline1-hind-SST-global':
                        'MPIESM_miklip_baseline1-hind-SST-global',
                   'MPIESM_miklip_baseline1-hist-SST-global':
                        'MPIESM_miklip_baseline1-hist-SST-global',
                   'MPIESM_miklip_baseline1-assim-SST-global':
                        'MPIESM_miklip_baseline1-assim-SST-global',
                   'ERSST': 'ERSSTv4.global.mean',
                   'FOSI-SST': 'FOSI.SST.global',
                   'FOSI-SSS': 'FOSI.SSS.global'}

file_descriptions = {'MPI-PM-DP-1D': 'decadal prediction ensemble area' +
                                     ' averages of SST/SSS/AMO.',
                     'MPI-PM-DP-3D': 'decadal prediction ensemble' +
                                     ' lat/lon/time of SST/SSS/AMO.',
                     'MPI-control-1D': 'area averages for the control run of' +
                                       ' SST/SSS.',
                     'MPI-control-3D': 'lat/lon/time for the control run of' +
                                       ' SST/SSS.',
                     'CESM-DP-SST': 'decadal prediction ensemble of global' +
                                    ' mean SSTs.',
                     'CESM-DP-SSS': 'decadal prediction ensemble of global' +
                                    ' mean SSS.',
                     'CESM-LE': 'uninitialized ensemble of global mean SSTs.',
                     'MPIESM_miklip_baseline1-hind-SST-global':
                        'initialized ensemble of global mean SSTs',
                     'MPIESM_miklip_baseline1-hist-SST-global':
                        'uninitialized ensemble of global mean SSTs',
                     'MPIESM_miklip_baseline1-assim-SST-global':
                        'assimilation in MPI-ESM of global mean SSTs',
                     'ERSST': 'observations of global mean SSTs.',
                     'FOSI-SST': 'reconstruction of global mean SSTs.',
                     'FOSI-SSS': 'reconstruction of global mean SSS.',
                     }


def get_datasets():
    """Prints out available datasets for the user to load."""
    for key in file_descriptions.keys():
        print(f"'{key}': {file_descriptions[key]}")


def _file_md5_checksum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


def open_dataset(name, cache=True, cache_dir=_default_cache_dir,
                 github_url='https://github.com/bradyrx/climpred',
                 branch='master', extension='sample_data/prediction', **kws):
    """Load example data or a mask from an online repository.

    This is a function from `xarray.tutorial` to load an online dataset
    with minimal package imports. I am copying it here because it looks like
    it will soon be deprecated. Also, I've added the ability to point to
    data files that are not in the main folder of the repo (i.e., they are
    in subfolders).

    Note that this requires an md5 file to be loaded. Check the github
    repo bradyrx/climdata for a python script that converts .nc files into
    md5 files.

    Args:
        name: (str) Name of the netcdf file containing the dataset, without
              the .nc extension.
        cache_dir: (str, optional) The directory in which to search
                   for and cache the data.
        cache: (bool, optional) If true, cache data locally for use on later
               calls.
        github_url: (str, optional) Github repository where the data is stored.
        branch: (str, optional) The git branch to download from.
        extension: (str, optional) Subfolder within the repository where the
                   data is stored.
        kws: (dict, optional) Keywords passed to xarray.open_dataset

    Returns:
        The desired xarray dataset.
    """
    if name.endswith('.nc'):
        name = name[:-3]
    # use aliases
    if name in file_alias_dict.keys():
        name = file_alias_dict[name]
    longdir = _os.path.expanduser(cache_dir)
    fullname = name + '.nc'
    localfile = _os.sep.join((longdir, fullname))
    md5name = name + '.md5'
    md5file = _os.sep.join((longdir, md5name))

    if not _os.path.exists(localfile):
        # This will always leave this directory on disk.
        # May want to add an option to remove it.
        if not _os.path.isdir(longdir):
            _os.mkdir(longdir)

        if extension is not None:
            url = '/'.join((github_url, 'raw', branch, extension, fullname))
            _urlretrieve(url, localfile)
            url = '/'.join((github_url, 'raw', branch, extension, md5name))
            _urlretrieve(url, md5file)
        else:
            url = '/'.join((github_url, 'raw', branch, fullname))
            _urlretrieve(url, localfile)
            url = '/'.join((github_url, 'raw', branch, md5name))
            _urlretrieve(url, md5file)

        localmd5 = _file_md5_checksum(localfile)
        with open(md5file, 'r') as f:
            remotemd5 = f.read()
        if localmd5 != remotemd5:
            _os.remove(localfile)
            msg = """
            MD5 checksum does not match, try downloading dataset again.
            """
            raise IOError(msg)

    ds = _open_dataset(localfile, **kws)

    if not cache:
        ds = ds.load()
        _os.remove(localfile)

    return ds
