import os

import xarray as xr
from climpred.loadutils import file_alias_dict as datasets


def test_open_dataset_locally():
    """Opens all files listed in file_alias_dict."""
    extension = 'sample_data/prediction'
    pwd = os.getcwd()
    for dataset_nc in datasets.values():
        f = pwd + '/' + extension + '/' + dataset_nc + '.nc'
        d = xr.open_mfdataset(f)
        assert d.nbytes > 1
