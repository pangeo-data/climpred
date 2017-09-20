"""
Simply a bunch of NCL colormaps. Most of this script is constant colormap
arrays and then one simple function to load them.

- `load_cmap` : Function to load in a non-matplotlib/cmocean colormap.

"""
import numpy as np
import esmtools.vis as vis
import pandas as pd
import matplotlib.colors as color

def load_cmap(name):
    """
    Returns a colormap to use in plotting.

    Parameters
    ----------
    name : str
        Identifier for colormap.
    """
    df = pd.DataFrame(eval(name))
    cmap = color.ListedColormap(df.values)
    return cmap

# COLORMAPS

# Rainbow
# ------

# 1. AMWG
amwg = np.array([[ 0.57647059,  0.43921569,  0.85882353],
       [ 0.        ,  0.        ,  0.78431373],
       [ 0.23529412,  0.39215686,  0.90196078],
       [ 0.47058824,  0.60784314,  0.94901961],
       [ 0.69019608,  0.87843137,  0.90196078],
       [ 0.1254902 ,  0.69803922,  0.66666667],
       [ 0.60392157,  0.80392157,  0.19607843],
       [ 0.18039216,  0.54509804,  0.34117647],
       [ 0.96078431,  0.90196078,  0.74509804],
       [ 0.87058824,  0.72156863,  0.52941176],
       [ 1.        ,  0.88235294,  0.        ],
       [ 1.        ,  0.64705882,  0.        ],
       [ 1.        ,  0.27058824,  0.        ],
       [ 0.69803922,  0.13333333,  0.13333333],
       [ 1.        ,  0.71372549,  0.75686275],
       [ 1.        ,  0.07843137,  0.57647059]])
