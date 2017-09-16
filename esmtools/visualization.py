"""
Objects dealing with anything visualization.

Color
-----
- `discrete_cmap` : Create a discrete colorbar for the visualization.
"""
import numpy as np
import matplotlib.pyplot as plt

def discrete_cmap(levels, base_cmap=None):
    """
    Returns a discretized colormap based on the specified input colormap.

    Parameters
    ----------
    levels : int
           Number of divisions for the color bar.
    base_cmap : string
           Colormap to discretize (can pull from cmocean, matplotlib, etc.)

    Returns
    ------
    discrete_cmap : 
    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    discrete_cmap = base.from_list(cmap_name, color_list, N)
    return discrete_cmap
