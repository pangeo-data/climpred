"""
Objects dealing with anything visualization.

Color
-----
- `discrete_cmap` : Create a discrete colorbar for the visualization.
"""
import numpy as np
import matplotlib.pyplot as plt

def discrete_cmap(levels, base_cmap):
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
    discrete_cmap : LinearSegmentedColormap
           Discretized colormap for plotting

    Examples
    --------
    import numpy as np
    import matplotlib.pyplot as plt
    import esmtools as et
    data = np.random.randn(50,50)
    plt.pcolor(data, vmin=-3, vmax=3, cmap=et.visualization.discrete_cmap(10,
               "RdBu"))
    plt.colorbar()
    plt.show()
    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, levels))
    cmap_name = base.name + str(levels)
    discrete_cmap = base.from_list(cmap_name, color_list, levels)
    return discrete_cmap
