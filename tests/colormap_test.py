"""
Test any new color palettes that have been added.

INPUT 1: cmap str
"""
import sys
import numpy as np
import esmtools as et
import matplotlib.pyplot as plt
name = sys.argv[1]
cmap = et.colormaps.load_cmap(name)
x = np.random.randn(50,50)
plt.pcolor(x, cmap=cmap)
plt.colorbar()
plt.set_title(name)
plt.show()
