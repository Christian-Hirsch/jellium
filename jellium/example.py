import sys
import numpy as np
from jellium.jellium_strip import jellium_strip
from jellium.tikz_drawer import draw_bridges

n = int(sys.argv[1])
fname = sys.argv[2]
size = np.array([20, 2])

trace = jellium_strip(size, n)
draw_bridges(trace[-1], size, fname)
