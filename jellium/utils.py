import numpy as np

def d_tor(x, size):
    """Distances on a 1-dimensional torus
    # Arguments
        x: point whose distance from origin is computed
        size: size of the torus
    # Result
        distance of x to the origin
    """
    return np.min(np.abs([x, x - size, x + size]),0)