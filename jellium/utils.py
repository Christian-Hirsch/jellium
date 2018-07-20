import numpy as np

def d_tor(x, size):
    """Distances on a torus
    # Arguments
        x: point whose distance from origin is computed
        size: size of the torus
    # Result
        distance of x to the origin
    """
    if type(size)==int: return np.min(np.abs([x, x - size, x + size]),0)
    
    size_rep = np.repeat(size[np.newaxis], x.shape[1], 0).transpose()
    dist = np.min(np.abs([x, x - size_rep, x + size_rep]),0)
    return np.hypot(*dist)