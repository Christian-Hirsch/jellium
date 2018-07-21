import numpy as np

def d_tor(x, size):
    """Distances on a torus
    # Arguments
        x: point whose distance from origin is computed
        size: torus size
    # Result
        distance of x to the origin
    """
    if type(size)==int: return np.min(np.abs([x, x - size, x + size]),0)
    
    size_rep = np.repeat(size[np.newaxis], x.shape[1], 0).transpose()
    dist = np.min(np.abs([x, x - size_rep, x + size_rep]), 0)
    return np.hypot(*dist)

def torify(x, size):
    """shift points to canonical torus domain
    # Arguments
        x: points to be shifted
        size: torus size
    # Result
        points shifted to canonical torus domain
    """
    rep_size = np.repeat(np.expand_dims(size,-1), x.shape[-1], -1)
    return x - rep_size * np.floor( (x/rep_size+.5))

def green_strip(x, size, EPS=1e-9):
    """Approximation for Green's function on strip-shaped torus
    # Arguments
        x: point on the strip
        size: strip size
    # Result
        Approximation for Green's function on a strip-shaped domain
    """
    dist = d_tor(x, size)
    return (dist > 1) * (-dist) + (dist < 1) * (-np.log(dist + EPS))   

def unwrap_edge(edge, size):
    """Unwraps a single edge on the torus
    # Arguments
        path: path on the torus
        size: array describing the size of the torus
    # Result
        two edges that not wrapping around the torus
    """
    shifts = np.matmul(np.diag(size),np.array([[0, 1, -1], [0, 1, -1]]))
    rep_diff = np.repeat((edge[0]-edge[1])[np.newaxis], 3, 0).transpose() 
    rep_siz = np.repeat(size[np.newaxis], 3, 0).transpose()
    shift = shifts[np.abs(rep_diff +  shifts) < 0.5 * rep_siz]

    p0_new, p1_new = edge[0] + shift.transpose(), edge[1] - shift.transpose()
    return [edge[0], p1_new], [p0_new, edge[1]]

def unwrap_path(path, size):
    """Unwraps a path on the torus
    # Arguments
        path: path on the torus
        size: array describing the size of the torus
    # Result
        list of sub-paths such that non of the sub-paths wraps around the torus
    """
    return  [edge 
             for i in range(path.shape[1])
             for edge in  unwrap_edge(np.array([path[:,i], path[:,((i+1) % path.shape[1])]]), size)              
            ]