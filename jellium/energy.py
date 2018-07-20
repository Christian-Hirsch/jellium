import numpy as np

def energy(conf, pot):
    """Computes the energy of a configuration associated with a pair potential
    # Arguments
        conf: configuration
        pot: pair potential
    # Result
        energy of the configuration under the pair potential
    """
    pair_int = np.array([[pot(x - y) for x in conf] for y in conf])
    mask = np.identity(len(conf)) == 0
    
    #energy of trajectories
    if len(conf.shape)==2:
        mask = np.moveaxis(np.repeat(mask[np.newaxis], conf.shape[1],0), 0 , -1)
        
    return np.sum(pair_int[mask])
