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
    return np.sum(pair_int[mask])

def particle_energy(particle, conf, pot):
    """Computes the contribution of a particle inside a configuration to the total energy 
    # Arguments
        particle: tagged particle
        conf: configuration
        pot: pair potential
    # Result
        energy contribution of the particle
    """
    return np.sum([pot(x - particle) for x in conf])
    
