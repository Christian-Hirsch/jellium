import numpy as np
from jellium.mcmc import mcmc
from jellium.utils import green_strip, torify
from jellium.bridge import bridge

def jellium_strip(size=np.array([20, 2]), 
                  n_part=10, 
                  beta=2, 
                  steps_mcmc=int(1e4), 
                  steps_bridge=200, 
                  state=None):
    """Wigner's jellium on strip-shaped torus
    # Arguments
        size: strip size
        beta: inverse temperature
        n_part: number of particles
        steps_mcmc: number of steps of the MCMC simulator
        steps_bridge: number of time steps for the bridges
        state = state for random number generator
    # Result
        trace of MCMC simulating the jellium
    """
    if state == None: 
        state = np.random.get_state()
    np.random.set_state(state)
    
    pair_pot = lambda x:  beta/(steps_bridge + 1) * green_strip(x, size=size)
    var=1


    #init val
    init_start_pts = np.expand_dims(((np.random.rand(n_part, 2) - .5) * size), -1)
    init_start_pts = np.repeat(init_start_pts, steps_bridge + 1, -1)
    init_bridges = np.array([[bridge(var, steps_bridge) for _ in range(2)] for _ in range(n_part)])
    init_val =  init_start_pts + init_bridges

    #proposal
    propose =  lambda x: torify(np.repeat(np.expand_dims(x[:,:,0] + np.random.randn(*(x[:,:,0].shape)), -1), 
                                      steps_bridge + 1,
                                      -1)
                                + np.array([[bridge(var, steps_bridge) for _ in range(2)] 
                                            for _ in range(x.shape[0])]), size)


    return mcmc(pair_pot, steps_mcmc, init_val, propose, state)