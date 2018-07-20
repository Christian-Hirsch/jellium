import numpy as np
from jellium.energy import energy

def mcmc(pair_pot, n_iter, init_val, propose, state=None):
    """MCMC for MH
    
    Simulates Markov chain converging to the posterior
    # Arguments
        pair_pot: pair potential to be used
        n_iter: number of iterations
        init_val: initial value of Markov chain
        propose: proposal function for a single parameter
        state: state of random number generator
    # Result
        energy of the configuration under the pair potential
    """
    if state==None: state = np.random.get_state()
    np.random.set_state(state)

    n_pts = init_val.shape[0]
    trace = np.zeros((n_iter ,)+ init_val.shape)
    cur_conf = init_val

    for iter in range(n_iter):
        new_conf = propose(cur_conf)
        new_energy, cur_energy = [energy(conf, pair_pot) for conf in [new_conf, cur_conf]]

        if(np.log(np.random.rand(1)) < cur_energy - new_energy):
            cur_conf = np.copy(new_conf)
        trace[iter] = cur_conf  
    return trace
    
