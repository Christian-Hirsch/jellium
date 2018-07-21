import pytest
import numpy as np
from jellium.energy import energy
from numpy.testing import assert_array_less 
from jellium.mcmc import mcmc
from jellium.bridge import bridge
from jellium.utils import d_tor
from jellium.jellium_strip import jellium_strip

def test_jellium_strip():
    size=np.array([20, 2])
    n_part = 10
    beta = 2
    
    steps_mcmc=int(1e4)
    steps_bridge=200
    EPS = 1e-3

    seed = 42
    np.random.seed(seed)
    state = np.random.get_state()
    

    trace =jellium_strip(size, n_part, beta, steps_mcmc, steps_bridge, state)
        
    assert trace.shape == (steps_mcmc, n_part, 2, steps_bridge+1)
    assert EPS < np.std(trace[-1, :, 0, 0])
