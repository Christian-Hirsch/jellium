import pytest
import numpy as np
from jellium.energy import energy
from numpy.testing import assert_array_less 
from jellium.mcmc import mcmc
from jellium.utils import d_tor

def test_mcmc():
    seed = 42
    np.random.seed(seed)

    EPS = 1e-2
    size = int(2e1)
    beta = 2
    pair_pot = lambda x: - beta * d_tor(x, size=size)
    n_iter = int(1e3)

    init_val = np.random.rand(size) * size - size/2
    torify = lambda x: x - size * np.floor( (x/size+.5))
    propose =  lambda x: torify(x + np.random.randn(size))

    #energy decrease factor
    decrease_factor = 2

    #size of rolling window for moving averages
    window = 50


    trace = mcmc(pair_pot, n_iter, init_val, propose, np.random.get_state())

    #correct shape
    assert trace.shape == (n_iter,) + init_val.shape
    
    #values should change
    assert_array_less(EPS, trace.var(axis = 0))

    #energy must decrease
    energy_trace = [energy(conf, lambda x: - d_tor(x, size=size)) for conf in trace]
    assert np.mean(energy_trace[:window]) - np.mean(energy_trace[-window:]) >  np.std(energy_trace[-window:]) * decrease_factor


