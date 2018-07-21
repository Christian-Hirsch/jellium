import pytest
import numpy as np
from jellium.energy import energy
from numpy.testing import assert_array_less 
from jellium.mcmc import mcmc
from jellium.bridge import bridge
from jellium.utils import d_tor, torify

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
    energy_trace = [energy(conf, pair_pot) for conf in trace]
    assert np.mean(energy_trace[:window]) - np.mean(energy_trace[-window:]) >  np.std(energy_trace[-window:]) * decrease_factor

def test_mcmc_dyn():
    seed = 42
    np.random.seed(seed)
    state = np.random.get_state()

    EPS = 1e-2
    size = int(2e1)
    beta = 2
    steps = 20
    pair_pot = lambda x: - beta/(steps + 1) * d_tor(x, size=size)
    n_iter = int(3e3)

    var = 1

    states = [state for state,_ in [[np.random.get_state(), np.random.rand()] for _ in range(int(size/2))]]
    init_start_pts = np.repeat((np.random.rand(int(size/2))-.5)[np.newaxis] * size, steps + 1, 0).transpose()
    init_bridges = np.array([bridge(var, steps, state) for state in states])
    init_val =  init_start_pts + init_bridges

    torify = lambda x: x - size * np.floor( (x/size+.5))
    propose =  lambda x: torify(np.repeat((x[:,0]+np.random.randn(x.shape[0]))[np.newaxis], steps + 1, 0).transpose()
                            +np.array([bridge(var, steps) for _ in range(x.shape[0])]))

    #energy decrease factor
    decrease_factor = 2

    #size of rolling window for moving averages
    window = 50


    trace = mcmc(pair_pot, n_iter, init_val, propose, state)

    #correct shape
    assert trace.shape == (n_iter,) + init_val.shape

    #values should change
    assert_array_less(EPS, trace.var(axis = 0))

    #energy must decrease
    energy_trace = [energy(conf, pair_pot) for conf in trace]
    assert np.mean(energy_trace[:window]) - np.mean(energy_trace[-window:]) >  np.std(energy_trace[-window:]) * decrease_factor
    
def test_mcmc_dyn_dim2():
    seed = 42
    np.random.seed(seed)
    state = np.random.get_state()

    EPS = 1e-2
    size = np.array([int(2e1), 2])
    n_part = int(size[0]/2)
    beta = 2
    steps = 20
    pair_pot = lambda x: - beta/(steps + 1) * d_tor(x, size=size)
    n_iter = int(2e3)

    var = 1

    states = [state for state,_ in [[np.random.get_state(), np.random.rand()] for _ in range(n_part)]]
    init_start_pts = np.expand_dims(((np.random.rand(n_part, 2) - .5) * size), -1)
    init_start_pts = np.repeat(init_start_pts, steps + 1, -1)
    init_bridges = np.array([[bridge(var, steps) for _ in range(2)] for _ in range(n_part)])
    init_val =  init_start_pts + init_bridges

    propose =  lambda x: torify(np.repeat(np.expand_dims(x[:,:,0] + np.random.randn(*(x[:,:,0].shape)), -1), steps + 1, -1)
                                + np.array([[bridge(var, steps) for _ in range(2)] for _ in range(x.shape[0])]), size)

    #energy decrease factor
    decrease_factor = 2

    #size of rolling window for moving averages
    window = 50


    trace = mcmc(pair_pot, n_iter, init_val, propose, state)

    #correct shape
    assert trace.shape == (n_iter,) + init_val.shape

    #values should change
    assert_array_less(EPS, trace.var(axis = 0))

    #energy must decrease
    energy_trace = [energy(conf, pair_pot) for conf in trace]
    assert np.mean(energy_trace[:window]) - np.mean(energy_trace[-window:]) >  np.std(energy_trace[-window:]) * decrease_factor
