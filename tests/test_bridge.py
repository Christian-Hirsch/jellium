import pytest
import numpy as np
from numpy.testing import assert_allclose
from jellium.bridge import bridge

def test_answer():
    var = 1
    seed = 42
    steps = 10
    EPS = 1e-2


    sample_bridge = bridge(var, steps, seed)

    #test endpoints
    assert len(sample_bridge)==steps+1
    assert 0 == pytest.approx(sample_bridge[0])
    assert 0 == pytest.approx(sample_bridge[steps])

    #test mean-variance
    n_samp = int(1e5)
    bridges = [bridge(var, steps, seed) for seed in range(n_samp)]
    bridges = np.stack(bridges)

    times = np.linspace(0, 1, num=steps+1, endpoint=True) 
    means = np.mean(bridges, axis=0)
    vars=np.var(bridges, axis=0) - times * (1 - times)

    assert_allclose(means, 0, atol=EPS)
    assert_allclose(vars, 0, atol=EPS)


