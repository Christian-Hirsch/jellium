import pytest
import numpy as np
from jellium.utils import d_tor

def test_d_tor():
    x = -6
    size = 10
    
    dist = d_tor(x, size)
    
    #test vectorization
    seed = 42
    size = int(1e2)

    np.random.seed(seed)

    conf = np.random.rand(size) * size - size/2
    
    assert len(d_tor(conf, size)) == size

