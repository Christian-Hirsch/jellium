import pytest
import numpy as np
from jellium.utils import d_tor

def test_d_tor():
    x = -6
    size = 10
    
    
    dist = d_tor(x, size)
    
    assert 4 == pytest.approx(dist)
    
    #test vectorization
    seed = 42
    size = int(1e2)

    np.random.seed(seed)

    conf = np.random.rand(size) * size - size/2
    
    assert len(d_tor(conf, size)) == size
    
    size_small = 1
    conf2 = np.random.rand(size) * size_small - size_small/2
    
    assert len(d_tor(np.array([conf, conf2]), np.array([size, size_small]))) == size

