import pytest
import numpy as np
from jellium.energy import energy
from jellium.utils import d_tor

def test_energy():
    seed = 42
    size = int(1e2)
    EPS = 1e-1

    np.random.seed(seed)

    conf = np.random.rand(size) * size - size/2

    assert energy(conf, lambda x: -d_tor(x, size)) != 0
    assert energy(conf, lambda x: -d_tor(x, size)) == pytest.approx(-size**3/4, rel=EPS)
