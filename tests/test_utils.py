import pytest
import numpy as np
from jellium.utils import d_tor, unwrap_edge, unwrap_path
from numpy.testing import assert_array_equal, assert_array_less


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
    
def test_unwrap_edge():
    edge = np.array([[ 9.83,  0.9 ], [ 9.81, -0.98]])
    size = np.array([20, 2])
    expected = np.array([[9.83, 0.9 ], [9.81, 1.02]]), np.array([[ 9.83, -1.1 ], [ 9.81, -0.98]])
    

    result = unwrap_edge(edge, size)

    assert_array_equal(result[0], expected[0])
    assert_array_equal(result[1], expected[1])
    
    
    edge2= np.array([[-4.57,  0.67], [-4.66, -0.86]])
    expected2 = np.array([[-4.57,  0.67], [-4.66, 1.14]]), np.array([[-4.57,  -0.33], [-4.66, -0.86]])
    
    
def test_unwrap_path():
    path = np.array([[ 4.73,  5.01,  4.96,  4.99,  4.93,  4.86,  4.71,  4.67,  4.61,
         4.55,  4.55,  4.57,  4.85,  5.13,  5.14,  4.92,  4.74,  4.54,
         4.53,  4.72,  4.73],
       [-0.54, -0.44, -0.2 , -0.34, -0.51, -0.71, -0.82, -0.81, -0.84,
        -0.75, -0.74, -0.41, -0.45, -0.38, -0.21, -0.12, -0.14, -0.7 ,
        -0.8 , -0.41, -0.54]])
    size = np.array([20,  2])
    
    edges = unwrap_path(path, size)
    
    #no wrapping around
    assert_array_less(np.max(np.abs(np.diff(edges, axis = 1))[:,0,:], 0), size/2)


