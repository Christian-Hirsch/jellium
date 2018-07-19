import numpy as np

def bridge(var, steps, seed=None):
    """1D Brownian bridge in the time interval [0,1]
    # Arguments
        var: variance of the Brownian bridge
        steps: number of time steps to simulate
        seed: seed for the randomness
    # Result
        trace of the bridge
    """
    np.random.seed(seed)

    incs = np.random.randn(steps)
    incs = np.insert(incs, 0, 0)
    incs = incs * np.sqrt(var / steps )

    brownian = np.cumsum(incs)
    return brownian - brownian[-1] * np.linspace(0, 1, num=steps+1, endpoint=True) 
