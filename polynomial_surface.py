import itertools
import numpy as np

########################################################################################################################

def polyfit2d(x, y, z, order=3):
    # the number of parameters to
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z, rcond=None)
    return m

def polyval2d(xy, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(xy[:,0])

    x = xy[:,0]
    y = xy[:,1]
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
