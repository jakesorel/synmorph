import numpy as np
from numba import jit,f4


"""
Periodic functions
------------------

Misc. functions to account for periodicity of the domain, all jitted to aid efficiency. 

In the rest of the code base, this is imported as "per"
"""

@jit(f4[:](f4[:],f4,f4))
def mod1(X, Lx, Ly):
    return np.array((np.mod(X[0], Lx), np.mod(X[1], Ly))).astype(np.float32)


@jit(f4[:,:](f4[:,:],f4,f4))
def mod2(X, Lx, Ly):
    return np.column_stack((np.mod(X[:, 0], Lx), np.mod(X[:, 1], Ly))).astype(np.float32)


@jit(f4[:,:,:](f4[:,:,:],f4,f4))
def mod3(X, Lx, Ly):
    return np.dstack((np.mod(X[:, :, 0], Lx), np.mod(X[:, :, 1], Ly))).astype(np.float32)


@jit(f4[:](f4[:],f4,f4))
def per1(X, Lx, Ly):
    x = np.mod(X[0] + Lx / 2, Lx) - Lx / 2
    y = np.mod(X[1] + Ly / 2, Ly) - Ly / 2
    return np.array((x, y)).astype(np.float32)


@jit(f4[:,:](f4[:,:],f4,f4))
def per(X, Lx, Ly):
    x = np.mod(X[:, 0] + Lx / 2, Lx) - Lx / 2
    y = np.mod(X[:, 1] + Ly / 2, Ly) - Ly / 2
    return np.column_stack((x, y)).astype(np.float32)


@jit(f4[:,:,:](f4[:,:,:],f4,f4))
def per3(X, Lx, Ly):
    x = np.mod(X[:, :, 0] + Lx / 2, Lx) - Lx / 2
    y = np.mod(X[:, :, 1] + Ly / 2, Ly) - Ly / 2
    return np.dstack((x, y)).astype(np.float32)
