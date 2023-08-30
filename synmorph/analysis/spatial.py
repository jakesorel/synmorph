import numpy as np
from numba import jit
import synmorph.periodic_functions as per


@jit(nopython=True)
def unwrap_positions(_x_save,L):
    x_save = _x_save.astype(np.float32)
    dx = per.per3(x_save[1:] - x_save[:-1],L,L)
    x_save_unwrapped = np.zeros_like(x_save)
    x = x_save[0].copy()
    x_save_unwrapped[0] = x
    for i in range(len(dx)):
        x += dx[i]
        x_save_unwrapped[i+1] = x.copy()
    return x_save_unwrapped

@jit(nopython=True)
def displacements(_x_save,L):
    x_save = _x_save.astype(np.float32)

    dx = per.per3(x_save[1:] - x_save[:-1],L,L)
    displacements_unwrapped = np.zeros_like(x_save)
    x = np.zeros_like(x_save[0])
    displacements_unwrapped[0] = x
    for i in range(len(dx)):
        x += dx[i]
        displacements_unwrapped[i+1] = x.copy()
    return displacements_unwrapped

@jit(nopython=True)
def mean_squared_displacement(_x_save,L):
    x_save = _x_save.astype(np.float32)

    displacements_unwrapped = displacements(x_save,L)
    squared_displacement = displacements_unwrapped[...,0]**2 + displacements_unwrapped[...,1]**2
    msd = squared_displacement@np.ones(squared_displacement.shape[1],dtype=np.float32)/squared_displacement.shape[1]
    return msd

@jit(nopython=True)
def diffusion_coefficient(x_save,t_span_save,L):
    msd = mean_squared_displacement(x_save,L)
    D = (msd*t_span_save).sum()/(t_span_save**2).sum()/4
    return D


@jit(nopython=True)
def diffusion_coefficient_by_ctype(x_save,t_span_save,c_types,L):
    n_ctypes = np.max(c_types)+1
    D = np.zeros(n_ctypes)
    for c_type in range(n_ctypes):
        D[c_type] = diffusion_coefficient(x_save[:, c_types == c_type], t_span_save, L)
    return D
