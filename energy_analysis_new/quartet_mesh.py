import numpy as np

from synmorph import tri_functions as trf
# from synmorph.force import get_tF,get_J,get_dvdr
from scipy import sparse
# import triangle as tr
from numba import jit

class QuartetGeom:
    def __init__(self,quartet_x,quartet_tri,quartet_neigh):
        self.quartet_x = quartet_x
        self.quartet_tri = quartet_tri
        self.quartet_neigh = quartet_neigh
        self.boundary_neigh = quartet_neigh == - 1
        self.tx = trf.tri_call3(quartet_x, quartet_tri)
        self.vs = circumcenter(self.tx)
        self.vn = trf.tri_call3(self.vs, quartet_neigh)  ##note the dodgy filling of -1s here and below.
        self.vn = trf.replace_val(self.vn,np.dstack((self.boundary_neigh,self.boundary_neigh)),np.nan)
        self.vp1 = trf.roll3(self.vn)
        self.vm1 = trf.roll3(self.vn, -1)
        self.v_x = disp23(self.vs, self.tx)
        self.v_vp1 = disp23(self.vs, self.vp1)
        self.lp1 = trf.tnorm(self.v_vp1)
        self.v_vm1 = disp23(self.vs, self.vm1)
        self.lm1 = trf.tnorm(self.v_vm1)
        self.vm1_x = disp33(self.vm1, self.tx)
        self.vp1_vm1 = disp33(self.vp1, self.vm1)
        self.P = trf.assemble_tri(self.lp1, quartet_tri)
        self.vp1_x = disp33(self.vp1, self.tx)
        self.A = trf.assemble_tri(0.5 * trf.tcross(self.vm1_x, self.v_x), quartet_tri)
        self.l_int = sparse.coo_matrix((self.lp1.ravel(), (quartet_tri.ravel(), trf.roll(quartet_tri, -1).ravel())))



@jit(nopython=True)
def circumcenter(C):
    ri, rj, rk = C.transpose(1, 2, 0)
    ax, ay = ri
    bx, by = rj
    cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    vs = np.empty((ax.size, 2), dtype=np.float64)
    vs[:, 0], vs[:, 1] = ux, uy
    return vs



@jit(nopython=True)
def disp33(x, y):
    return x - y


@jit(nopython=True)
def disp23(x, y):
    return np.expand_dims(x, 1) - y
