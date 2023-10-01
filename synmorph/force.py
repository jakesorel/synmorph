import numpy as np
from numba import jit,f4,i4

import synmorph.periodic_functions as per
import synmorph.tri_functions as trf


class Force:
    """
    Force class
    -----------

    This class is used to calculate the passive mechanics under the SPV model.

    Takes in an instance of the Tissue class, and within it, the Mesh class, and uses information about the geometry to calculate the forces on the cell centroids.

    These forces are accessible in self.F.
    """

    def __init__(self, tissue):
        self.t = tissue
        self.Jp, self.Jm = None, None  ##triangular forms of the J matrix, considering J_ij for CW (Jp) and CCW (Jm) neighbours.
        self.F = None
        self.F_soft = 0  # Soft-repulsion between cell centroids, included for stability following Barton et al.
        self.get_J()
        self.get_F_mechanics()
        if self.t.a == 0:
            self.get_F_soft()
        self.F = sum_F(self.F, self.F_soft)

        self.dA = None
        self.dP = None

    def get_J(self):
        ##for now only with ref. to W, but generalizable.
        self.Jp = get_J(self.t.W, self.t.tc_types, self.t.tc_typesp, np.max(self.t.c_types)+1)
        self.Jm = get_J(self.t.W, self.t.tc_types, self.t.tc_typesm, np.max(self.t.c_types)+1)

    def get_F_mechanics(self):
        """
        Calculates the forces on cell centroids given the del of the energy functional.
        tF is the triangulated form of the forces; i.e. the components of the total force on cell i by each of the involved triangles in the triangulation.

        Energy functional is given by:

        E = Sum_i (kappa_A/2) * (Ai - A0)^2 + (kappa_P/2) * (Pi - P0)^2 + Sum_j J_ij*lij
        """
        tF = get_tF(self.t.mesh.vp1_vm1,
                    self.t.mesh.v_vm1,
                    self.t.mesh.v_vp1,
                    self.t.mesh.v_x,
                    self.t.mesh.lm1,
                    self.t.mesh.lp1,
                    self.Jm,
                    self.Jp,
                    self.t.kappa_A,
                    self.t.kappa_P,
                    self.t.A0,
                    self.t.P0,
                    self.t.mesh.A,
                    self.t.mesh.P,
                    self.t.mesh.tri)
        self.F = trf.assemble_tri3(tF,
                                   self.t.mesh.tri)  ##this assembles the total force on each cell centroid by summing the contributions from each triangle.
        return self.F

    def get_F_soft(self):
        """
        Soft repulsion between cell centroids. Spring-like force under E_soft = Sum_bond k*(|r_i - r_j| - a)^2 where bond = {i,j} for pairs of centroids i,j for which |r_i - r_j| <= a
        :return:
        """
        self.F_soft = trf.assemble_tri3(get_tFsoft(self.t.mesh.tx,
                                                   self.t.a,
                                                   self.t.k,
                                                   self.t.mesh.L),
                                        self.t.mesh.tri)


@jit(f4[:,:](f4[:,:],i4[:,:],i4[:,:],i4))
def get_J(W, tc_types, neigh_tctypes, nc_types):
    return W.take(tc_types.ravel() + nc_types * neigh_tctypes.ravel()).reshape(-1, 3)


@jit(f4[:,:,:,:](f4[:,:,:]))
def get_dvdr(v_x):
    """

    Calculates ∂v_j/dr_i the Jacobian for all cells in each triangulation

    Last two dims: ((dvx/drx,dvx/dry),(dvy/drx,dvy/dry))

    These are lifted from Mathematica

    :param x_v_: (n_v x 3 x 2) np.float32 array of cell centroid positions for each cell in each triangulation (first two dims follow order of triangulation)
    :param vs: (n_v x 2) np.float32 array of vertex positions, corresponding to each triangle in the triangulation
    :param L: Domain size (np.float32)
    :return: Jacobian for each cell of each triangulation (n_v x 3 x 2 x 2) np.float32 array (where the first 2 dims follow the order of the triangulation.
    """
    x_v = -v_x
    dvdr = np.empty(x_v.shape + (2,),dtype=np.float32)
    for i in range(3):
        ax = x_v[:, i%3, 0]
        ay = x_v[:, i%3, 1]
        bx = x_v[:, (i + 1) % 3, 0]
        by = x_v[:, np.mod(i + 1, 3), 1]
        cx = x_v[:, np.mod(i + 2, 3), 0]
        cy = x_v[:, np.mod(i + 2, 3), 1]
        # dhx/drx
        dvdr[:, i, 0, 0] = (ax * (by - cy)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((by - cy) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        # dhy/drx
        dvdr[:, i, 0, 1] = (bx ** 2 + by ** 2 - cx ** 2 + 2 * ax * (-bx + cx) - cy ** 2) / (
                2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((by - cy) * (
                (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        # dhx/dry
        dvdr[:, i, 1, 0] = (-bx ** 2 - by ** 2 + cx ** 2 + 2 * ay * (by - cy) + cy ** 2) / (
                2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy))) - ((-bx + cx) * (
                (ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (-ay + cy) + (ay - by) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

        # dhy/dry
        dvdr[:, i, 1, 1] = (ay * (-bx + cx)) / ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) - ((-bx + cx) * (
                (bx ** 2 + by ** 2) * (ax - cx) + (ax ** 2 + ay ** 2) * (-bx + cx) + (-ax + bx) * (
                cx ** 2 + cy ** 2))) / (2. * ((ay - by) * cx + ax * (by - cy) + bx * (-ay + cy)) ** 2)

    return dvdr



@jit(f4[:,:,:](f4[:,:,:],f4[:,:,:],f4[:,:,:],f4[:,:,:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:],f4[:],f4[:],f4[:],i4[:,:]))
def get_tF(vp1_vm1, v_vm1, v_vp1, v_x, lm1, lp1, Jm, Jp, kappa_A, kappa_P, A0, P0, A, P, tri):
    dAdv_j = np.dstack((vp1_vm1[:, :, 1], -vp1_vm1[:, :, 0])) * 0.5  ##shoelace theorem: i.e. derivative of cross product.

    dPdv_j_m = v_vm1 / np.expand_dims(lm1, 2)
    dPdv_j_p = v_vp1 / np.expand_dims(lp1, 2)

    ##guard against divide by 0 error
    m_mask = np.isnan(dPdv_j_m)
    p_mask = np.isnan(dPdv_j_p)
    if np.any(m_mask):
        dPdv_j_m = trf.replace_val(dPdv_j_m,m_mask,0)
    if np.any(p_mask):
        dPdv_j_p = trf.replace_val(dPdv_j_p,p_mask,0)


    dPdv_j = dPdv_j_p + dPdv_j_m

    dtEdv_l_v_j = dPdv_j_m * np.expand_dims(Jm, 2) + dPdv_j_p * np.expand_dims(Jp, 2)

    dtEdA = trf.tri_call(2 * kappa_A * (A - A0), tri)
    dtEdP = trf.tri_call(2 * kappa_P * (P - P0), tri)

    dtE_dv = np.expand_dims(dtEdA, 2) * dAdv_j + np.expand_dims(dtEdP, 2) * dPdv_j + dtEdv_l_v_j
    dtE_dv = dtE_dv[:, 0] + dtE_dv[:, 1] + dtE_dv[:, 2]  # sum over the three contributions

    dvdr = get_dvdr(v_x)  # order is wrt cell i
    dtE_dv = np.expand_dims(dtE_dv, 2)

    dEdr_x = dtE_dv[:, 0] * dvdr[:, :, 0, 0] + dtE_dv[:, 1] * dvdr[:, :, 0, 1]
    dEdr_y = dtE_dv[:, 0] * dvdr[:, :, 1, 0] + dtE_dv[:, 1] * dvdr[:, :, 1, 1]

    dEdr = np.dstack((dEdr_x, dEdr_y))

    F = - dEdr
    return F.astype(np.float32)

@jit(f4[:,:,:](f4[:,:,:],f4,f4,f4))
def get_tFsoft(tx, a, k, L):
    """
    Additional "soft" pair-wise repulsion at short range to prevent unrealistic and sudden changes in triangulation.

    Repulsion is on the imediate neighbours (i.e. derived from the triangulation)

    And is performed respecting periodic boudnary conditions (system size = L)

    Suppose l_{ij} = \| r_i - r_j \
    F_soft = -k(l_{ij} - 2a)(r_i - r_j) if l_{ij} < 2a; and =0 otherwise

    :param Cents: Cell centroids on the triangulation (n_v x 3 x 2) **np.ndarray** of dtype **np.float64**
    :param a: Cut-off distance of spring-like interaction (**np.float64**)
    :param k: Strength of spring-like interaction (**np.float64**)
    :param CV_matrix: Cell-vertex matrix representation of the triangulation (n_c x n_v x 3)
    :param n_c: Number of cells (**np.int64**)
    :param L: Domain size/length (**np.float64**)
    :return: F_soft
    """
    rj = trf.roll3(tx, 1)
    rij = per.per3(tx - rj, L, L)
    lij = trf.tnorm(rij)
    norm_ij = rij / np.expand_dims(lij, 2)
    tFsoft_ij = (np.expand_dims(-k * (lij - 2 * a) * (lij < 2 * a), 2) * norm_ij).astype(np.float32)
    tFsoft = tFsoft_ij - trf.roll3(tFsoft_ij, -1)
    return tFsoft


@jit(f4[:,:](f4[:,:],f4[:,:]))
def sum_F(F, F_soft):
    return F + F_soft
