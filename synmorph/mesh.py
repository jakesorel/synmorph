import _pickle as cPickle
import bz2
import pickle

import numpy as np
import triangle as tr
from numba import jit, i4, f4, boolean
from scipy.sparse import coo_matrix

import synmorph.periodic_functions as per
import synmorph.tri_functions as trf


class Mesh:
    """
    Mesh class
    ----------

    Deals with triangulation of a set of points, and calculates relevant geometries for use in force calculations.

    Triangulation algorithm takes some options, which can be tweaked for efficiency. Notably equiangulation.
    """

    def __init__(self, x=None, L=None, tri=None, fill=True, id=None, name=None, load=None, run_options=None):
        assert run_options is not None, "Specify run options"

        if id is None:
            self.id = {}
        else:
            self.id = id
        self.run_options = run_options
        self.name = name
        self.x = x
        self.L = L
        self.n_c = []
        self.n_v = []
        self.vs = []
        self.tri = []
        self.neigh = []
        self.k2s = []
        self.tx = []
        self.vs = []
        self.vn = []
        self.vp1 = []
        self.vm1 = []
        self.v_x = []
        self.lv_x = []
        self.v_vp1 = []
        self.lp1 = []
        self.v_vm1 = []
        self.lm1 = []
        self.vp1_x = []
        self.vm1_x = []
        self.vp1_vm1 = []
        self.A = []
        self.P = []
        self.A_components = []
        self.l_int = []

        self.grid_x, self.grid_y = np.mgrid[-1:2, -1:2]
        self.grid_x[0, 0], self.grid_x[1, 1] = self.grid_x[1, 1], self.grid_x[0, 0]
        self.grid_y[0, 0], self.grid_y[1, 1] = self.grid_y[1, 1], self.grid_y[0, 0]
        self.grid_xy = np.array([self.grid_x.ravel(), self.grid_y.ravel()]).T.astype(np.float32)

        if load is not None:
            self.load(load)
        elif tri is not None:
            self.tri = tri
            self.n_v = self.tri.shape[0]
            self.neigh = trf.get_neighbours(self.tri)
            self.update_from_tri()
        elif fill:
            self.n_c = self.x.shape[0]
            self.update_x(self.x)

    def update(self):
        self.triangulate()
        self.tri_format()
        self.get_displacements()
        self.get_A()
        self.get_P()
        self.get_l_interface()

    def update_x(self, x):
        self.x = x
        self.triangulate()
        self.tri_format()
        self.get_displacements()
        self.get_A()
        self.get_P()
        self.get_l_interface()


    def update_from_tri(self):
        self.tri_format()
        self.get_displacements()
        self.get_A()
        self.get_P()
        self.get_l_interface()

    def save(self, name, id=None, dir_path="", compressed=False):
        self.name = name
        if id is None:
            self.id = {}
        else:
            self.id = id
        if compressed:
            with bz2.BZ2File(dir_path + "/" + self.name + "_mesh" + '.pbz2', 'w') as f:
                cPickle.dump(self.__dict__, f)
        else:
            pikd = open(dir_path + "/" + self.name + "_mesh" + '.pickle', 'wb')
            pickle.dump(self.__dict__, pikd)
            pikd.close()

    def load(self, fname):
        if fname.split(".")[1] == "pbz2":
            fdict = cPickle.load(bz2.BZ2File(fname, 'rb'))
        else:
            pikd = open(fname, 'rb')
            fdict = pickle.load(pikd)
            pikd.close()
        if (self.run_options != fdict["run_options"]) and (self.run_options is not None):
            print("Specified run options do not match those from the loaded file. Proceeding...")
        self.__dict__ = fdict

    def get_vertices(self):
        """
        Get vertex locations, given cell centroid positions and triangulation. I.e. calculate the circumcentres of
        each triangle

        :return V: Vertex coordinates (nv x 2)
        """
        V = trf.circumcenter(self.tx, self.L)
        return V
    #
    def _triangulate(self):
        """
        Calculates the periodic triangulation on the set of points x.

        Stores:
            self.n_v = number of vertices (int32)
            self.tri = triangulation of the vertices (nv x 3) matrix.
                Cells are stored in CCW order. As a convention, the first entry has the smallest cell id
                (Which entry comes first is, in and of itself, arbitrary, but is utilised elsewhere)
            self.vs = coordinates of each vertex; (nv x 2) matrix
            self.neigh = vertex ids (i.e. rows of self.vs) corresponding to the 3 neighbours of a given vertex (nv x 3).
                In CCW order, where vertex i {i=0..2} is opposite cell i in the corresponding row of self.tri
            self.neighbours = coordinates of each neighbouring vertex (nv x 3 x 2) matrix

        :param x: (nc x 2) matrix with the coordinates of each cell
        """


        # 1. Tile cell positions 9-fold to perform the periodic triangulation
        #   Calculates y from x. y is (9nc x 2) matrix, where the first (nc x 2) are the "true" cell positions,
        #   and the rest are translations

        if type(self.A) is np.ndarray:
            maxA = np.max(self.A)
            max_d = np.sqrt(maxA/np.pi)*5 ##2.5 cell diameters on average
            if not max_d > self.L/50:
                max_d = self.L
        else:
            max_d = self.L

        y,dictionary = generate_triangulation_mask(self.x.astype(np.float32),self.L,max_d)

        # 2. Perform the triangulation on y
        #   The **triangle** package (tr) returns a dictionary, containing the triangulation.
        #   This triangulation is extracted and saved as tri
        t = tr.triangulate({"vertices": y})

        tri = t["triangles"]

        # Del = Delaunay(y)
        # tri = Del.simplices
        n_c = self.x.shape[0]

        # 3. Find triangles with **at least one** cell within the "true" frame (i.e. with **at least one** "normal cell")
        #   (Ignore entries with -1, a quirk of the **triangle** package, which denotes boundary triangles
        #   Generate a mask -- one_in -- that considers such triangles
        #   Save the new triangulation by applying the mask -- new_tri
        tri = tri[(tri != -1).all(axis=1)]
        one_in = (tri < n_c).any(axis=1)
        new_tri = tri[one_in]

        # 4. Remove repeats in new_tri
        #   new_tri contains repeats of the same cells, i.e. in cases where triangles straddle a boundary
        #   Use remove_repeats function to remove these. Repeats are flagged up as entries with the same trio of
        #   cell ids, which are transformed by the mod function to account for periodicity. See function for more details

        n_tri = dictionary[new_tri]
        n_tri = trf.remove_repeats(n_tri, n_c)

        # tri_same = (self.tri == n_tri).all()

        # 6. Store outputs
        self.n_v = n_tri.shape[0]
        self.tri = n_tri
        self.neigh = trf.get_neighbours(n_tri)
    #
    #
    # def _triangulate(self):
    #     """
    #     Calculates the periodic triangulation on the set of points x.
    #
    #     Stores:
    #         self.n_v = number of vertices (int32)
    #         self.tri = triangulation of the vertices (nv x 3) matrix.
    #             Cells are stored in CCW order. As a convention, the first entry has the smallest cell id
    #             (Which entry comes first is, in and of itself, arbitrary, but is utilised elsewhere)
    #         self.vs = coordinates of each vertex; (nv x 2) matrix
    #         self.neigh = vertex ids (i.e. rows of self.vs) corresponding to the 3 neighbours of a given vertex (nv x 3).
    #             In CCW order, where vertex i {i=0..2} is opposite cell i in the corresponding row of self.tri
    #         self.neighbours = coordinates of each neighbouring vertex (nv x 3 x 2) matrix
    #
    #     :param x: (nc x 2) matrix with the coordinates of each cell
    #     """
    #
    #
    #     # 1. Tile cell positions 9-fold to perform the periodic triangulation
    #     #   Calculates y from x. y is (9nc x 2) matrix, where the first (nc x 2) are the "true" cell positions,
    #     #   and the rest are translations
    #     y = trf.make_y(self.x, self.L * self.grid_xy)
    #
    #     # 2. Perform the triangulation on y
    #     #   The **triangle** package (tr) returns a dictionary, containing the triangulation.
    #     #   This triangulation is extracted and saved as tri
    #     t = tr.triangulate({"vertices": y})
    #
    #     tri = t["triangles"]
    #
    #     # Del = Delaunay(y)
    #     # tri = Del.simplices
    #     n_c = self.x.shape[0]
    #
    #     # 3. Find triangles with **at least one** cell within the "true" frame (i.e. with **at least one** "normal cell")
    #     #   (Ignore entries with -1, a quirk of the **triangle** package, which denotes boundary triangles
    #     #   Generate a mask -- one_in -- that considers such triangles
    #     #   Save the new triangulation by applying the mask -- new_tri
    #     tri = tri[(tri != -1).all(axis=1)]
    #     one_in = (tri < n_c).any(axis=1)
    #     new_tri = tri[one_in]
    #
    #     # 4. Remove repeats in new_tri
    #     #   new_tri contains repeats of the same cells, i.e. in cases where triangles straddle a boundary
    #     #   Use remove_repeats function to remove these. Repeats are flagged up as entries with the same trio of
    #     #   cell ids, which are transformed by the mod function to account for periodicity. See function for more details
    #
    #     n_tri = trf.remove_repeats(new_tri, n_c)
    #
    #     # tri_same = (self.tri == n_tri).all()
    #
    #     # 6. Store outputs
    #     self.n_v = n_tri.shape[0]
    #     self.tri = n_tri
    #     self.neigh = trf.get_neighbours(n_tri)
    #

    def triangulate(self):
        if type(self.k2s) is list or not self.run_options["equiangulate"]:
            self._triangulate()
            self.k2s = get_k2(self.tri, self.neigh)
        else:
            tri, neigh, k2s, failed = re_triangulate(self.x, self.tri, self.neigh, self.k2s, self.tx, self.L, self.n_v,
                                                     self.vs, max_runs=self.run_options["equi_nkill"])
            if failed:
                self._triangulate()
                self.k2s = get_k2(self.tri, self.neigh)
            else:
                self.tri, self.neigh, self.k2s = tri, neigh, k2s

    def tri_format(self):
        self.tx = trf.tri_call3(self.x, self.tri)
        self.vs = self.get_vertices()
        self.vn = trf.tri_call3(self.vs, self.neigh)
        self.vp1 = trf.roll3(self.vn, 1)
        self.vm1 = trf.roll3(self.vn, -1)

    def get_displacements(self):
        self.v_x = disp23(self.vs, self.tx, self.L)
        self.v_vp1 = disp23(self.vs, self.vp1, self.L)
        self.lp1 = trf.tnorm(self.v_vp1)
        self.v_vm1 = disp23(self.vs, self.vm1, self.L)
        self.lm1 = trf.tnorm(self.v_vm1)
        self.vp1_x = disp33(self.vp1, self.tx, self.L)
        self.vm1_x = disp33(self.vm1, self.tx, self.L)
        self.vp1_vm1 = disp33(self.vp1, self.vm1, self.L)

    def get_P(self):
        """
        Identical to **get_P** but accounts for periodic triangulation

        Calculates perimeter of each cell (considering periodic boundary conditions).

        This is just the sum of the ls.

        :param neighbours: (nv x 3 x 2) matrix considering the coordinates of each neighbouring vertex of each vertex
        :param vs: (nv x 2) matrix considering coordinates of each vertex
        :return: self.P saves the areas of each cell
        """
        self.P = trf.assemble_tri(self.lp1, self.tri)
        return self.P

    def get_A(self):
        """
        Calculates the area of the smaller triangles between vertex {i,j}, its counterclockwise neighbour wrt. the cell centre {i,j} and the cell centre itself {i,j}
        :return:
        """
        self.A_components = 0.5 * trf.tcross(self.vm1_x, self.v_x)
        self.A = trf.assemble_tri(self.A_components, self.tri)

    def get_l_interface(self):
        self.l_int = coo_matrix((self.lp1.ravel(), (self.tri.ravel(), trf.roll_int(self.tri, -1).ravel())))


@jit(f4[:, :, :](f4[:, :, :], f4[:, :, :], f4),cache=True)
def disp33(x, y, L):
    return per.per3(x - y, L, L)


@jit(f4[:, :, :](f4[:, :], f4[:, :, :], f4),cache=True)
def disp23(x, y, L):
    return per.per3(np.expand_dims(x, 1) - y, L, L)


@jit(f4[:, :, :](f4[:, :, :], f4[:, :], f4),cache=True)
def disp32(x, y, L):
    return per.per3(x - np.expand_dims(y, 1), L, L)


@jit(i4[:, :](i4[:, :], i4[:, :]),cache=True)
def get_k2(tri, neigh):
    """
    To determine whether a given neighbouring pair of triangles needs to be re-triangulated, one considers the sum of
    the pair angles of the triangles associated with the cell centroids that are **not** themselves associated with the
    adjoining edge. I.e. these are the **opposite** angles.

    Given one cell centroid/angle in a given triangulation, k2 defines the column index of the cell centroid/angle in the **opposite** triangle

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: Neighbourhood matrix (n_v x 3) np.int32 array
    :return:
    """
    three = np.array([0, 1, 2])
    nv = tri.shape[0]
    k2s = np.empty((nv, 3), dtype=np.int32)
    for i in range(nv):
        for k in range(3):
            neighbour = neigh[i, k]
            k2 = ((neigh[neighbour] == i) * three).sum()
            k2s[i, k] = k2
    return k2s


#
# @jit(nopython=True)
# def get_retriangulation_mask(angles,neigh,k2s,ntri):
#     neigh_angles = angles.take(neigh.ravel()*3 + k2s.ravel()).reshape(ntri,3)
#     mask = ((neigh_angles + angles) > np.pi)
#     return mask
#

@jit(boolean[:, :](f4[:, :], i4[:, :], f4[:], i4[:, :], i4[:, :], i4, f4[:, :], f4),cache=True)
def get_retriangulation_mask(x, tri, lv_x, neigh, k2s, ntri, vs, L):
    d_cell = tri.take(neigh * 3 + k2s).reshape(ntri, 3)
    # rad_0 = per.per(tx[:,1] - vs,L,L)
    # rad_0 = np.sqrt(rad_0[:,0]**2 + rad_0[:,1]**2)
    xd = trf.tri_call3(x, d_cell)
    rad_d = per.per3(xd - np.expand_dims(vs, 1), L, L)
    rad_d = np.sqrt(rad_d[..., 0] ** 2 + rad_d[..., 1] ** 2)
    mask = rad_d < np.expand_dims(lv_x, 1)
    return mask


@jit(i4(boolean[:]),cache=True)
def get_first_nonzero(flat_mask):
    i = 0
    while ~flat_mask[i]:
        i += 1
    return i


@jit(i4(boolean[:]),cache=True)
def get_any_nonzero(flat_mask):
    i = int(np.random.random() * flat_mask.size)
    while ~flat_mask[i]:
        i = int(np.random.random() * flat_mask.size)
    return i


@jit((i4[:, :], i4[:, :], i4[:, :], i4, i4),cache=True)
def get_quartet(tri, neigh, k2s, tri_0i, tri_0j):
    a, b, d = np.roll(tri[tri_0i], -tri_0j)
    tri_1i, tri_1j = neigh[tri_0i, tri_0j], k2s[tri_0i, tri_0j]
    c = tri[tri_1i, tri_1j]

    # quartet = np.array((a,b,c,d))

    tri0_da = (tri_0j + 1) % 3
    da_i = neigh[tri_0i, tri0_da]
    da_j = k2s[tri_0i, tri0_da]
    da = tri[da_i, da_j]

    tri0_ab = (tri_0j - 1) % 3
    ab_i = neigh[tri_0i, tri0_ab]
    ab_j = k2s[tri_0i, tri0_ab]
    ab = tri[ab_i, ab_j]

    tri1_cd = (tri_1j - 1) % 3
    cd_i = neigh[tri_1i, tri1_cd]
    cd_j = k2s[tri_1i, tri1_cd]
    cd = tri[cd_i, cd_j]

    tri1_bc = (tri_1j + 1) % 3
    bc_i = neigh[tri_1i, tri1_bc]
    bc_j = k2s[tri_1i, tri1_bc]
    bc = tri[bc_i, bc_j]

    return tri_0i, tri_0j, tri_1i, tri_1j, a, b, c, d, da, ab, bc, cd, da_i, ab_i, bc_i, cd_i, da_j, ab_j, bc_j, cd_j


@jit(nopython=True,cache=True)
def tri_update(val, quartet_info):
    val_new = val.copy()
    tri_0i, tri_0j, tri_1i, tri_1j, a, b, c, d, da, ab, bc, cd, da_i, ab_i, bc_i, cd_i, da_j, ab_j, bc_j, cd_j = quartet_info
    val_new[tri_0i, (tri_0j - 1) % 3] = val[tri_1i, tri_1j]
    val_new[tri_1i, (tri_1j - 1) % 3] = val[tri_0i, tri_0j]
    return val_new


@jit(nopython=True,cache=True)
def update_mesh(quartet_info, tri, neigh, k2s):
    """
    Update tri, neigh and k2. Inspect the equiangulation code for some inspo.
    :return:
    """

    tri_0i, tri_0j, tri_1i, tri_1j, a, b, c, d, da, ab, bc, cd, da_i, ab_i, bc_i, cd_i, da_j, ab_j, bc_j, cd_j = quartet_info

    neigh_new = neigh.copy()
    k2s_new = k2s.copy()

    tri_new = tri_update(tri, quartet_info)

    neigh_new[tri_0i, tri_0j] = neigh[tri_1i, (tri_1j + 1) % 3]
    neigh_new[tri_0i, (tri_0j + 1) % 3] = neigh[bc_i, bc_j]
    neigh_new[tri_0i, (tri_0j + 2) % 3] = neigh[tri_0i, (tri_0j + 2) % 3]
    neigh_new[tri_1i, tri_1j] = neigh[tri_0i, (tri_0j + 1) % 3]
    neigh_new[tri_1i, (tri_1j + 1) % 3] = neigh[da_i, da_j]
    neigh_new[tri_1i, (tri_1j + 2) % 3] = neigh[tri_1i, (tri_1j + 2) % 3]

    k2s_new[tri_0i, tri_0j] = k2s[tri_1i, (tri_1j + 1) % 3]
    k2s_new[tri_0i, (tri_0j + 1) % 3] = k2s[bc_i, bc_j]
    k2s_new[tri_0i, (tri_0j + 2) % 3] = k2s[tri_0i, (tri_0j + 2) % 3]
    k2s_new[tri_1i, tri_1j] = k2s[tri_0i, (tri_0j + 1) % 3]
    k2s_new[tri_1i, (tri_1j + 1) % 3] = k2s[da_i, da_j]
    k2s_new[tri_1i, (tri_1j + 2) % 3] = k2s[tri_1i, (tri_1j + 2) % 3]

    neigh_new[bc_i, bc_j] = tri_0i
    k2s_new[bc_i, bc_j] = tri_0j
    neigh_new[da_i, da_j] = tri_1i
    k2s_new[da_i, da_j] = tri_1j

    return tri_new, neigh_new, k2s_new




@jit((f4[:, :], i4[:, :], i4[:, :], i4[:, :], f4[:, :, :], f4, i4, f4[:, :], i4),cache=True)
def needs_re_triangulating(x, _tri, _neigh, _k2s, tx0, L, ntri, vs0, max_runs=10):
    tri, neigh, k2s = _tri.copy(), _neigh.copy(), _k2s.copy()
    # lv_x = trf.tnorm(disp23(vs0, tx0, L))
    v_x = per.per(vs0 - tx0[:, 0], L, L)
    lv_x = np.sqrt(v_x[..., 0] ** 2 + v_x[..., 1] ** 2)

    mask = get_retriangulation_mask(x, tri, lv_x, neigh, k2s, ntri, vs0, L)
    return mask.any()

@jit((f4[:, :], i4[:, :], i4[:, :], i4[:, :], f4[:, :, :], f4, i4, f4[:, :], i4),cache=True)
def re_triangulate(x, _tri, _neigh, _k2s, tx0, L, ntri, vs0, max_runs=10):
    tri, neigh, k2s = _tri.copy(), _neigh.copy(), _k2s.copy()
    # lv_x = trf.tnorm(disp23(vs0, tx0, L))
    v_x = per.per(vs0 - tx0[:, 0], L, L)
    lv_x = np.sqrt(v_x[..., 0] ** 2 + v_x[..., 1] ** 2)

    mask = get_retriangulation_mask(x, tri, lv_x, neigh, k2s, ntri, vs0, L)
    continue_loop = mask.any()
    failed = False
    n_runs = 0
    if continue_loop:
        tx = tx0.copy()
        vs = vs0.copy()
        while (continue_loop):
            mask_flat = mask.ravel()
            q = get_first_nonzero(mask_flat)
            tri_0i, tri_0j = q // 3, q % 3
            quartet_info = get_quartet(tri, neigh, k2s, tri_0i, tri_0j)
            tri, neigh, k2s = update_mesh(quartet_info, tri, neigh, k2s)
            tx = tri_update(tx, quartet_info)

            tri_0i, tri_1i = quartet_info[0], quartet_info[2]
            tx_changed = np.stack((tx[tri_0i], tx[tri_1i]))
            vs_changed = trf.circumcenter(tx_changed, L)
            vs[tri_0i], vs[tri_1i] = vs_changed
            v_x_changed = per.per(vs_changed - tx_changed[:, 0], L, L)
            lv_x_changed = np.sqrt(v_x_changed[..., 0] ** 2 + v_x_changed[..., 1] ** 2)
            # lv_x_changed = trf.tnorm(disp23(vs_changed, tx_changed, L))
            lv_x[tri_0i], lv_x[tri_1i] = lv_x_changed
            mask = get_retriangulation_mask(x, tri, lv_x, neigh, k2s, ntri, vs, L)
            if n_runs > max_runs:
                failed = True
                continue_loop = False
            if not mask.any():
                continue_loop = False
            n_runs += 1
    return tri, neigh, k2s, failed

@jit(nopython=True)
def generate_triangulation_mask(x,L,max_d):
    ys = np.zeros((0,2),dtype=np.float32)
    dictionary = np.zeros((0),dtype=np.int32)
    for i in [0,-1,1]:
        for j in [0,-1,1]:
            y = (x + np.array((i, j)) * L).astype(np.float32)
            if j == 0:
                if i == 0:
                    mask = np.ones_like(x[:,0],dtype=np.bool_)
                else:
                    val = L*(1-i)/2
                    mask = np.abs(x[:,0]-val)<max_d
            elif i == 0:
                val = L * (1 - j) / 2
                mask = np.abs(x[:, 1] - val) < max_d
            else:
                val_x = L * (1 - i) / 2
                val_y = L * (1 - j) / 2
                mask = np.sqrt((x[:,0]-val_x)**2 + (x[:,1]-val_y)**2) < max_d
            ys = np.row_stack((ys,y[mask]))
            dictionary = np.concatenate((dictionary,np.nonzero(mask)[0].astype(np.int32)))
    return ys,dictionary

# @jit(nopython=True)
# def re_triangulate(x,_tri,_neigh,_k2s,L,ntri):
#     tri,neigh,k2s = _tri.copy(),_neigh.copy(),_k2s.copy()
#     angles = trf.tri_angles_periodic(x, tri, L)
#     neigh_angles = angles.take(neigh.ravel()*3 + k2s.ravel()).reshape(ntri,3)
#     interior_angles = neigh_angles + angles
#     mask = get_retriangulation_mask(angles,neigh,k2s,ntri)
#     n_runs = 0
#     continue_loop = mask.any()
#     failed = False
#     while (continue_loop):
#         mask_flat = mask.ravel()
#         q = get_any_nonzero(mask_flat)
#         # q = np.argmax(interior_angle.ravel())
#         tri_0i, tri_0j = q//3,q%3
#         quartet_info = get_quartet(tri,neigh,k2s,tri_0i,tri_0j)
#         tri_new, neigh_new, k2s_new = update_mesh(quartet_info, tri, neigh, k2s)
#         # trin, neighn, k2sn = update_mesh(quartet_info, tri, neigh, k2s)
#
#         angles_new = trf.tri_angles_periodic(x, tri_new, L)
#         neigh_angles_new = angles_new.take(neigh_new.ravel() * 3 + k2s_new.ravel()).reshape(ntri, 3)
#         interior_angles_new = neigh_angles_new + angles_new
#         mask_new = get_retriangulation_mask(angles_new, neigh_new, k2s_new, ntri)
#         if mask_new.sum()<mask.sum():
#         # if interior_angles_new.ravel()[mask_new.ravel()].sum()< interior_angles.ravel()[mask.ravel()].sum():
#             tri,neigh,k2s = tri_new,neigh_new,k2s_new
#             mask = mask_new
#             interior_angles = interior_angles_new.copy()
#         else:
#             failed = True
#             continue_loop = False
#         if not mask.any():
#             continue_loop = False
#         n_runs += 1
#     return tri,neigh,k2s,failed
#
# t0 = time.time()
# for i in range(int(1e4)):
#     re_triangulate(x, tri, neigh, k2s, L, ntri)
# t1= time.time()
# print(t1-t0)
