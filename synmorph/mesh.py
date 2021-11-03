import _pickle as cPickle
import bz2
import pickle

import numpy as np
import triangle as tr
from numba import jit
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

    def __init__(self, x=None, L=None, fill=True, id=None, name=None, load=None, run_options=None):
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
        self.grid_xy = np.array([self.grid_x.ravel(), self.grid_y.ravel()]).T

        if load is not None:
            self.load(load)
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
        y = trf.make_y(self.x, self.L * self.grid_xy)

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
        n_tri = trf.remove_repeats(new_tri, n_c)

        # tri_same = (self.tri == n_tri).all()

        # 6. Store outputs
        self.n_v = n_tri.shape[0]
        self.tri = n_tri
        self.neigh = trf.get_neighbours(n_tri)

    def triangulate(self):
        if type(self.k2s) is list:
            self._triangulate()
            self.k2s = get_k2(self.tri, self.neigh)
        else:
            self.Angles = trf.tri_angles_periodic(self.x, self.tri, self.L)
            if (self.k2s >= 3).sum() != 0:
                self._triangulate()
                self.k2s = get_k2(self.tri, self.neigh)
            else:
                mask = ((self.Angles[self.neigh, self.k2s] + self.Angles) < np.pi)
                if not mask.all():
                    if self.run_options["equiangulate"]:
                        self.equiangulate(self.x, mask)
                    else:
                        self._triangulate()
                        self.k2s = get_k2(self.tri, self.neigh)
        self.tri_format()

    def equiangulate(self, x, mask):
        """

        Fill this in properly later ...

        Consider the sum of the angles opposite every interface. If this is >180, then equiangulate.

        mask defines the cells/angles for which the sum with a neighbouring cell/angle is >180. These come in pairs

        Equiangulation works by looping through the following, until there exist no such pairs:
            1. Pick an edge for which the angles > 180. This is defined by "chosen_cell" and "chosen_opposite_cell", which are actually triangles.
            2. Replace the triangle entries for each of these triangles, such that the edge is swapped from the four cells
            3. Recompute the neighbours, but only for these two triangles, and their surrounding (4) neighbours (=6)
            4. Recalculate the angles and the mask and repeat.

        Notes:
            -- One worry is that equiangulation fails. May be important in the future to include a fail-safe back up of recomputation.
            -- Would be good to jit this function

        :param x:
        :param mask:
        :return:
        """

        timeout = self.run_options["equi_nkill"]
        k = 0
        while (not mask.all()) and (k < timeout):

            changed_tris, j = np.nonzero(~mask)
            chosen_cell = changed_tris[0]
            cell_mask = np.zeros(3, dtype=np.bool)
            cell_mask[j[0]] = True
            chosen_opposite_cell = self.neigh[chosen_cell, cell_mask][0]

            cells = np.roll(self.tri[chosen_cell], -j[0])
            opposite_cells = self.tri[chosen_opposite_cell]
            opposite_cells = np.roll(opposite_cells, - self.k2s[chosen_cell, cell_mask])

            self.tri[chosen_cell] = cells[0], opposite_cells[0], cells[2]
            self.tri[chosen_opposite_cell] = opposite_cells[0], cells[0], opposite_cells[2]

            self.Angles[[chosen_cell, chosen_opposite_cell]] = trf.tri_angles_periodic(x, self.tri[
                [chosen_cell, chosen_opposite_cell]], self.L)
            # self.Angles = tri_angles_periodic(x,self.tri,self.L)
            self.tx = x[self.tri]
            self.vs = self.get_vertices()

            modify_neighbours = np.concatenate([self.neigh[chosen_cell], self.neigh[chosen_opposite_cell]])
            modify_neighbours.sort()
            self.neigh[modify_neighbours] = -1

            n_neigh = trf.get_neighbours(self.tri, self.neigh, Range=modify_neighbours)
            self.neigh = n_neigh
            self.vn = trf.tri_call3(self.vs, self.neigh)

            self.k2s = get_k2(self.tri, self.neigh)
            if (self.k2s >= 3).sum() != 0:
                self._triangulate()
                self.k2s = get_k2(self.tri, self.neigh)
                mask[:] = True
            else:
                mask = ((self.Angles[self.neigh, self.k2s] + self.Angles) < np.pi)
            k += 1
        if k == timeout:
            self._triangulate()
            self.k2s = get_k2(self.tri, self.neigh)

    def tri_format(self):
        self.tx = trf.tri_call3(self.x, self.tri)
        self.vs = self.get_vertices()
        self.vn = trf.tri_call3(self.vs, self.neigh)
        self.vp1 = trf.roll3(self.vn)
        self.vm1 = trf.roll3(self.vn, -1)

    def get_displacements(self):
        self.v_x = disp23(self.vs, self.tx, self.L)
        self.lv_x = trf.tnorm(self.v_x)
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
        self.l_int = coo_matrix((self.lp1.ravel(), (self.tri.ravel(), trf.roll(self.tri, -1).ravel())))


@jit(nopython=True)
def disp33(x, y, L):
    return per.per3(x - y, L, L)


@jit(nopython=True)
def disp23(x, y, L):
    return per.per3(np.expand_dims(x, 1) - y, L, L)


@jit(nopython=True)
def disp32(x, y, L):
    return per.per3(x - np.expand_dims(y, 1), L, L)


@jit(nopython=True, cache=True)
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
# def get_cv_mat(tri):
#     tri_flat = tri.ravel()
#     ntri = tri.shape[0]
#     ntri_flat = tri_flat.size
#     cv_mat = coo_matrix((np.ones(ntri_flat,dtype=np.int64),(tri_flat,np.repeat(np.arange(ntri),3))))
#
# @jit(nopython=True)
# def get_tris_by_cell(tri):
#     n_c = tri.max() + 1
#     tri_i = [None]*n_c
#     for i in range(n_c):
#         mask = tri==i
#         mask = mask[:,0] + mask[:,1] + mask[:,2]
#         tri_i[i] = np.nonzero(mask)[0]
#     return tri_i
#

###get vertices by cell

##sort vertices around cell centre

##These are the regions that will then be plotted.
