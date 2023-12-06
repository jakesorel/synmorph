import numpy as np
from numba import f4,i4
from numba import njit as jit
from scipy.sparse import coo_matrix

"""
Triangulation functions
-----------------------

Misc functions that have been optimized for the triangulated data-structure.

This includes rolls etc. 

Also includes functions to convert an array of cell-properties to the triangulated form, and to sum components from each triangle back into a cell-type property 1D array 
"""


@jit(f4[:, :](f4[:, :], i4),cache=True)
def roll(x, direc=1):
    """
    Jitted equivalent to np.roll(x,-direc,axis=1)
    direc = 1 --> counter-clockwise
    direc = -1 --> clockwise
    :param x:
    :return:
    """
    if direc == -1:  # old "roll_forward"
        return_val = np.column_stack((x[:, 2], x[:, :2]))
    elif direc == 1:  # old "roll_reverse"
        return_val = np.column_stack((x[:, 1:3], x[:, 0]))
    return return_val


@jit(i4[:, :](i4[:, :], i4),cache=True)
def roll_int(x, direc):
    """
    Jitted equivalent to np.roll(x,-direc,axis=1)
    direc = 1 --> counter-clockwise
    direc = -1 --> clockwise
    :param x:
    :return:
    """
    if direc == -1:  # old "roll_forward"
        return_val = np.column_stack((x[:, 2], x[:, :2]))
    elif direc == 1:  # old "roll_reverse"
        return_val = np.column_stack((x[:, 1:3], x[:, 0]))
    return return_val


@jit(f4[:, :,:](f4[:, :,:], i4),cache=True)
def roll3(x, direc):
    """
    Like roll, but when x has shape (nv x 3 x 2) ie is a vector, rather than scalar, quantity.
    :param x:
    :param direc:
    :return:
    """
    x_out = np.empty_like(x,dtype=np.float32)
    x_out[:, :, 0], x_out[:, :, 1] = roll(x[:, :, 0], direc=direc), roll(x[:, :, 1], direc=direc)
    return x_out



def order_tris(tri):
    """
    For each triangle (i.e. row in **tri**), order cell ids in ascending order
    :param tri: Triangulation (n_v x 3) np.int32 array
    :return: the ordered triangulation
    """
    nv = tri.shape[0]
    for i in range(nv):
        Min = np.argmin(tri[i])
        tri[i] = tri[i, Min], tri[i, np.mod(Min + 1, 3)], tri[i, np.mod(Min + 2, 3)]
    return tri


def remove_repeats(tri, n_c):
    """
    For a given triangulation (nv x 3), remove repeated entries (i.e. rows)
    The triangulation is first re-ordered, such that the first cell id referenced is the smallest. Achieved via
    the function order_tris. (This preserves the internal order -- i.e. CCW)
    Then remove repeated rows via lexsort.
    NB: order of vertices changes via the conventions of lexsort
    Inspired by...
    https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array
    :param tri: (nv x 3) matrix, the triangulation
    :return: triangulation minus the repeated entries (nv* x 3) (where nv* is the new # vertices).
    """
    tri = order_tris(np.mod(tri, n_c))
    sorted_tri = tri[np.lexsort(tri.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_tri, axis=0), 1))
    return sorted_tri[row_mask]


@jit(f4[:,:](f4[:,:],f4[:,:]),cache=True)
def make_y(x, Lgrid_xy):
    """
    Makes the (9) tiled set of coordinates used to perform the periodic triangulation.
    :param x: Cell centroids (n_c x 2) np.float32 array
    :param Lgrid_xy: (9 x 2) array defining the displacement vectors for each of the 9 images of the tiling
    :return: Tiled set of coordinates (9n_c x 2) np.float32 array
    """
    n_c = x.shape[0]
    y = np.empty((n_c * 9, x.shape[1]),dtype=np.float32)
    for k in range(9):
        y[k * n_c:(k + 1) * n_c] = x + Lgrid_xy[k]
    return y


@jit(f4[:, :](f4[:, :, :]),cache=True)
def tnorm(x):
    """
    Calculate the L1 norm of a set of vectors that are given in triangulated form:

    (nv x 3 x 2) ->> (nv x 3)
    :param x:
    :return:
    """
    return np.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)


@jit(f4[:, :](f4[:], i4[:, :]),cache=True)
def tri_call(val, tri):
    """
    when val has shape (n,3)
    Equiv. to:
    >> val[tri]
    :param val:
    :param tri:
    :return:
    """
    return val.take(tri.ravel()).reshape(-1, 3)

@jit(i4[:, :](i4[:], i4[:, :]),cache=True)
def tri_call_int(val, tri):
    """
    when val has shape (n,3)
    Equiv. to:
    >> val[tri]
    :param val:
    :param tri:
    :return:
    """
    return val.take(tri.ravel()).reshape(-1, 3)


@jit(f4[:, :,:](f4[:,:], i4[:, :]),cache=True)
def tri_call3(val, tri):
    """
    When val has shape (n,3,2)
    Equiv to:
    >> val[tri]
    :param val:
    :param tri:
    :return:
    """
    vali, valj = val[:, 0], val[:, 1]
    return np.dstack((tri_call(vali, tri), tri_call(valj, tri)))


@jit(f4[:, :](f4[:, :], i4[:, :], i4),cache=True)
def tri_mat_call(mat, tri, direc=-1):
    """
    If matrix element {i,j} corresponds to the edge value connecting cells i and j,
    then this function returns the edge value connecting a vertex to its counter-clockwise neighbour
    Or equivalently the case where j is CW to i in a given triangle.
    Swap CCW for CW if direc = 1
    :param mat:
    :param tri:
    :param direc:
    :return:
    """
    # return np.dstack((mat[i, j] for (i, j) in zip(tri, roll(tri, direc))))

    nv = tri.shape[0]
    tmat = np.empty((nv, 3),dtype=np.float32)
    tri_roll = roll_int(tri, direc)
    for k in range(nv):
        for m in range(3):
            tri_i, tri_k = tri[k, m], tri_roll[k, m]
            tmat[k, m] = mat[tri_i, tri_k]
    return tmat

def CV_matrix(tri_list, n_v, n_c):
    """
    Generate cell-vertex boolean matrix. This is currently not in use.
    :param tri_list:
    :param n_v:
    :param n_c:
    :return:
    """
    CV_matrix = np.zeros((n_c, n_v, 3))
    for i in range(3):
        CV_matrix[tri_list[:, i], np.arange(n_v), i] = 1
    return CV_matrix

#
# @jit(nopython=True)
# def tri_sum(n_c, CV_matrix, tval):
#     val_sum = np.zeros(n_c)
#     for i in range(3):
#         val_sum += np.asfortranarray(CV_matrix[:, :, i]) @ np.asfortranarray(tval[:, i])
#     return val_sum

#
# @jit(nopython=True)
# def cosine_rule(a, b, c):
#     return np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))


# @jit(nopython=True)
# def clip(x, xmin, xmax):
#     xflat = x.ravel()
#     minmask = xflat < xmin
#     maxmask = xflat > xmax
#     xflat[minmask] = xmin
#     xflat[maxmask] = xmax
#     return xflat.reshape(x.shape)
#

@jit(cache=True)
def replace_val(x, mask, xnew):
    xflat = x.ravel()
    maskflat = mask.ravel()
    xflat[maskflat] = xnew
    return xflat.reshape(x.shape)


@jit(cache=True)
def replace_vec(x, mask, xnew):
    xflat = x.ravel()
    maskflat = mask.ravel()
    xflat[maskflat] = xnew.ravel()[maskflat]
    return xflat.reshape(x.shape)


@jit(f4[:,:](f4[:,:,:],f4[:,:,:]),cache=True)
def tcross(A, B):
    """
    Cross product of two triangulated vectors, each of shape nv x 3 x 2
    :param A:
    :param B:
    :return:
    """
    return A[:, :, 0] * B[:, :, 1] - A[:, :, 1] * B[:, :, 0]


@jit(f4[:,:](f4[:,:,:],f4[:,:,:]),cache=True)
def tdot(A, B):
    """
    Dot product of two triangulated vectors, each of shape nv x 3 x 2
    :param A:
    :param B:
    :return:
    """
    return A[:, :, 0] * B[:, :, 0] + A[:, :, 1] * B[:, :, 1]


@jit(f4[:,:,:,:](f4[:,:,:],f4[:,:,:]),cache=True)
def touter(A, B):
    """
    Outer product of two triangulated vectors, each of shape nv x 3 x 2
    :param A:
    :param B:
    :return:
    """
    return np.dstack((np.dstack((A[:, :, 0] * B[:, :, 0], A[:, :, 1] * B[:, :, 0])),
                      np.dstack((A[:, :, 0] * B[:, :, 1], A[:, :, 1] * B[:, :, 1])))).reshape(-1, 3, 2, 2)


# @jit(nopython=True)
# def tdet(A):
#     a1,a2,a3 = A[:,:,0]
#     b1,b2,b3 = A[:, :, 1]
#     c1,c2,c3 = A[:,:,2]
#     return a1*(b2*c3 - b3*c2) - a2*(b1*c3 - b3*c1) + a3*(b1*c2 - b2* c1)

# @jit(nopython=True)
# def tidentity(nv):
#     """
#     Generate an identity matrix for each element of a triangulation
#
#     I ~ (nv x 3 x 2 x 2), where the last two dims are an identity matrix.
#     :param nv:
#     :return:
#     """
#     I = np.zeros((nv, 3, 2, 2))
#     I[:, :, 0, 0] = 1
#     I[:, :, 1, 1] = 1
#     return I


# @jit(nopython=True)
# def tmatmul(A, B):
#     """
#
#     matrix multiplication of two triangulated matrices. Not in use atm.
#     :param A:
#     :param B:
#     :return:
#     """
#     AT, BT = A.T, B.T
#     return np.dstack(((AT[0] * BT[0, 0] + AT[1] * BT[1, 0]).T,
#                       (AT[0] * BT[0, 1] + AT[1] * BT[1, 1]).T))

@jit(f4[:](f4[:,:]),cache=True)
def sum_tri(A):
    return A[:,0] + A[:,1] + A[:,2]

@jit(f4[:](f4[:,:]),cache=True)
def prod_tri(A):
    return A[:,0] * A[:,1] * A[:,2]

def assemble_tri(tval, tri):
    """
    Sum all components of a given cell property.
    I.e. (nv x 3) --> (nc x 1)
    :param tval:
    :param tri:
    :return:
    """
    vals = coo_matrix((tval.ravel(), (tri.ravel(), np.zeros_like(tri.ravel()))), shape=(tri.max() + 1, 1))
    return vals.toarray().ravel()


def assemble_tri3(tval, tri):
    """
    The same as above, but for vector quantities

    (nv x 3 x 2) --> (nc x 2)
    :param tval:
    :param tri:
    :return:
    """
    vals = coo_matrix((tval.ravel(), (np.repeat(tri.ravel(), 2), np.tile((0, 1), tri.size))), shape=(tri.max() + 1, 2))
    return vals.toarray()


@jit(cache=True)
def find_neighbour_val(A, neighbours):
    """
    Check this
    :param A:
    :param neighbours:
    :return:
    """
    B = np.empty_like(A)
    for i, tneighbour in enumerate(neighbours):
        for j, neighbour in enumerate(tneighbour):
            B[i, j] = A[neighbour, j]
    return B


@jit(cache=True)
def repeat_mat(A):
    return np.dstack((A, A, A, A)).reshape(-1, 3, 2, 2)


@jit(cache=True)
def repeat_vec(A):
    return np.dstack((A, A))


@jit(f4[:,:](f4[:,:,:],f4),cache=True)
def circumcenter(C, L):
    """
    Find the circumcentre (i.e. vertex position) of each triangle in the triangulation.

    :param C: Cell centroids for each triangle in triangulation (n_c x 3 x 2) np.float32 array
    :param L: Domain size (np.float32)
    :return: Circumcentres/vertex-positions (n_v x 2) np.float32 array
    """
    ri, rj, rk = C.transpose(1, 2, 0)
    r_mean = (ri + rj + rk) / 3
    disp = r_mean - L / 2
    ri, rj, rk = np.mod(ri - disp, L), np.mod(rj - disp, L), np.mod(rk - disp, L)
    ax, ay = ri
    bx, by = rj
    cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    vs = np.empty((ax.size, 2), dtype=np.float32)
    vs[:, 0], vs[:, 1] = ux, uy
    vs = np.mod(vs + disp.T, L).astype(np.float32)
    return vs


def normalise(x):
    return (x - x.min()) / (x.max() - x.min())


def hexagonal_lattice(_rows=3, _cols=3, noise=0.0005, A=None):
    """
    Assemble a hexagonal lattice
    :param rows: Number of rows in lattice
    :param cols: Number of columns in lattice
    :param noise: Noise added to cell locs (Gaussian SD)
    :return: points (nc x 2) cell coordinates.
    """
    if A is None:
        A = 1.
    _A = np.max(A)
    rows = int(np.round(_rows/np.sqrt(_A)))
    cols = int(np.round(_cols/np.sqrt(_A)))
    points = []
    for row in range(rows * 2):
        for col in range(cols):
            x = (col + (0.5 * (row % 2))) * np.sqrt(3)
            y = row * 0.5
            x += np.random.normal(0, noise)
            y += np.random.normal(0, noise)
            points.append((x, y))
    points = np.asarray(points)
    if A is not None:
        points = points * np.sqrt(2 * np.sqrt(3) / 3) * np.sqrt(_A)
    return points


@jit(cache=True)
def get_neighbours(tri):
    """
    Given a triangulation, find the neighbouring triangles of each triangle.

    By convention, the column i in the output -- neigh -- corresponds to the triangle that is opposite the cell i in that triangle.

    Can supply neigh, meaning the algorithm only fills in gaps (-1 entries)

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: neighbourhood matrix to update {Optional}
    :return: (n_v x 3) np.int32 array, storing the three neighbouring triangles. Values correspond to the row numbers of tri
    """
    n_v = tri.shape[0]
    neigh = np.ones_like(tri, dtype=np.int32) * -1
    tri_compare = np.concatenate((tri.T, tri.T)).T.reshape((-1, 3, 2))
    for j in range(n_v):
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip, tri_sample_flip)).reshape(3, 2)
        for k in range(3):
            if neigh[j, k] == -1:
                neighb, l = np.nonzero((tri_compare[:, :, 0] == tri_i[k, 0]) * (tri_compare[:, :, 1] == tri_i[k, 1]))
                neighb, l = neighb[0], l[0]
                neigh[j, k] = neighb
                neigh[neighb, np.mod(2 - l, 3)] = j
    return neigh.astype(np.int32)


@jit(f4[:, :](f4[:, :], i4[:, :], f4),cache=True)
def tri_angles_periodic(x, tri, L):
    """
    Same as **tri_angles** apart from accounts for periodic triangulation (i.e. the **L**)

    Find angles that make up each triangle in the triangulation. By convention, column i defines the angle
    corresponding to cell centroid i

    :param x: Cell centroids (n_c x 2) np.float32 array
    :param tri: Triangulation (n_v x 3) np.int32 array
    :param L: Domain size (np.float32)
    :return: tri_angles (n_v x 3) np.flaot32 array (in radians)
    """
    three = np.array([0, 1, 2],dtype=np.int32)
    i_b = np.mod(three + 1, 3)
    i_c = np.mod(three + 2, 3)

    C = np.empty((tri.shape[0], 3, 2),dtype=np.float32)
    for i, TRI in enumerate(tri):
        C[i] = x[TRI]
    a2 = (np.mod(C[:, i_b, 0] - C[:, i_c, 0] + L / 2, L) - L / 2) ** 2 + (
            np.mod(C[:, i_b, 1] - C[:, i_c, 1] + L / 2, L) - L / 2) ** 2
    b2 = (np.mod(C[:, :, 0] - C[:, i_c, 0] + L / 2, L) - L / 2) ** 2 + (
            np.mod(C[:, :, 1] - C[:, i_c, 1] + L / 2, L) - L / 2) ** 2
    c2 = (np.mod(C[:, i_b, 0] - C[:, :, 0] + L / 2, L) - L / 2) ** 2 + (
            np.mod(C[:, i_b, 1] - C[:, :, 1] + L / 2, L) - L / 2) ** 2

    cos_Angles = (b2 + c2 - a2) / (2 * np.sqrt(b2) * np.sqrt(c2))
    Angles = np.arccos(cos_Angles)
    return Angles.astype(np.float32)

