import numpy as np
import synmorph.tri_functions as trf
from numba import jit
from scipy.sparse import coo_matrix

@jit(nopython=True)
def connected_components(tri, c_types, n_c):
    """
    For a two-state system, performs an efficient DFS-like search on the
    triangulation to determine the number of connected components in each
    of the subgraphs for cell types A and B.

    :param tri: triangulation
    :param c_types: cell types for each corresponding cell index
    :param n_c: number of cells
    :return:
    """
    lookup = np.arange(n_c)
    mask = np.array((True, True))
    dtri_flat, dtrip1_flat = tri.ravel(), trf.roll_int(tri,1).ravel()
    same_edges = (c_types[dtri_flat] == c_types[dtrip1_flat])
    dtri_flat, dtrip1_flat = dtri_flat[same_edges], dtrip1_flat[same_edges]
    while mask.any():
        orig, rolled = lookup.take(dtri_flat), lookup.take(dtrip1_flat)
        mask = orig > rolled
        for j in np.unique(orig[mask]):
            lookup[lookup == j] = np.min(rolled[orig == j])
    bclook = np.bincount(lookup, minlength=lookup.size)
    bclookn0 = bclook != 0
    orig_ids = lookup[bclookn0]
    num_in_cluster = bclook[bclookn0]
    ctype0_mask = c_types[orig_ids] == 0
    ctype1_mask = ~ctype0_mask
    num_in_cluster0, num_in_cluster1 = num_in_cluster[ctype0_mask], num_in_cluster[ctype1_mask]
    order0, order1 = np.argsort(num_in_cluster0), np.argsort(num_in_cluster1)
    orig_ids0, orig_ids1 = orig_ids[ctype0_mask].take(order0), orig_ids[ctype1_mask].take(order1)
    num_in_cluster0, num_in_cluster1 = num_in_cluster0.take(order0), num_in_cluster1.take(order1)

    lookup_new = lookup.copy()
    for i, j in enumerate(orig_ids0):
        mask = lookup == j
        lookup_new[mask] = -i - 1
    for i, j in enumerate(orig_ids1):
        mask = lookup == j
        lookup_new[mask] = i + 1
    return (num_in_cluster0, num_in_cluster1), lookup_new

@jit(nopython=True)
def count_connected_components(tri, c_types, n_c):
    (num_in_cluster0, num_in_cluster1), lookup_new = connected_components(tri, c_types, n_c)
    return num_in_cluster0.size,num_in_cluster1.size

@jit(nopython=True)
def boundary_cells(tri, num_in_cluster0=None, num_in_cluster1=None, lookup=None, c_types=None, n_c=None):
    """
    For a two state system (A,B), calculate the number of cells of each type that lie on a
    boundary between cell types A and B

    Can use **connected_components** to pre-compute the number of cells in each cluster and the lookup table.
    :return:
    """
    if (num_in_cluster0 is None) or (num_in_cluster1 is None) or (lookup is None):
        assert c_types is not None, "Must specify c_types"
        assert n_c is not None, "Must specify the number of cells, n_c"
        (num_in_cluster0, num_in_cluster1), lookup = connected_components(tri, c_types, n_c)
    tri_flat = tri.ravel()
    sct_flat = lookup[tri_flat]
    sctp1_flat = trf.roll_int(sct_flat.reshape(tri.shape),1).ravel()
    num_boundary0, num_boundary1 = num_in_cluster0.copy(), num_in_cluster1.copy()
    for i in range(num_boundary0.size):
        k = -i - 1
        mask = (sct_flat == k) * (sctp1_flat != k)
        num_boundary0[i] = (np.bincount(tri_flat[mask]) != 0).sum()
    for i in range(num_boundary1.size):
        k = i + 1
        mask = (sct_flat == k) * (sctp1_flat != k)
        num_boundary1[i] = (np.bincount(tri_flat[mask]) != 0).sum()
    return num_boundary0, num_boundary1


@jit(nopython=True)
def boundary_cell_count(tri, num_in_cluster0=None, num_in_cluster1=None, lookup=None, c_types=None, n_c=None):
    if (num_in_cluster0 is None) or (num_in_cluster1 is None) or (lookup is None):
        assert c_types is not None, "Must specify c_types"
        assert n_c is not None, "Must specify the number of cells, n_c"
        (num_in_cluster0, num_in_cluster1), lookup= connected_components(tri, c_types, n_c)
    num_boundary0, num_boundary1 = boundary_cells(tri, num_in_cluster0, num_in_cluster1, lookup, c_types, n_c)
    return num_boundary0.sum(),num_boundary1.sum()


@jit(nopython=True)
def get_entropy(num_in_cluster0=None, num_in_cluster1=None, tri=None, c_types=None, n_c=None):
    """
    Entropy_i = -Sum_i {p_{ij}log(p_{ij})}

    where i is cell type A or B, j is which of the multiple clusters it is in.
    p_ij = N_ij/Sum_j(N_ij)

    :param num_in_cluster0:
    :param num_in_cluster1:
    :return:
    """
    if (num_in_cluster0 is None) or (num_in_cluster1 is None):
        assert tri is not None, "Must specify c_types"
        assert c_types is not None, "Must specify c_types"
        assert n_c is not None, "Must specify the number of cells, n_c"
        (num_in_cluster0, num_in_cluster1), lookup = connected_components(tri, c_types, n_c)
    p0 = num_in_cluster0/num_in_cluster0.sum()
    p1 = num_in_cluster1/num_in_cluster1.sum()
    return -(np.log(p0+1e-19)*p0).sum(), -(np.log(p1+1e-19)*p1).sum()

@jit(nopython=True)
def get_boundary_entropy(num_in_cluster0=None,num_in_cluster1=None,num_boundary0=None, num_boundary1=None, tri=None, c_types=None, n_c=None):
    """
    Consider four possible types of states a cell can be in:
    central vs boundary, A vs B.
    Compute entropy for A and B.

    :param num_in_cluster0:
    :param num_in_cluster1:
    :param num_boundary0:
    :param num_boundary1:
    :return:
    """
    if (num_in_cluster0 is None) or (num_in_cluster1 is None) or (num_boundary0 is None) or (num_boundary1 is None):
        assert tri is not None, "Must specify c_types"
        assert c_types is not None, "Must specify c_types"
        assert n_c is not None, "Must specify the number of cells, n_c"
        (num_in_cluster0, num_in_cluster1), lookup = connected_components(tri, c_types, n_c)
        num_boundary0, num_boundary1 = boundary_cells(tri,num_in_cluster0,num_in_cluster1,lookup)
    num_central0 = num_in_cluster0 - num_boundary0
    num_central1 = num_in_cluster1 - num_boundary1
    p0 = np.concatenate((num_central0/num_in_cluster0.sum(),
                        num_boundary0/num_in_cluster0.sum()))
    p1 = np.concatenate((num_central1/num_in_cluster1.sum(),
                        num_boundary1/num_in_cluster1.sum()))
    p0 += 1e-19
    p1 += 1e-19
    return -(np.log(p0)*p0).sum(), -(np.log(p1)*p1).sum()

def get_mat_from_tri(tri):
    dtri_flat, dtrip1_flat = tri.ravel(), trf.roll_int(tri,1).ravel()
    mat = coo_matrix(([1]*len(dtri_flat),(dtri_flat,dtrip1_flat)))
    return mat

def get_lost_edges(tri0,tri1):
    tri0mat = get_mat_from_tri(tri0)
    tri1mat = get_mat_from_tri(tri1)
    lost_edges = np.array((tri0mat > tri1mat).nonzero()).T
    if lost_edges.size != 0:
        lost_edges = lost_edges[lost_edges[:,0]<lost_edges[:,1]]
        return lost_edges
    else:
        return None
