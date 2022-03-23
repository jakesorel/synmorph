import numpy as np
from numba import jit
from scipy import sparse
import synmorph.tri_functions as trf

@jit(nopython=True)
def get_edge_list(tri):
    """ Includes double counting of edges."""
    edges = np.empty((tri.shape[0]*3,2),dtype=np.int64)
    for i,tr in enumerate(tri):
        for j in range(3):
            edges[3*i+j] = np.roll(tr,j)[:2]
    # edges = np.row_stack((np.roll(tri, i, axis=1)[:, :2] for i in range(3)))
    return edges

@jit(nopython=True)
def get_edge_list_non_duplicated(tri):
    """ Includes double counting of edges."""
    edges = np.empty((tri.shape[0]*3,2),dtype=np.int64)
    for i,tr in enumerate(tri):
        for j in range(3):
            edges[3*i+j] = np.roll(tr,j)[:2]
    edges = edges[edges[:,0]<edges[:,1]]
    # edges = np.row_stack((np.roll(tri, i, axis=1)[:, :2] for i in range(3)))
    return edges

def get_quartets_from_tri(tri,neigh,k2s):
    edges = np.empty((tri.shape[0],3,2),dtype=np.int64)
    for i,tr in enumerate(tri):
        for j in range(3):
            edges[i,j] = np.roll(tr,j)[:2]
    emask = edges[...,0] < edges[...,1]
    first_tri,js = np.nonzero(emask)
    paired_tris = neigh[emask]
    paired_k2s = k2s[emask]
    ne = emask.sum()
    a,b,d= np.zeros((3,ne),dtype=np.int32)
    for i, (ft,j) in enumerate(zip(first_tri,js)):
        a[i], b[i], d[i] = np.roll(tri[ft],-j)
    c = tri[paired_tris,paired_k2s]
    quartets = np.column_stack((a,b,c,d))

    ##Note, can call a quartet by picking any tri, and picking a cell in the tri as "A"
    return quartets

@jit(nopython=True)
def get_a_in_tri(tri,a):
    mask =  tri == a
    return mask

@jit(nopython=True)
def any1(mask):
    return mask[:, 0] + mask[:, 1] + mask[:, 2]

@jit(nopython=True)
def get_CCW_neighbour(da_i,da_j,tri,neigh,k2s):
    da_CW = (da_j-1)%3
    Na_i = neigh[da_i,da_CW]
    Na_j = k2s[da_i,da_CW]
    Na = tri[Na_i,Na_j]
    return Na_i,Na_j,Na

@jit(nopython=True)
def get_sole_neighbours(da_i,da_j,ab,tri,neigh,k2s):
    Nas = np.ones((3),dtype=np.int64)*-1
    Na_i,Na_j = da_i,da_j
    continue_while = True
    i = 0
    while continue_while:
        # if i > 2:
        #     raise ValueError
        # else:
        Na_i,Na_j,Na = get_CCW_neighbour(Na_i,Na_j,tri,neigh,k2s)
        if Na !=ab:
            Nas[i] = Na
            i+=1
        else:
            continue_while = False
    return Nas

@jit(nopython=True)
def choose_random_quartet(ntri):
    tri_0i,tri_0j = int(np.random.random()*ntri),int(3*np.random.random())
    return tri_0i,tri_0j

@jit(nopython=True)
def get_quartet(tri,neigh,k2s,tri_0i,tri_0j):
    a,b,d = np.roll(tri[tri_0i],-tri_0j)
    tri_1i,tri_1j = neigh[tri_0i,tri_0j],k2s[tri_0i,tri_0j]
    c = tri[tri_1i,tri_1j]

    # quartet = np.array((a,b,c,d))

    tri0_da =(tri_0j+1)%3
    da_i = neigh[tri_0i,tri0_da]
    da_j = k2s[tri_0i,tri0_da]
    da = tri[da_i,da_j]

    tri0_ab =(tri_0j-1)%3
    ab_i = neigh[tri_0i,tri0_ab]
    ab_j = k2s[tri_0i,tri0_ab]
    ab = tri[ab_i,ab_j]


    tri1_cd =(tri_1j-1)%3
    cd_i = neigh[tri_1i,tri1_cd]
    cd_j = k2s[tri_1i,tri1_cd]
    cd = tri[cd_i,cd_j]

    tri1_bc =(tri_1j+1)%3
    bc_i = neigh[tri_1i,tri1_bc]
    bc_j = k2s[tri_1i,tri1_bc]
    bc = tri[bc_i,bc_j]

    return tri_0i,tri_0j,tri_1i,tri_1j,a,b,c,d,da,ab,bc,cd,da_i,ab_i,bc_i,cd_i,da_j,ab_j,bc_j,cd_j


@jit(nopython=True)
def get_quartet_neigh_cids(quartet_info,tri,neigh,k2s):
    tri_0i,tri_0j,tri_1i,tri_1j,a,b,c,d,da,ab,bc,cd,da_i,ab_i,bc_i,cd_i,da_j,ab_j,bc_j,cd_j = quartet_info

    sn_a = get_sole_neighbours(da_i, da_j, ab, tri, neigh, k2s)
    sn_b = get_sole_neighbours(ab_i, ab_j, bc, tri, neigh, k2s)
    sn_c = get_sole_neighbours(bc_i, bc_j, cd, tri, neigh, k2s)
    sn_d = get_sole_neighbours(cd_i, cd_j, da, tri, neigh, k2s)

    neigh_cids = np.zeros((20),dtype=np.int64)
    neigh_cids[:8:4] = a,da
    neigh_cids[8::4] = sn_a
    neigh_cids[1:9:4] = b,ab
    neigh_cids[9::4] = sn_b
    neigh_cids[2:10:4] = c,bc
    neigh_cids[10::4] = sn_c
    neigh_cids[3:11:4] = d,cd
    neigh_cids[11::4] = sn_d
    return neigh_cids

@jit(nopython=True)
def get_quartet_state(neigh_cids,cols):
    state = cols.take(neigh_cids)
    return state

@jit(nopython=True)
def get_hash_from_state(state):
    return ((3**(19-np.arange(20)))*state).sum()

@jit(nopython=True)
def swap_ids(i,j,init_list):
    list_new = init_list.copy()
    list_new[i::4],list_new[j::4] = init_list[j::4],init_list[i::4]
    return list_new

@jit(nopython=True)
def swap_cols(state):
    state_new = state.copy()
    state_new[state==1] = 2
    state_new[state==2] = 1
    return state_new


@jit(nopython=True)
def get_cannonical_hash(state):
    lr_symm = swap_ids(1,3,state)
    ud_symm = swap_ids(0,2,state)
    lrud_symm = swap_ids(0,2,lr_symm)

    orig_inv = swap_cols(state)
    lr_symm_inv = swap_cols(lr_symm)
    ud_symm_inv = swap_cols(ud_symm)
    lrud_symm_inv = swap_cols(lrud_symm)


    orig_hash = get_hash_from_state(state)
    lr_hash = get_hash_from_state(lr_symm)
    ud_hash = get_hash_from_state(ud_symm)
    lrud_hash = get_hash_from_state(lrud_symm)

    orig_hash_inv = get_hash_from_state(orig_inv)
    lr_hash_inv = get_hash_from_state(lr_symm_inv)
    ud_hash_inv = get_hash_from_state(ud_symm_inv)
    lrud_hash_inv = get_hash_from_state(lrud_symm_inv)
    return np.array((orig_hash,lr_hash,ud_hash,lrud_hash,orig_hash_inv,lr_hash_inv,ud_hash_inv,lrud_hash_inv)).min()



@jit(nopython=True)
def get_state_after_t1(state):
    """
    Compute the hash after the T1 transition
    :return:
    """
    state_new = state.copy()
    state_new[0::4],state_new[1::4],state_new[2::4],state_new[3::4] = state[3::4],state[0::4],state[1::4],state[2::4]
    return state_new

@jit(nopython=True)
def get_quartet_degree(state):
    degree_a = (state[8::4] != 0).sum() + 4
    degree_b = (state[9::4] != 0).sum() + 5
    degree_c = (state[10::4] != 0).sum() + 4
    degree_d = (state[11::4] != 0).sum() + 5
    degrees = np.array((degree_a,degree_b,degree_c,degree_d))
    return degrees

@jit(nopython=True)
def check_ok_for_t1(state):
    degrees = get_quartet_degree(state)
    accept = True
    if (degrees[0]>=7)+(degrees[2]>=7)+(degrees[1]<=5)+(degrees[3]<=5):
        accept=False
    return accept



@jit(nopython=True)
def get_t1_degree(new_state):
    degree_a = (new_state[8::4] == 0).sum() + 4
    degree_b = (new_state[9::4] == 0).sum() + 5
    degree_c = (new_state[10::4] == 0).sum() + 4
    degree_d = (new_state[11::4] == 0).sum() + 5
    degrees = np.array((degree_a,degree_b,degree_c,degree_d))
    return degrees
#
# @jit(nopython=True)
# def check_t1_degree(new_state):
#     degrees = get_t1_degree(new_state)
#     accept = True
#     if (degrees<5).any()+(degrees>7).any():
#         accept = False
#     return accept


@jit(nopython=True)
def update_mesh(quartet_info,tri,neigh,k2s):
    """
    Update tri, neigh and k2. Inspect the equiangulation code for some inspo.
    :return:
    """

    tri_0i,tri_0j,tri_1i,tri_1j,a,b,c,d,da,ab,bc,cd,da_i,ab_i,bc_i,cd_i,da_j,ab_j,bc_j,cd_j = quartet_info

    tri_new = tri.copy()
    neigh_new = neigh.copy()
    k2s_new = k2s.copy()

    ###SWAP C FOR D
    tri_new[tri_0i,(tri_0j-1)%3] = c
    tri_new[tri_1i,(tri_1j-1)%3] = a

    ##fix the neighbours
    neigh_new[tri_0i,tri_0j] = neigh[tri_1i,(tri_1j+1)%3]
    neigh_new[tri_0i,(tri_0j+1)%3] = neigh[bc_i,bc_j]
    neigh_new[tri_0i,(tri_0j+2)%3] = neigh[tri_0i,(tri_0j+2)%3]
    neigh_new[tri_1i,tri_1j] = neigh[tri_0i,(tri_0j+1)%3]
    neigh_new[tri_1i,(tri_1j+1)%3] = neigh[da_i,da_j]
    neigh_new[tri_1i,(tri_1j+2)%3] = neigh[tri_1i,(tri_1j+2)%3]

    ##And the k2s (may need to check this...)
    k2s_new[tri_0i,tri_0j] = k2s[tri_1i,(tri_1j+1)%3]
    k2s_new[tri_0i,(tri_0j+1)%3] = k2s[bc_i,bc_j]
    k2s_new[tri_0i,(tri_0j+2)%3] = k2s[tri_0i,(tri_0j+2)%3]
    k2s_new[tri_1i,tri_1j] = k2s[tri_0i,(tri_0j+1)%3]
    k2s_new[tri_1i,(tri_1j+1)%3] = k2s[da_i,da_j]
    k2s_new[tri_1i,(tri_1j+2)%3] = k2s[tri_1i,(tri_1j+2)%3]


    #adjust neighbourhoods of neighbours.
    neigh_new[bc_i,bc_j] = tri_0i
    k2s_new[bc_i,bc_j] = tri_0j
    neigh_new[da_i,da_j] = tri_1i
    k2s_new[da_i,da_j]= tri_1j


    ###The get_sole_neighbours function is now failing. This is indicative of something going wrong in the above ^^^

    return tri_new,neigh_new,k2s_new

@jit(nopython=True)
def perform_t1(tri, neigh, k2s,cols,ntri):
    t1_degree_ok = False
    niter = 0
    tri_new,neigh_new,k2s_new = tri.copy(), neigh.copy(), k2s.copy()
    quartet_hash=None
    while (not t1_degree_ok)*(niter<400):
        choice = choose_random_quartet(ntri)
        quartet_info = get_quartet(tri, neigh, k2s, *choice)
        neigh_cids = get_quartet_neigh_cids(quartet_info, tri, neigh, k2s)
        state = get_quartet_state(neigh_cids,cols)
        # t1_state = get_state_after_t1(state)
        t1_degree_ok = check_ok_for_t1(state)
        niter +=1
    if niter>=399:
        raise ValueError
    else:
        quartet_hash = get_cannonical_hash(state)
        tri_new,neigh_new,k2s_new = update_mesh(quartet_info,tri,neigh,k2s)
    return quartet_hash,tri_new,neigh_new,k2s_new



def build_adj(tri,nc):
    edges = get_edge_list_non_duplicated(tri)
    adj = sparse.coo_matrix(([True]*edges.shape[0],
                             (edges[:,0],edges[:,1])),shape=(nc,nc))
    adj += adj.T
    # adj = sparse.csr_matrix(adj)
    return adj

def get_cc(tri,nc,ctypes):
    """
    Get the connectivity matrix from a state vector. Uses scipy function.
    Min value will be 2, as discussed.
    :param state:
    :return:
    """

    edges = get_edge_list(tri)
    state_edges = ctypes[edges]
    same_edges = (state_edges[:,0] == state_edges[:,1])
    adj_mat = sparse.coo_matrix(([1]*sum(same_edges),(edges[same_edges,0],edges[same_edges,1])),shape = (nc,nc))
    ncc,ids = sparse.csgraph.connected_components(adj_mat,directed=False)
    return ncc

@jit(nopython=True)
def tri_num_connected(tri,ctypes,nc):
    lookup = np.arange(nc)
    mask = np.array((True,True))
    dtri_flat,dtrip1_flat = tri.ravel(), trf.roll(tri).ravel()
    same_edges = (ctypes[dtri_flat] == ctypes[dtrip1_flat])
    dtri_flat,dtrip1_flat = dtri_flat[same_edges],dtrip1_flat[same_edges]
    while mask.any():
        orig,rolled = lookup.take(dtri_flat), lookup.take(dtrip1_flat)
        mask = orig > rolled
        for j in np.unique(orig[mask]):
            lookup[lookup==j] = np.min(rolled[orig==j])
    bclook = np.bincount(lookup,minlength=lookup.size)
    bclookn0 = bclook!=0
    orig_ids = lookup[bclookn0]
    num_in_cluster = bclook[bclookn0]
    ctype0_mask = ctypes[orig_ids] == 0
    ctype1_mask = ~ctype0_mask
    num_in_cluster0,num_in_cluster1 = num_in_cluster[ctype0_mask],num_in_cluster[ctype1_mask]
    order0,order1 = np.argsort(num_in_cluster0),np.argsort(num_in_cluster1)
    orig_ids0,orig_ids1 = orig_ids[ctype0_mask].take(order0),orig_ids[ctype1_mask].take(order1)
    num_in_cluster0,num_in_cluster1 = num_in_cluster0.take(order0),num_in_cluster1.take(order1)

    lookup_new = lookup.copy()
    for i,j in enumerate(orig_ids0):
        mask = lookup==j
        lookup_new[mask] = -i-1
    for i,j in enumerate(orig_ids1):
        mask = lookup==j
        lookup_new[mask] = i+1
    return (num_in_cluster0,num_in_cluster1),lookup_new

@jit(nopython=True)
def get_n_boundary_cells(tri,num_in_cluster0,num_in_cluster1,lookup):
    tri_flat = tri.ravel()
    sct_flat = lookup[tri_flat]
    sctp1_flat = trf.roll(sct_flat.reshape(tri.shape)).ravel()
    num_boundary0, num_boundary1 = num_in_cluster0.copy(),num_in_cluster1.copy()
    for i in range(num_boundary0.size):
        k = -i-1
        mask = (sct_flat==k)*(sctp1_flat!=k)
        num_boundary0[i] = (np.bincount(tri_flat[mask]) != 0).sum()
    for i in range(num_boundary1.size):
        k = i+1
        mask = (sct_flat == k) * (sctp1_flat != k)
        num_boundary1[i] = (np.bincount(tri_flat[mask]) != 0).sum()
    return num_boundary0,num_boundary1
    ##do by ctype.



@jit(nopython=True)
def do_walk(tri0,neigh0,k2s0,cols,n_iter = 2000):
    # tri0,neigh0,k2s0 = tri.copy(), neigh.copy(), k2s.copy()
    ntri = tri0.shape[0]
    tri = tri0.copy()
    neigh = neigh0.copy()
    k2s = k2s0.copy()
    quartet_hashes = np.zeros((n_iter),dtype=np.int64)
    for i in range(n_iter):
        quartet_hash,tri,neigh,k2s = perform_t1(tri, neigh, k2s, cols, ntri)
        quartet_hashes[i] = quartet_hash
    return quartet_hashes


@jit(nopython=True)
def get_topology_hash(num_in_cluster0,num_in_cluster1,num_boundary0,num_boundary1,max_cluster_number):
    hash_mat = np.zeros((4,max_cluster_number),dtype=np.int64)
    hash_mat[0,:num_in_cluster0.size] = num_in_cluster0
    hash_mat[1,:num_in_cluster1.size] = num_in_cluster1
    hash_mat[2,:num_boundary0.size] = num_boundary0
    hash_mat[3,:num_boundary1.size] = num_boundary1
    hash_mat_flat = hash_mat.ravel()
    hash = ""
    if np.max(hash_mat)<100:
        for el in hash_mat_flat:
            if el<10:
                hash += "0"
            hash += str(el)
    return hash

def hash_to_topology(hash,max_cluster_number):
    hash_mat_flat = np.zeros((4*max_cluster_number),dtype=np.int64)
    for i in range(4*max_cluster_number):
        hash_mat_flat[i] = int(hash[2*i:2*(i+1)])
    return hash_mat_flat.reshape(4,-1)


