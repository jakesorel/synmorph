import numpy as np
import matplotlib.pyplot as plt
import synmorph as sm
from energy_analysis.quartets_from_mesh import *
from synmorph.force import get_tF,get_J,get_dvdr
from synmorph.mesh import disp23,disp33,get_k2
from scipy import sparse

tissue_params = {"L": 9,
                 "A0": 1,
                 "P0": 3.81,
                 "kappa_A": 1,
                 "kappa_P": 0.1,
                 "W": np.array(((0, 0.0762), (0.0762, 0))),
                 "a": 0,
                 "k": 0}
active_params = {"v0": 1e-1,
                 "Dr": 1e-1}
init_params = {"init_noise": 0.00005,
               "c_type_proportions": (0.5,0.5)}
run_options = {"equiangulate": True,
               "equi_nkill": 3}
simulation_params = {"dt": 0.025,
                     "tfin": 30,
                     "tskip": 10,
                     "grn_sim": None}
save_options = {"save": "last",
                "result_dir": "results",
                "name": "ctype_example",
                "compressed": True}


sim = sm.simulation(tissue_params=tissue_params,
                    active_params=active_params,
                    init_params=init_params,
                    simulation_params=simulation_params,
                    run_options=run_options,
                    save_options=save_options)



@jit(nopython=True)
def quartet_to_tri(neigh_cids):
    a,da =neigh_cids[:8:4]
    sn_a = neigh_cids[8::4]
    b,ab = neigh_cids[1:9:4]
    sn_b = neigh_cids[9::4]
    c,bc = neigh_cids[2:10:4]
    sn_c = neigh_cids[10::4]
    d,cd = neigh_cids[3:11:4]
    sn_d = neigh_cids[11::4]
    quartet_tri = np.zeros((22 - (neigh_cids == -1).sum(),3),dtype=np.int64)
    quartet_tri[0] = a,b,d
    quartet_tri[1] = c,d,b
    quartet_tri[2] = d,da,a
    quartet_tri[3] = a,ab,b
    quartet_tri[4] = b,bc,c
    quartet_tri[5] = c,cd,d
    i = 6
    a_neighs = np.concatenate((np.array((da,)),sn_a,np.array((ab,))))
    a_neighs = a_neighs[a_neighs!=-1]
    for j in range(a_neighs.size-1):
        quartet_tri[i] = a,a_neighs[j],a_neighs[j+1]
        i+=1
    b_neighs = np.concatenate((np.array((ab,)),sn_b,np.array((bc,))))
    b_neighs = b_neighs[b_neighs!=-1]
    for j in range(b_neighs.size-1):
        quartet_tri[i] = b,b_neighs[j],b_neighs[j+1]
        i+=1
    c_neighs = np.concatenate((np.array((bc,)),sn_c,np.array((cd,))))
    c_neighs = c_neighs[c_neighs!=-1]
    for j in range(c_neighs.size-1):
        quartet_tri[i] = c,c_neighs[j],c_neighs[j+1]
        i+=1
    d_neighs = np.concatenate((np.array((cd,)),sn_d,np.array((da,))))
    d_neighs = d_neighs[d_neighs!=-1]
    for j in range(d_neighs.size-1):
        quartet_tri[i] = d,d_neighs[j],d_neighs[j+1]
        i+=1
    return quartet_tri


@jit(nopython=True)
def normalize_neigh_cids(neigh_cids):
    norm_neigh_cids = np.zeros(20,dtype=np.int64)
    not_empty_mask = neigh_cids!=-1
    norm_neigh_cids[~not_empty_mask] = -1
    norm_neigh_cids[not_empty_mask] = np.arange(not_empty_mask.sum())
    return norm_neigh_cids



@jit(nopython=True)
def get_neighbours_quartet(tri):
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
    for j in np.arange(n_v):  # range(n_v):
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip, tri_sample_flip)).reshape(3, 2)
        for k in range(3):
            if neigh[j, k] == -1:
                mask = (tri_compare[:, :, 0] == tri_i[k, 0]) * (tri_compare[:, :, 1] == tri_i[k, 1])
                if mask.any():
                    neighb, l = np.nonzero(mask)
                    neighb, l = neighb[0], l[0]
                    neigh[j, k] = neighb
                    # neigh[neighb, np.mod(2 - l, 3)] = j
    return neigh


@jit(nopython=True)
def get_tF_quartet(vp1_vm1, v_vm1, v_vp1, v_x, lm1, lp1, Jm, Jp, kappa_A, kappa_P, A0, P0, A, P, quartet_tri):
    dAdv_j = np.dstack(
        (vp1_vm1[:, :, 1], -vp1_vm1[:, :, 0])) * 0.5  ##shoelace theorem: i.e. derivative of cross product.

    dPdv_j_m = v_vm1 / np.expand_dims(lm1, 2)
    dPdv_j_p = v_vp1 / np.expand_dims(lp1, 2)
    dPdv_j_m = trf.replace_val(dPdv_j_m, np.isnan(dPdv_j_m), 0)
    dPdv_j_p = trf.replace_val(dPdv_j_m, np.isnan(dPdv_j_p), 0)

    dPdv_j = dPdv_j_p + dPdv_j_m

    dtEdv_l_v_j = dPdv_j_m * np.expand_dims(Jm, 2) + dPdv_j_p * np.expand_dims(Jp, 2)

    dtEdA = trf.tri_call(2 * kappa_A * (A - A0), quartet_tri)
    dtEdP = trf.tri_call(2 * kappa_P * (P - P0), quartet_tri)

    dtE_dv = np.expand_dims(dtEdA, 2) * dAdv_j + np.expand_dims(dtEdP, 2) * dPdv_j + dtEdv_l_v_j
    dtE_dv = dtE_dv[:, 0] + dtE_dv[:, 1] + dtE_dv[:, 2]  # sum over the three contributions

    dvdr = get_dvdr(v_x)  # order is wrt cell i

    dtE_dv = np.expand_dims(dtE_dv, 2)

    dEdr_x = dtE_dv[:, 0] * dvdr[:, :, 0, 0] + dtE_dv[:, 1] * dvdr[:, :, 0, 1]
    dEdr_y = dtE_dv[:, 0] * dvdr[:, :, 1, 0] + dtE_dv[:, 1] * dvdr[:, :, 1, 1]

    dEdr = np.dstack((dEdr_x, dEdr_y))
    F = - dEdr
    return F

tri = sim.t.mesh.tri
neigh = sim.t.mesh.neigh
ctypes = sim.t.c_types
cols = np.zeros((ctypes.size+1),dtype=np.int64)
cols[:-1] = ctypes + 1
k2s = sim.t.mesh.k2s
ntri = tri.shape[0]
nc = tri.max() + 1
L = sim.t.mesh.L
W = sim.t.W

def append0(vec):
    return np.concatenate((vec,(0,)))


choice = choose_random_quartet(ntri)
quartet_info = get_quartet(tri,neigh,k2s,*choice)
neigh_cids = get_quartet_neigh_cids(quartet_info,tri,neigh,k2s)
norm_neigh_cids = normalize_neigh_cids(neigh_cids)
quartet_tri = quartet_to_tri(norm_neigh_cids)
quartet_neigh = get_neighbours_quartet(quartet_tri)
quartet_k2s = get_k2(quartet_tri,quartet_neigh)
quartet_x = sim.t.mesh.x[neigh_cids[neigh_cids!=-1]]
quartet_ctypes = ctypes[neigh_cids[neigh_cids!=-1]]
# quartet_ctypes *=0
###set the coefficients of the energy functional for the ghost cells to 0
kappa_A_quartet = np.zeros_like(quartet_ctypes,dtype=np.int64)
kappa_A_quartet[:4] = sim.t.kappa_A[0]*10
kappa_P_quartet = np.zeros_like(quartet_ctypes,dtype=np.int64)
kappa_P_quartet[:4] = sim.t.kappa_P[0]
A0_quartet = np.zeros_like(quartet_ctypes,dtype=np.int64)
A0_quartet[:4] = sim.t.A0[0]
P0_quartet = np.zeros_like(quartet_ctypes,dtype=np.int64)
P0_quartet[:4] = sim.t.P0[0]

plt.scatter(quartet_x[:,0],quartet_x[:,1],c=quartet_ctypes)
for lbl,nlbl in zip(neigh_cids,norm_neigh_cids):
    if lbl!=-1:
        xx = sim.t.mesh.x
        plt.text(xx[lbl,0],xx[lbl,1],nlbl)
plt.show()

def get_F_quartet(quartet_x):
    tx = trf.tri_call3(quartet_x, quartet_tri)
    vs = trf.circumcenter(tx, L)
    vn = trf.tri_call3(vs, quartet_neigh) ##note the dodgy filling of -1s here and below.
    vp1 = trf.roll3(vn)
    vm1 = trf.roll3(vn, -1)
    v_x = disp23(vs, tx, L)
    lv_x = trf.tnorm(v_x)
    v_vp1 = disp23(vs, vp1, L)
    lp1 = trf.tnorm(v_vp1)
    v_vm1 = disp23(vs, vm1, L)
    lm1 = trf.tnorm(v_vm1)
    vp1_x = disp33(vp1, tx, L)
    vm1_x = disp33(vm1, tx, L)
    vp1_vm1 = disp33(vp1, vm1, L)

    P = trf.assemble_tri(lp1, quartet_tri)
    A_components = 0.5 * trf.tcross(vm1_x, v_x)
    A = trf.assemble_tri(A_components, quartet_tri)

    tc_types = trf.tri_call(quartet_ctypes, quartet_tri)
    tc_typesp = trf.roll(tc_types, -1)
    tc_typesm = trf.roll(tc_types, 1)
    nc_types = 2

    Jp = get_J(W, tc_types, tc_typesp, nc_types)
    Jm = get_J(W, tc_types, tc_typesm, nc_types)



    ##set the interface energies for the ghost cells to 0, including the interface contribution w/ quartet cells.
    quartet_mask = quartet_tri <4
    Jp *= quartet_mask
    Jm *= quartet_mask


    # Jp*=0
    # Jm*=0

    #
    # Jp *=0
    # Jm *=0

    tF = get_tF_quartet(vp1_vm1,
                v_vm1,
                v_vp1,
                v_x,
                lm1,
                lp1,
                Jm,
                Jp,
                kappa_A_quartet,
                kappa_P_quartet,
                A0_quartet,
                P0_quartet,
                A,
                P,
                quartet_tri)
    F = trf.assemble_tri3(tF,
                               quartet_tri)  ##this assembles the total force on each cell centroid by summing the contributions from each triangle.
    return F

def get_E_quartet(quartet_x):
    tx = trf.tri_call3(quartet_x, quartet_tri)
    vs = trf.circumcenter(tx, L)
    vn = trf.tri_call3(vs, quartet_neigh) ##note the dodgy filling of -1s here and below.
    vp1 = trf.roll3(vn)
    vm1 = trf.roll3(vn, -1)
    v_x = disp23(vs, tx, L)
    lv_x = trf.tnorm(v_x)
    v_vp1 = disp23(vs, vp1, L)
    lp1 = trf.tnorm(v_vp1)
    v_vm1 = disp23(vs, vm1, L)
    lm1 = trf.tnorm(v_vm1)
    vp1_x = disp33(vp1, tx, L)
    vm1_x = disp33(vm1, tx, L)
    vp1_vm1 = disp33(vp1, vm1, L)

    P = trf.assemble_tri(lp1, quartet_tri)
    A_components = 0.5 * trf.tcross(vm1_x, v_x)
    A = trf.assemble_tri(A_components, quartet_tri)

    # l_int = sparse.coo_matrix((lp1.ravel(), (quartet_tri.ravel(), trf.roll(quartet_tri, -1).ravel())))

    tc_types = trf.tri_call(quartet_ctypes, quartet_tri)
    tc_typesp = trf.roll(tc_types, -1)
    tc_typesm = trf.roll(tc_types, 1)
    nc_types = 2

    Jp = get_J(W, tc_types, tc_typesp, nc_types)
    Jm = get_J(W, tc_types, tc_typesm, nc_types)



    ##set the interface energies for the ghost cells to 0, including the interface contribution w/ quartet cells.
    quartet_mask = quartet_tri <4
    Jp *= quartet_mask
    Jm *= quartet_mask

    # Jp*=0
    # Jm*=0
    int_energy = trf.assemble_tri(Jp*lp1,quartet_tri)
    E = kappa_A_quartet*(A-A0_quartet)**2 + kappa_P_quartet*(P-P0_quartet)**2 + int_energy
    return E.sum()

nt = 3000
quartet_x_save = np.zeros(((nt,)+quartet_x.shape))
quartet_x_step = quartet_x.copy()
dt = 0.01
for i in range(nt):
    quartet_x_step += dt*get_F_quartet(quartet_x_step)
    quartet_x_save[i] = quartet_x_step


E_save = np.array([get_E_quartet(quartet_x_step) for quartet_x_step in quartet_x_save])
plt.plot(E_save)
plt.show()
##it works. but is degenerate


"""
Reflect neighbouring points 
"""




#
#


def graph_plot_x(adj,pos):
    start,end = adj.nonzero()
    x = np.zeros((start.size*3,2))
    x[2::3] = None
    x[::3] = pos[start]
    x[1::3] = pos[end]
    return x

edges = get_edge_list(quartet_tri)

adj = sparse.coo_matrix(([True]*len(edges),(edges[:,0],edges[:,1])))

quartet_x = quartet_x_save[0]
plt.scatter(quartet_x[:,0],quartet_x[:,1])
plt.plot(*graph_plot_x(adj,quartet_x).T)
# for i, (lbl,nlbl) in enumerate(zip(neigh_cids[neigh_cids!=-1],norm_neigh_cids[neigh_cids!=-1])):
#     plt.text(quartet_x[i,0],quartet_x[i,1],nlbl)
# plt.quiver(quartet_x[:4,0],quartet_x[:4,1],F[:4,0],F[:4,1])

i = np.argmin(E_save)
quartet_x = quartet_x_save[i]
plt.scatter(quartet_x[:,0],quartet_x[:,1])
plt.plot(*graph_plot_x(adj,quartet_x).T)

# for i, (lbl,nlbl) in enumerate(zip(neigh_cids[neigh_cids!=-1],norm_neigh_cids[neigh_cids!=-1])):
#     plt.text(quartet_x[i,0],quartet_x[i,1],nlbl)
# plt.quiver(quartet_x[:4,0],quartet_x[:4,1],F[:4,0],F[:4,1])

plt.show()


"""
We want a function that takes in a quartet and returns the minimum energy. 

"""

def cost(X):
    quartet_x = X.reshape(-1,2)
    return get_E_quartet(quartet_x)

from scipy.optimize import minimize,NonlinearConstraint



@jit(nopython=True)
def t2_constraint(X):
    quartet_x = X.reshape(-1,2)
    tx = trf.tri_call3(quartet_x, quartet_tri)
    vs = trf.circumcenter(tx, L)
    return ((vs[0]-vs[1])**2).sum()

def voronoi_constraint_all(X):
    quartet_x = X.reshape(-1,2)
    Angles = trf.tri_angles_periodic(quartet_x, quartet_tri, L)
    sumAngle = Angles[quartet_neigh, quartet_k2s] + Angles
    return np.pi - sumAngle.ravel()

def voronoi_constraint(X,i):
    quartet_x = X.reshape(-1,2)
    Angles = trf.tri_angles_periodic(quartet_x, quartet_tri, L)
    sumAngle = Angles[quartet_neigh, quartet_k2s] + Angles
    return np.pi - sumAngle.ravel()[i]
#
# @jit(nopython=True)
# def t2_constraint_abd(X):
#     quartet_x = X.reshape(-1,2)
#     return ((quartet_x[0]-quartet_x[1])**2).sum() - ((quartet_x[0]-quartet_x[3])**2).sum()
#
# @jit(nopython=True)
# def t2_constraint_bcd(X):
#     quartet_x = X.reshape(-1,2)
#     return ((quartet_x[1]-quartet_x[2])**2).sum() - ((quartet_x[1]-quartet_x[3])**2).sum()




quartet_x0 = quartet_x# + np.random.normal(0,0.05,quartet_x.shape)

# cons = [{'type': 'eq', 'fun': t2_constraint}]
cons = []
for i in np.nonzero(quartet_neigh.ravel()!=-1)[0]:
        cons += [{'type': 'ineq', 'fun': voronoi_constraint,'args':(i,)}]
        # {'type': 'eq', 'fun': t2_constraint_abd},
        # {'type': 'eq', 'fun': t2_constraint_bcd}]



res = minimize(cost,quartet_x0, method='SLSQP', constraints=cons)
quartet_xfin = res.x.reshape(-1,2)

plt.scatter(quartet_x0[:,0],quartet_x0[:,1])
plt.plot(*graph_plot_x(adj,quartet_x0).T)
plt.scatter(quartet_xfin[:,0],quartet_xfin[:,1])
plt.plot(*graph_plot_x(adj,quartet_xfin).T)

plt.show()
from scipy.spatial import Voronoi, voronoi_plot_2d
fig = voronoi_plot_2d(Voronoi(quartet_xfin), show_vertices=False, line_colors='orange',
                line_width=2, line_alpha=0.6, point_size=2)
ax = plt.gca()
for i in range(quartet_x.shape[0]):
    ax.text(quartet_xfin[i,0],quartet_xfin[i,1],i)
plt.scatter(quartet_xfin[:,0],quartet_xfin[:,1],c=quartet_ctypes)
plt.show()


"""
Embed quartet within a full tissue. 
"""


tri = sim.t.mesh.tri
neigh = sim.t.mesh.neigh
ctypes = sim.t.c_types
cols = np.zeros((ctypes.size+1),dtype=np.int64)
cols[:-1] = ctypes + 1
k2s = sim.t.mesh.k2s
ntri = tri.shape[0]
nc = tri.max() + 1
L = sim.t.mesh.L
W = sim.t.W

choice = choose_random_quartet(ntri)
quartet_info = get_quartet(tri,neigh,k2s,*choice)
neigh_cids = get_quartet_neigh_cids(quartet_info,tri,neigh,k2s)
norm_neigh_cids = normalize_neigh_cids(neigh_cids)
quartet_tri = quartet_to_tri(norm_neigh_cids)
quartet_neigh = get_neighbours_quartet(quartet_tri)
quartet_x = sim.t.mesh.x[neigh_cids[neigh_cids!=-1]]
quartet_ctypes = ctypes[neigh_cids[neigh_cids!=-1]]
quartet_ctypes *=0