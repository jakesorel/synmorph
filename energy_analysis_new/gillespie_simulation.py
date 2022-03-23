import numpy as np
import synmorph as sm
from functools import reduce
from numba import jit
from scipy import sparse
import synmorph.tri_functions as trf
import time
from energy_analysis.quartets_from_mesh import *
import json

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

tri = sim.t.mesh.tri
neigh = sim.t.mesh.neigh
ctypes = sim.t.c_types
cols = np.zeros((ctypes.size+1),dtype=np.int64)
cols[:-1] = ctypes + 1
k2s = sim.t.mesh.k2s
ntri = tri.shape[0]
nc = tri.max() + 1

with open('energy_analysis_new/hash_energies.json') as json_file:
    energy_dict = json.load(json_file)

energy_dict_keys = list(energy_dict.keys())
for key in energy_dict_keys:
    energy_dict[int(key)] = energy_dict.pop(key)


@jit(nopython=True)
def get_all_hashes(tri,neigh,k2s,ntri,cols):
    tri0_is = np.repeat(np.arange(ntri),3)
    tri0_js = np.zeros(ntri*3,dtype=np.int64)
    tri0_js[1::3] = 1
    tri0_js[2::3] = 2
    mask = neigh.ravel()>tri0_is #ignore duplicates.
    tri0_is,tri0_js = tri0_is[mask],tri0_js[mask]
    mesh_hashes = np.zeros(len(tri0_is),dtype=np.int64)
    for i, (tri0_i,tri0_j) in enumerate(zip(tri0_is,tri0_js)):
        quartet_info = get_quartet(tri, neigh, k2s, tri0_i,tri0_j)
        neigh_cids = get_quartet_neigh_cids(quartet_info, tri, neigh, k2s)
        state = get_quartet_state(neigh_cids, cols)
        quartet_hash = get_cannonical_hash(state)
        mesh_hashes[i] = quartet_hash
    return mesh_hashes,tri0_is,tri0_js

@jit(nopython=True)
def get_Ea(energies,beta,p0=3.9,r=5):
    W01 = beta*p0*2/r
    a = energies[:,0] + energies[:,2]*W01
    b = energies[:,1] + energies[:,3]*W01
    return b**2/(4*a)

@jit(nopython=True)
def get_rates(Ea,T):
    return np.exp(-Ea/T)

@jit(nopython=True)
def hash_choice(rates,idx):
    scaled = np.concatenate((np.array((0,)),np.cumsum(rates) / np.sum(rates)))
    p = np.random.random()
    return ((scaled[1:]>p)*(scaled[:-1]<p)*idx).sum()



energy_dict_keys = np.array(list(energy_dict.keys()))


@jit(nopython=True)
def make_choice_given_energies(energies,mesh_hashes,tri0_is,tri0_js,beta=0.1,T=0.1):

    idx = np.arange(len(mesh_hashes))
    mask = ~np.isnan(energies[:,0])
    idx= idx[mask]
    energies = energies[mask]



    rates = get_rates(get_Ea(energies,beta),T)

    mean_wait = 1/np.sum(rates)
    del_t = np.random.exponential(mean_wait)


    chosen_idx = hash_choice(rates,idx)
    tri0_i,tri0_j = tri0_is.take(chosen_idx),tri0_js.take(chosen_idx)
    chosen_hash = mesh_hashes.take(chosen_idx)
    return chosen_hash,tri0_i,tri0_j,del_t

@jit(nopython=True)
def get_hash_mask(mesh_hashes,energy_dict_keys):
    return np.sum(mesh_hashes == np.expand_dims(energy_dict_keys, 1), 0).astype(np.bool_)

@jit(nopython=True)
def apply_hash_mask(energy_dict_keys,mesh_hashes,tri0_is,tri0_js):
    hash_mask = get_hash_mask(mesh_hashes,energy_dict_keys)
    return mesh_hashes[hash_mask],tri0_is[hash_mask],tri0_js[hash_mask]


def gillespie_choice(tri,neigh,k2s,ntri,cols,beta=0.1,T=0.1):
    mesh_hashes,tri0_is,tri0_js = get_all_hashes(tri,neigh,k2s,ntri,cols)
    # energies = np.array(list(map(lambda x: energy_dict[x], mesh_hashes)))

    energies = np.ones((len(mesh_hashes),4))*np.nan
    for i, hsh in enumerate(mesh_hashes):
        try:
            energies[i] = energy_dict[hsh]
        except:
            energies[i] = np.nan
    return make_choice_given_energies(energies,mesh_hashes,tri0_is,tri0_js,beta,T)


def perform_gillespie_choice(tri,neigh,k2s,ntri,cols,beta=0.1,T=0.1):
    chosen_hash,tri0_i,tri0_j,del_t = gillespie_choice(tri,neigh,k2s,ntri,cols,beta,T)
    quartet_info = get_quartet(tri, neigh, k2s, tri0_i, tri0_j)
    tri_new, neigh_new, k2s_new = update_mesh(quartet_info, tri, neigh, k2s)
    return chosen_hash,del_t,tri_new,neigh_new,k2s_new

def run_gillespie(tri0,neigh0,k2s0,ntri,cols,beta,T,n_iter=1000):
    tri,neigh,k2s = tri0.copy(),neigh0.copy(),k2s0.copy()
    hash_save = np.zeros(n_iter,dtype=np.int64)
    t_span = np.zeros(n_iter+1)
    n_cc = np.zeros((n_iter+1,2))
    (num_in_cluster0, num_in_cluster1), lookup_new = tri_num_connected(tri, ctypes, nc)
    n_cc[0] = num_in_cluster0.size,num_in_cluster1.size
    t = 0
    t0 = time.time()
    for i in range(n_iter):
        chosen_hash,del_t, tri,neigh,k2s = perform_gillespie_choice(tri, neigh, k2s, ntri, cols, beta, T)
        t += del_t
        hash_save[i] = chosen_hash
        t_span[i+1] = t
        (num_in_cluster0, num_in_cluster1), lookup_new = tri_num_connected(tri, ctypes, nc)
        n_cc[i+1] = num_in_cluster0.size,num_in_cluster1.size
    t1= time.time()
    print(t1-t0)
    return hash_save,t_span,n_cc

def regularise_time_series(_t_spans,dt=0.1):
    t_spans = np.array(_t_spans)
    max_t = t_spans[:,-1].min()
    t_span_regular = np.arange(0,max_t+dt,dt)
    val_indices = np.zeros((t_span_regular.size,len(t_spans)),dtype=np.int64)
    for i, (tl,tu) in enumerate(zip(t_span_regular[:-1],t_span_regular[1:])):
        val_indices[i+1:] = np.argmax((t_spans > tl)*1,axis=1)
    return t_span_regular,val_indices

tri0,neigh0,k2s0 = tri.copy(),neigh.copy(),k2s.copy()

hash_save,t_span,n_cc = run_gillespie(tri.copy(),neigh.copy(),k2s.copy(),ntri,cols,0.1,0.1,n_iter=1000)
plt.plot(t_span,n_cc[:,0])
plt.show()

t_spans, n_ccs = [],[]
n_rep = 5
for i in range(n_rep):
    hash_save, t_span, n_cc = run_gillespie(tri.copy(), neigh.copy(), k2s.copy(), ntri, cols, 0.05, 0.075, n_iter=2000)
    t_spans.append(t_span)
    n_ccs.append(n_cc)

for (t_span,n_cc) in zip(t_spans,n_ccs):
    plt.plot(t_span,n_cc.sum(axis=1))
plt.show()

sorted_bool = np.array([n_cc.sum(axis=1)==2 for n_cc in n_ccs])
t_span_regular,val_indices = regularise_time_series(t_spans,dt=5)
sorted_bool_regular = np.array([sb.take(vi) for (sb,vi) in zip(sorted_bool,val_indices.T)])
sorted_bool_regular = sorted_bool.ravel().take(val_indices.ravel()).reshape(val_indices.shape)


plt.plot(sorted_bool_regular.mean(axis=1))
plt.show()


"""
This is clearly not working... 
Systems often sort out of a sorted config. 
Further, it's pretty slow.

Some things to check: 
-- Energy fits are OK. Checked. 
-- Energy landscapes look reasonable. Checked. 
-- Choice of Ea calculation is robust. Checked. 


The energy itself may be a bit suspect. 
Many topologies have very similar energy values. Intuitively, this means that quartet transitions do not discriminate neighbourhood. Consequently, there's no asymmetry in the T1, meaning sorted configs aren't favoured. This is my theory anyway. 

So to solve this, we are back to the same problem of optimising over many hashes (45k). 

Is there any way this can be done efficiently? 

One possible option
-- Pinning the positions of the central quartet, for a given heterotypic neighbourhood, optimise positions of the ghost cells for a given beta. 
-->> Would need to ignore the symmetry thing used before. 
--->> May need to optimise for central cell position too. This could be done in a constrained manner with symmetry arguments. 
-- Then run the optimisation as before, putting all vertices on springs. 
-- Would then need to interpolate to get E as a function of beta. 
-- Need to check whether this is stable.... 

"""
