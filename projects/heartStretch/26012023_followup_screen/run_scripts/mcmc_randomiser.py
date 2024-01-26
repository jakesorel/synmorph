import sys
import os
import numpy as np
sys.dont_write_bytecode = True

SCRIPT_DIR = "../../../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pickle
import time
import synmorph as sm
import fcntl
import linecache
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError
from multiprocessing import cpu_count
import threading
import h5py, gzip
import synmorph as sm
import time
from synmorph.sim_plotting import plot_vor
import matplotlib.pyplot as plt
from numba import jit
import synmorph.tri_functions as trf

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

@jit(nopython=True)
def get_frac_boundary(is_notch,tri,trip1):
    notch_tri = trf.tri_call(is_notch.astype(np.float32),tri)
    notch_trip1 = trf.tri_call(is_notch.astype(np.float32),trip1)
    return (notch_tri * (1-notch_trip1)).sum()/notch_tri.sum()

@jit(nopython=True)
def random_swap(is_notch,n_c):
    is_notch_new = is_notch.copy()
    i,j = int(np.random.random()*n_c), int(np.random.random()*n_c)
    is_notch_new[i] = is_notch[j]
    is_notch_new[j] = is_notch[i]
    return is_notch_new

@jit(nopython=True)
def perform_swap(E,is_notch,n_c,T,tri,trip1):
    is_notch_new = random_swap(is_notch,n_c)
    E_new = get_frac_boundary(is_notch_new,tri,trip1)
    p_transition = np.exp(-(E_new-E)/T)
    p = np.random.random()
    if p < p_transition:
    # if direc*E_new > direc*E:
        is_notch = is_notch_new.copy()
        E = E_new
    return E,is_notch



if __name__ == "__main__":
    base_name = "26012023_followup_screen"

    N_set,N_notch,N_seed,N_kappa,N_kappaA = 10,10,10,10,10

    total_sims = N_notch * N_seed * N_kappa*(N_kappa+1)/2 * N_kappaA
    pnotch_range = np.linspace(0, 1, N_notch)
    seed_range = np.arange(10)
    init_space = np.meshgrid(pnotch_range,seed_range,indexing="ij")
    try:
        mkdir("../scan_initialisation")
    except:
        print("assuming exists")


    def run_init(seed_index):

        ##Set up the initial condition.
        n_cell = 250


        tissue_params = {"L": 1.0,
                         "A0": 1/n_cell,
                         "P0": 3.81,
                         "kappa_A": 1,
                         "kappa_P": 0.1,
                         "W": np.array(((0, 0.0), (0.0, 0))).astype(np.float32),
                         "a": 0.,
                         "k": 0.}
        active_params = {"v0": 0.,
                         "Dr": 1e-1}
        init_params = {"init_noise": 0.2,
                       "c_type_proportions": (1.0, 0.0)}
        run_options = {"equiangulate": True,
                       "equi_nkill": 10}
        simulation_params = {"dt": 0.25,
                             "tfin": 500,
                             "tskip": 10,
                             "dt_grn": 0.05,
                             "grn_sim": "heart_stretch"}
        save_options = {"save": "skeleton",
                        "result_dir": "results",
                        "name": "heart_stretch",
                        "compressed": True}
        grn_params = {"p_notch": 0.5,
                      "L_init": np.sqrt(n_cell),
                      "L_min": 1.0,
                      "kappa_A_P": 0.1,
                      "kappa_A_N": 0.1,
                      "kappa_P_P": 0.1,
                      "kappa_P_N": 0.1,
                      "A0_P": 1.0,
                      "A0_N": 1.0,
                      "P0_P": 3.75,
                      "P0_N": 3.75,
                      "init_pressure": 0.0,
                      "fin_pressure": 0.0,
                      "pressure_start_time": 30.0,
                      "pressure_slope": 0.2,
                      "mu_L": 0.01,
                      "n_t": int(simulation_params["tfin"] / simulation_params["dt"]),
                      "notch_distribution":"random",
                      "osc_level":0,
                      "heart_period":80}
        sim = sm.simulation(tissue_params=tissue_params,
                            active_params=active_params,
                            init_params=init_params,
                            simulation_params=simulation_params,
                            run_options=run_options,
                            save_options=save_options,
                            grn_params=grn_params)

        sim.simulate()

        is_notch_set_list = []
        for i1, p_notch in enumerate(pnotch_range):

            n_c = sim.t.mesh.n_c
            tri = sim.t.mesh.tri
            trip1 = np.roll(tri ,1 ,axis=1)


            is_notch = np.zeros(n_c, dtype=bool)
            is_notch[:int(np.round(p_notch * n_c))] = True
            np.random.shuffle(is_notch)

            if (p_notch > 0)*(p_notch < 1):


                E0 = get_frac_boundary(is_notch,tri,trip1)
                E = E0
                n_iter = 100000
                E_save_min = np.zeros(n_iter)
                is_notchs_min = np.zeros((n_iter,len(is_notch)),dtype=bool)
                for i in range(n_iter):
                    E,is_notch = perform_swap(E,is_notch,n_c,0.001,tri,trip1)
                    E_save_min[i] = E
                    is_notchs_min[i] = is_notch.copy()

                E0 = get_frac_boundary(is_notch,tri,trip1)
                E = E0
                n_iter = 100000
                E_save_max = np.zeros(n_iter)
                is_notchs_max = np.zeros((n_iter,len(is_notch)),dtype=bool)
                for i in range(n_iter):
                    E,is_notch = perform_swap(E,is_notch,n_c,-0.001,tri,trip1)
                    E_save_max[i] = E
                    is_notchs_max[i] = is_notch.copy()

                E_sets = np.linspace(E_save_max.min(),E_save_max.max(),N_set)
                is_notch_sets = np.zeros((N_set,len(is_notch)),dtype=bool)
                for i, E in enumerate(E_sets):
                    dE = (E_save_max-E)**2
                    idx = np.nonzero(dE == dE.min())[0][0]
                    is_notch_sets[i] = is_notchs_max[idx]
                is_notch_set_list.append(is_notch_sets)
            else:
                is_notch_sets = np.zeros((N_set,len(is_notch)),dtype=bool)
                for i in range(N_set):
                    is_notch_sets[i] = is_notch.copy()
                is_notch_set_list.append(is_notch_sets)

        is_notch_set_list = np.array(is_notch_set_list)
        np.savez("../scan_initialisation/x_%d.npz" % (seed_index),sim.x_save[-1])
        np.savez("../scan_initialisation/is_notch_%d.npz" % (seed_index),is_notch_set_list)


    Parallel(n_jobs=-1, backend="loky", prefer="threads")(delayed(run_init)(i) for i in range(N_seed))
