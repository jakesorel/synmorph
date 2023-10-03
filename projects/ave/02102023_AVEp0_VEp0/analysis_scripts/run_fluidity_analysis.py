import os
import sys


SCRIPT_DIR = "../../../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

import json
import numpy as np
from synmorph.analysis import geometrical as geo
from synmorph.analysis import topological as top
from synmorph.analysis import spatial as sp
from synmorph.utils import *
import pickle
import pandas as pd
import fcntl
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError
from multiprocessing import cpu_count
import threading
import synmorph.analysis.spatial as spatial
from scipy import sparse



def get_edges_from_tri(tri):
    return np.row_stack([np.column_stack((tri[:,i],tri[:,(i+1)%3])) for i in range(3)])

def calculate_t1s_frame_to_frame(edges0,edges1):
    if np.any(edges0!=edges1):
        adj_mat0 = sparse.coo_matrix((np.ones_like(edges0[:,0],dtype=bool),(edges0[:,0],edges0[:,1])),shape=(np.max(edges0)+1,np.max(edges0)+1))
        adj_mat1 = sparse.coo_matrix((np.ones_like(edges1[:,0],dtype=bool),(edges1[:,0],edges1[:,1])),shape=(np.max(edges0)+1,np.max(edges0)+1))
        n_t1s = int((adj_mat0!=adj_mat1).sum()/4)
    else:
        n_t1s = 0

    return n_t1s

def calculate_t1s_frame_to_frame_by_ctype(edges0,edges1,c_types):
    n_ctypes = np.max(c_types)+1
    n_t1s = np.zeros((n_ctypes,n_ctypes),dtype=np.int32)
    if np.any(edges0!=edges1):
        adj_mat0 = sparse.coo_matrix((np.ones_like(edges0[:,0],dtype=bool),(edges0[:,0],edges0[:,1])),shape=(np.max(edges0)+1,np.max(edges0)+1))
        adj_mat1 = sparse.coo_matrix((np.ones_like(edges1[:,0],dtype=bool),(edges1[:,0],edges1[:,1])),shape=(np.max(edges0)+1,np.max(edges0)+1))

        t1_mat = (adj_mat0!=adj_mat1)
        for i in range(n_ctypes):
            for j in range(n_ctypes):
                n_t1s[i,j] = int(t1_mat[c_types == i].T[c_types == j].sum()/2) ##NOTE THE DENOMINATOR IS DIFFERENT. This is because it is measuring how many pairs of cells gain or lose contact.
    return n_t1s

def calculate_t1s_across_sim(tri_save):
    edges_save = np.array(list(map(get_edges_from_tri,tri_save)))
    edges0s = edges_save[:-1]
    edges1s = edges_save[1:]

    t1s_by_frame = np.array([calculate_t1s_frame_to_frame(edges0,edges1) for (edges0,edges1) in zip(edges0s,edges1s)])
    return t1s_by_frame


def calculate_t1s_across_sim_by_ctype(tri_save,c_types):
    edges_save = np.array(list(map(get_edges_from_tri,tri_save)))
    edges0s = edges_save[:-1]
    edges1s = edges_save[1:]

    def _calculate_t1s_frame_to_frame_by_ctype(edges0,edges1):
        return calculate_t1s_frame_to_frame_by_ctype(edges0, edges1, c_types)

    t1s_by_frame = np.array([_calculate_t1s_frame_to_frame_by_ctype(edges0,edges1) for (edges0,edges1) in zip(edges0s,edges1s)])
    return t1s_by_frame


def diffusion_coefficient_by_ctype(x_save,t_span_save,c_types,L):
    return spatial.diffusion_coefficient_by_ctype(x_save,t_span_save,c_types,L)



def run(sim_name):

    pikd = open("../scan_dicts/%s.pickle" % sim_name, 'rb')
    scan_dict = pickle.load(pikd)
    pikd.close()

    L = scan_dict["tissue_params"]["L"]
    sim_dict = load_hdf5_skeleton("../scan_results/%s_simulation.h5.gz"%sim_name,L)


    x_save = np.array(sim_dict["x_save"],dtype=np.float32)
    t_span_save = np.arange(0,scan_dict["simulation_params"]["tfin"],scan_dict["simulation_params"]["dt"]*scan_dict["simulation_params"]["tskip"],dtype=np.float32)
    tri_save = np.array(sim_dict["tri_save"],dtype=np.int32)
    c_types = np.array(sim_dict["c_types"],dtype=np.int32)
    if not os.path.exists("../analysis_results"):
        os.mkdir("../analysis_results")
    if not os.path.exists("../analysis_results/fluidity"):
        os.mkdir("../analysis_results/fluidity")

    t1s_ctype_rate = (calculate_t1s_across_sim_by_ctype(tri_save, c_types).sum(axis=0)/np.max(t_span_save)).ravel()
    pair_labels = ["-".join(val) for val in np.mgrid[:len(t1s_ctype_rate),:len(t1s_ctype_rate)].reshape(2,-1).T.astype(str)]

    t1s_rate = (calculate_t1s_across_sim(tri_save).sum()/np.max(t_span_save))
    D_ctype = spatial.diffusion_coefficient_by_ctype(x_save, t_span_save, c_types, L)
    D = spatial.diffusion_coefficient(x_save, t_span_save, L)

    idx = int(sim_name.split("_")[-1])

    file = open("../analysis_results/fluidity/%s.csv"%sim_name,"w+")
    file.write(str(idx)+","+str(D)+","+",".join(D_ctype.astype(str))+","+str(t1s_rate)+","+",".join(t1s_ctype_rate.astype(str)) + "\n")
    file.close()


if __name__ == "__main__":
    if not os.path.exists("../analysis_results"):
        os.mkdir("../analysis_results")

    N = 20
    M = 100
    total_sims = N**2 * M
    sims_per_lot = M
    slurm_index = int(sys.argv[1])
    range_to_sample = np.arange(slurm_index*sims_per_lot,(slurm_index+1)*sims_per_lot)

    def run_analysis_all(i):
        sim_name = "02102023_AVEp0_VEp0_%d" % i

        run(sim_name)

    Parallel(n_jobs=-1,backend="loky", prefer="threads")(delayed(run_analysis_all)(i) for i in range_to_sample)

