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


# sim_name = "23032022_W01_AVEp0_VEp0_980"


def run_analysis(sim_name):

    pikd = open("../scan_dicts/%s.pickle" % sim_name, 'rb')
    scan_dict = pickle.load(pikd)
    pikd.close()

    L = scan_dict["tissue_params"]["L"]
    sim_dict = load_hdf5_skeleton("../scan_results/%s_simulation.h5.gz"%sim_name,L)



    run_options = scan_dict["run_options"]

    x = np.array(sim_dict["x_save"],dtype=np.float32)
    t = np.arange(0,scan_dict["simulation_params"]["tfin"],scan_dict["simulation_params"]["dt"]*scan_dict["simulation_params"]["tskip"],dtype=np.float32)
    tri = np.array(sim_dict["tri_save"],dtype=np.int32)
    c_types = np.array(sim_dict["c_types"],dtype=np.int32)
    # x_unwrapped = sp.unwrap_positions(x, L)
    meshes = geo.mesh_assembler(x, tri, L, run_options)

    ctri_save = c_types[tri]
    hit_boundary = ((ctri_save == 0).any(axis=2)) * ((ctri_save == 3).any(axis=2))
    if hit_boundary.any():
        ti_hit_boundary = np.nonzero(hit_boundary)[0][0]
    else:
        ti_hit_boundary = -1
    with open("../analysis_results/t_hit_boundary.txt", "a+") as g:
        fcntl.flock(g, fcntl.LOCK_EX)
        g.write(sim_name + "," + str(ti_hit_boundary) + "\n")
        fcntl.flock(g, fcntl.LOCK_UN)

    def get_AVE_x(x, c_types, c_type=0):
        mask = c_types == c_type
        AVE_x = x[:, mask]
        return AVE_x

    def get_average_centroid(x_sample):
        return x_sample.mean(axis=1)

    def get_average_velocity(x_sample, t):
        dt = t[1] - t[0]
        v = (x_sample[1:] - x_sample[:-1]) / dt
        return v.mean(axis=1)

    def get_scalar_from_vector(v):
        return np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)

    AVE_x = get_AVE_x(x, c_types)
    av_AVE_x = get_average_centroid(AVE_x)
    av_AVE_d = get_scalar_from_vector(av_AVE_x - av_AVE_x[0])
    av_AVE_v = get_average_velocity(AVE_x, t)
    av_AVE_speed = get_scalar_from_vector(av_AVE_v)
    bulk_AVE_velocity = ((av_AVE_x - av_AVE_x[0])[1:] - (av_AVE_x - av_AVE_x[0])[:-1]) / (t[1] - t[0])
    bulk_AVE_speed = get_scalar_from_vector(bulk_AVE_velocity)

    ##Geometrical
    average_P = geo.apply_c_type_fn_to_meshes(geo.average_perimeter_by_cell_type, meshes, c_types)
    average_A = geo.apply_c_type_fn_to_meshes(geo.average_area_by_cell_type, meshes, c_types)
    average_p0 = geo.apply_c_type_fn_to_meshes(geo.average_shape_index_by_cell_type, meshes, c_types)

    # average_eccentricity = geo.apply_c_type_fn_to_meshes(geo.average_eccentricies_by_cell_type,meshes,c_types)

    def get_ave_connected_components(mesh):
        return top.count_connected_components(mesh.tri, c_types, len(mesh.x))[0]

    ave_connected_components = np.array(list(map(get_ave_connected_components, meshes)))

    def append_NaN(x):
        return np.concatenate((x, (np.nan,)))

    grn_params = scan_dict["grn_params"]
    AVE_p0 = np.repeat(grn_params["AVE_p0"], len(AVE_x[:, 0]))
    VE_p0 = np.repeat(grn_params["nonAVE_p0"], len(AVE_x[:, 0]))
    W01 = 0.0

    df = pd.DataFrame({"t": t, "PARAM_AVE_p0": AVE_p0, "PARAM_VE_p0": VE_p0,
                       "v_AVE_x": av_AVE_x[:, 0], "av_AVE_y": av_AVE_x[:, 1], "av_av_AVE_d": av_AVE_d,
                       "av_AVE_vx": append_NaN(av_AVE_v[:, 0]), "av_AVE_vy": append_NaN(av_AVE_v[:, 1]),
                       "av_AVE_speed": append_NaN(av_AVE_speed), "bulk_AVE_speed": append_NaN(bulk_AVE_speed),
                       "average_P_AVE": average_P[:, 0], "average_P_VE": average_P[:, 1],
                       "average_A_AVE": average_A[:, 0], "average_A_VE": average_A[:, 1],
                       "average_p0_AVE": average_p0[:, 0], "average_p0_VE": average_p0[:, 1],
                       # "average_ecc_AVE":average_eccentricity[:,0],"average_ecc_VE":average_eccentricity[:,1],
                       "AVE_connected_components": ave_connected_components})
    if not os.path.exists("../analysis_results"):
        os.mkdir("../analysis_results")
    if not os.path.exists("../analysis_results/all"):
        os.mkdir("../analysis_results/all")

    df.to_csv("../analysis_results/all/%s_analysis.csv" % (sim_name), index=None)




if __name__ == "__main__":
    if not os.path.exists("../analysis_results"):
        os.mkdir("../analysis_results")

    N = 20
    M = 100
    total_sims = N**2 * M
    sims_per_lot = 100
    slurm_index = int(sys.argv[1])
    range_to_sample = np.arange(slurm_index*sims_per_lot,(slurm_index+1)*sims_per_lot)

    def run_analysis_all(i):

        sim_name = "02102023_AVEp0_VEp0_%d" % i
        if not os.path.exists("../analysis_results/all/%s_analysis.csv"%sim_name):


            run_analysis(sim_name)
        else:
            print(sim_name,"exists")

    for i in range_to_sample:
        run_analysis_all(i)
    # Parallel(n_jobs=-1,backend="loky", prefer="threads")(delayed(run_analysis_all)(i) for i in range_to_sample)

