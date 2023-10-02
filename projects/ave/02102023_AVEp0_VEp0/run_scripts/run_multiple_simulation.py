import sys
import os
import numpy as np
sys.dont_write_bytecode = True

SCRIPT_DIR = "../../../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pickle
import time
import synmorph as sm

from synmorph.analysis import geometrical as geo
from synmorph.analysis import topological as top
from synmorph.utils import *
import pickle
import pandas as pd
import fcntl
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError
import threading

try:
    import thread
except ImportError:
    import _thread as thread


def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt

def exit_after(s):
    '''
    use as decorator to exit process if
    function takes longer than s seconds
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer


def run_simulation(path_name):
    pikd = open(path_name, 'rb')
    scan_dict = pickle.load(pikd)
    pikd.close()

    print(scan_dict)
    sim = sm.simulation(tissue_params=scan_dict["tissue_params"],
                        active_params=scan_dict["active_params"],
                        init_params=scan_dict["init_params"],
                        simulation_params=scan_dict["simulation_params"],
                        grn_params=scan_dict["grn_params"],
                        run_options=scan_dict["run_options"],
                        save_options=scan_dict["save_options"])

    sim.simulate(progress_bar=True)
    return sim


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
    base_name = "02102023_AVEp0_VEp0"

    if not os.path.exists("../scan_summary/%s_full_summary.csv" % base_name):
        with open("../scan_summary/%s_full_summary.csv" % base_name, "w+") as g:
            fcntl.flock(g, fcntl.LOCK_EX)
            g.write("counter,W01,AVE_p0,VE_p0,seed,scan_dict_name\n")
            fcntl.flock(g, fcntl.LOCK_UN)
            # g.close()

    if not os.path.exists("../scan_summary/%s_path_names.txt" % base_name):
        with open("../scan_summary/%s_path_names.txt" % base_name, "w+") as g:
            fcntl.flock(g, fcntl.LOCK_EX)
            # g.write("path_name\n")
            fcntl.flock(g, fcntl.LOCK_UN)

    N = 20
    M = 100
    total_sims = N**2 * M
    sims_per_lot = 100
    slurm_index = int(sys.argv[1])
    print("Slurm index", slurm_index)
    range_to_sample = np.arange(slurm_index*sims_per_lot,(slurm_index+1)*sims_per_lot)

    def run_analysis_i(i):
        run_analysis("02102023_AVEp0_VEp0_%d" % i)


    def run_job(i,equiangulate=True):
        t_0 = time.time()
        if not os.path.exists("../scan_results/02102023_AVEp0_VEp0_%d_simulation.h5.gz"%i):
            print("Simulating %d" % i)
            [i1, i2, j] = np.unravel_index(i, (N, N, M))

            AVE_p0_range = np.linspace(3.4, 4.5, N)
            VE_p0_range = np.linspace(3.4, 4.5, N)
            seed_range = 2023 + np.arange(M, dtype=int)
            W01 = 0.0
            AVE_p0 = AVE_p0_range[i1]
            VE_p0 = VE_p0_range[i2]
            AVE_v0 = 0.05
            lambda_P = 0.2
            seed = seed_range[j]

            scan_dict_name = base_name + "_" + "%s" % i
            df_entry = np.array([i, W01, AVE_p0, VE_p0, seed, scan_dict_name])

            with open("../scan_summary/%s_full_summary.csv" % (base_name), "a+") as g:
                fcntl.flock(g, fcntl.LOCK_EX)
                g.write(",".join(df_entry.astype(str)) + "\n")
                fcntl.flock(g, fcntl.LOCK_UN)

            with open("../scan_summary/%s_path_names.txt" % (base_name), "a+") as g:
                fcntl.flock(g, fcntl.LOCK_EX)
                g.write(scan_dict_name + "\n")
                fcntl.flock(g, fcntl.LOCK_UN)

            tissue_params = {"L": 17.0,
                             "A0": 1.,
                             "P0": 3.4,
                             "kappa_A": 1.,
                             "kappa_P": lambda_P,
                             "W": (np.array(((0.0, W01, W01, 0.1), (W01, 0, 0, 0.5), (W01, 0, 0, 0.5),
                                             (0.1, 0.5, 0.5, 0.1))) * 1).astype(np.float32),
                             "a": 0.,
                             "k": 0.}
            active_params = {"v0": 2e-1,
                             "Dr": 5e-3}
            init_params = {"init_noise": 0.1,
                           "c_type_proportions": (1.0, 0)}
            run_options = {"equiangulate": equiangulate,
                           "equi_nkill": 10}
            simulation_params = {"dt": 0.1,
                                 "tfin": 300,
                                 "tskip": 20,
                                 "dt_grn": 0.05,
                                 "grn_sim": "grn_ave_couple_orientation",
                                 "tinit": 10,
                                 "random_seed": int(seed)}
            grn_params = {"n_AVE_cells": 20,
                          "AVE_alpha_dir": 0.2,
                          "non_AVE_alpha_dir": 0,
                          "AVE_v0": 0.05,
                          "non_AVE_v0": 0,
                          "AVE_alpha0": np.pi / 2,
                          "boundary_frac": 0.08,
                          "AVE_A0": 0.54,
                          "exe_frac": 0.0,
                          "AVE_p0": AVE_p0,
                          "nonAVE_p0": VE_p0,
                          "ExEVE_p0": 4.0}
            save_options = {"save": "hdf5",
                            "result_dir": "../scan_results",
                            "name": scan_dict_name,
                            "compressed": True}

            scan_dict = {"tissue_params": tissue_params, "active_params": active_params, "init_params": init_params,
                         "run_options": run_options, "simulation_params": simulation_params, "grn_params": grn_params,
                         "save_options": save_options}

            pikd = open("../scan_dicts/%s" % scan_dict_name + ".pickle", 'wb')
            pickle.dump(scan_dict, pikd)
            pikd.close()
            print(i)

            path_name = "../scan_dicts/%s" % scan_dict_name + ".pickle"

            t0 = time.time()
            run_simulation(path_name)
            t1 = time.time()
            print(t1 - t0)
            out_file = open("../scan_summary/02102023_AVEp0_VEp0_result_log.txt", "a")
            out_file.write("%s_%.2f" % (path_name, (t1 - t0)) + "\n")
            out_file.close()
            t_1 = time.time()
            print("Simulation completed in ", np.round(t_1-t_0),"s")
        else:
            print("Simulation %d exists, skipping"%i)

    # @exit_after(500)
    # def run_job_timed(i):
    #     return run_job(i,True)
    #
    # @exit_after(1800)
    # def run_job_timed_no_equiangulate(i):
    #     return run_job(i,False)
    #
    #
    # t_tot_0 = time.time()
    # try:
    #     Parallel(n_jobs=-1,backend="loky", prefer="threads")(delayed(run_job)(i,True) for i in range_to_sample)
    # except:
    #     try:
    #         Parallel(n_jobs=8,backend="loky", prefer="threads")(delayed(run_job)(i,True) for i in range_to_sample)
    #     except:
    #         try:
    #             Parallel(n_jobs=8,backend="loky", prefer="threads")(delayed(run_job)(i,False) for i in range_to_sample)
    #         except:
    #             Parallel(n_jobs=4,backend="loky", prefer="threads")(delayed(run_job)(i,False) for i in range_to_sample)
    #
    # Parallel(n_jobs=-1,backend="loky", prefer="threads")(delayed(run_analysis_i)(i) for i in range_to_sample)
    #


    for i in range_to_sample:
        run_job(i,equiangulate=True)
        # try:
        #     run_job_timed(i)
        # except:
        #     print("Equiangulation timed out")
        #     try:
        #         run_job_timed_no_equiangulate(i)
        #     except:
        #         print("Forced triangulation timed out too.. giving up")
    Parallel(n_jobs=-1,backend="loky", prefer="threads")(delayed(run_analysis_i)(i) for i in range_to_sample)


    t_tot_1 = time.time()
    print("400 simulations completed in ",t_tot_0-t_tot_0,"s")


"""
Note, may need to do some tidying up of the dataset. 

Need to deal with instances where the code crashed mid export or mid zip. 

mid export --> .h5 present, but .gz not present. Here, need to repeat the simulation
mid zip --> .h5 and .gz both present. Here, need to repeat the zipping. 
"""