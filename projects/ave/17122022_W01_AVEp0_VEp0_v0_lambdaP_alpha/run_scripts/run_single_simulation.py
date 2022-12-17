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

    sim.simulate(progress_bar=False)


if __name__ == "__main__":

    base_name = "17122022_W01_AVEp0_VEp0_v0_lambdaP_alpha"

    if not os.path.exists("../scan_summary/%s_full_summary.csv" % base_name):
        with open("../scan_summary/%s_full_summary.csv" % base_name, "w+") as g:
            fcntl.flock(g, fcntl.LOCK_EX)
            g.write("counter,W01,AVE_p0,VE_p0,AVE_v0,lambda_P,alpha,seed,scan_dict_name\n")
            fcntl.flock(g, fcntl.LOCK_UN)
            # g.close()

    if not os.path.exists("../scan_summary/%s_path_names.txt" % base_name):
        with open("../scan_summary/%s_path_names.txt" % base_name, "w+") as g:
            fcntl.flock(g, fcntl.LOCK_EX)
            # g.write("path_name\n")
            fcntl.flock(g, fcntl.LOCK_UN)


    j, k, Nper = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    i = j + k * Nper
    [i1,i2,i3,i4, i5, i6, j] = np.array(list(np.base_repr(i, 10).zfill(7))).astype(int)
    N = 10
    W01_range = np.logspace(-3, -1, N)
    AVE_p0_range = np.linspace(3.4, 5, N)
    VE_p0_range = np.linspace(3.4, 5, N)
    AVE_v0_range = np.logspace(-3, -1, N)
    lambda_P_range = np.logspace(-2, 0, N)
    alpha_range = np.logspace(-3, 0, N)
    seed_range = 2022 + np.arange(N, dtype=int)
    W01 = W01_range[i1]
    AVE_p0 = AVE_p0_range[i2]
    VE_p0 = VE_p0_range[i3]
    AVE_v0 = AVE_v0_range[i4]
    lambda_P = lambda_P_range[i5]
    alpha = alpha_range[i6]
    seed = seed_range[j]
    counter = np.flip([i1, i2, i3, i4, i5, i6, j])
    counter = (N ** np.arange(len(counter)) * counter).sum()

    scan_dict_name = base_name + "_" + "%s" % counter
    df_entry = np.array([counter, W01, AVE_p0, VE_p0, AVE_v0, lambda_P, alpha, seed, scan_dict_name])

    with open("../scan_summary/%s_full_summary.csv" % (base_name), "a+") as g:
        fcntl.flock(g, fcntl.LOCK_EX)
        g.write(",".join(df_entry.astype(str)) + "\n")
        fcntl.flock(g, fcntl.LOCK_UN)

    with open("../scan_summary/%s_path_names.txt" % (base_name), "a+") as g:
        fcntl.flock(g, fcntl.LOCK_EX)
        g.write(scan_dict_name + "\n")
        fcntl.flock(g, fcntl.LOCK_UN)


    tissue_params = {"L": 15,
                     "A0": 1,
                     "P0": 3.6,
                     "kappa_A": 1,
                     "kappa_P": lambda_P,
                     "W": np.array(((0.0, W01, W01, 0.1), (W01, 0, 0, 0.1), (W01, 0, 0, 0.1),
                                    (0.1, 0.1, 0.1, 0.1))) * 1,
                     "a": 0,
                     "k": 0}
    active_params = {"v0": 2e-1,
                     "Dr": 5e-3}
    init_params = {"init_noise": 0.1,
                   "c_type_proportions": (1.0, 0)}
    run_options = {"equiangulate": True,
                   "equi_nkill": 10}
    simulation_params = {"dt": 0.01,
                         "tfin": 300,
                         "tskip": 10,
                         "dt_grn": 0.025,
                         "grn_sim": "grn_ave_couple_orientation",
                         "tinit": 10,
                         "random_seed": int(seed)}
    grn_params = {"n_AVE_cells": 20,
                  "AVE_alpha_dir": alpha,
                  "non_AVE_alpha_dir": 0,
                  "AVE_v0": AVE_v0,
                  "non_AVE_v0": 0,
                  "AVE_alpha0": -np.pi / 2,
                  "boundary_frac": 0.08,
                  "AVE_A0": 0.54,
                  "exe_frac": 0.0,
                  "AVE_p0": AVE_p0,
                  "nonAVE_p0": VE_p0}
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
    print(counter)

    path_name = "../scan_dicts/%s" % scan_dict_name + ".pickle"

    t0 = time.time()
    run_simulation(path_name)
    t1 = time.time()
    print(t1-t0)
    out_file = open("../scan_summary/17122022_W01_AVEp0_VEp0_v0_lambdaP_alpha_result_log.txt","a")
    out_file.write("%s_%.2f"%(path_name,(t1-t0)) + "\n")
    out_file.close()
