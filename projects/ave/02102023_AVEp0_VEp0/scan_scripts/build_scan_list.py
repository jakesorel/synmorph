import numpy as np
import pandas as pd
# import codecs, json
import pickle
import os
import fcntl
import sys
import os
sys.dont_write_bytecode = True

"""
For scan for 
differential adhesion
AVE p0
VE p0 
AVE v0
lambda_P
alpha (alignment)
over seeds. 
"""

N = 10
W01_range = np.logspace(-3,-1,N)
AVE_p0_range = np.linspace(3.4,5,N)
VE_p0_range = np.linspace(3.4,5,N)
AVE_v0_range = np.logspace(-3,-1,N)
lambda_P_range = np.logspace(-2,0,N)
alpha_range = np.logspace(-3,0,N)
seed_range = 2022 + np.arange(N,dtype=int)
print(N**7)

base_name = "17122022_W01_AVEp0_VEp0_v0_lambdaP_alpha"

df_save = pd.DataFrame()

if not os.path.exists("../scan_summary/temp"):
    os.mkdir("../scan_summary/temp")

if not os.path.exists("../scan_summary/%s_full_summary.csv"%base_name):
    with open("../scan_summary/%s_full_summary.csv"%base_name, "w+") as g:
        fcntl.flock(g, fcntl.LOCK_EX)
        g.write("counter,W01,AVE_p0,VE_p0,AVE_v0,lambda_P,alpha,seed,scan_dict_name\n")
        fcntl.flock(g, fcntl.LOCK_UN)
        # g.close()

if not os.path.exists("../scan_summary/%s_path_names.txt"%base_name):
    with open("../scan_summary/%s_path_names.txt"%base_name, "w+") as g:
        fcntl.flock(g, fcntl.LOCK_EX)
        # g.write("path_name\n")
        fcntl.flock(g, fcntl.LOCK_UN)


# for i1, W01 in enumerate(W01_range):
#     for i2, AVE_p0 in enumerate(AVE_p0_range):
#         for i3, VE_p0 in enumerate(VE_p0_range):
#             for i4, AVE_v0 in enumerate(AVE_v0_range):
#                 for i5,lambda_P in enumerate(lambda_P_range):
#                     for i6, alpha in enumerate(alpha_range):
#                         for j, seed in enumerate(seed_range):
j, k, Nper = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
i = j + k * Nper
[i4,i5,i6,j] = np.array(list(np.base_repr(i,N).zfill(4))).astype(int)


# counter = i
for i1, W01 in enumerate(W01_range):
    for i2, AVE_p0 in enumerate(AVE_p0_range):
        for i3, VE_p0 in enumerate(VE_p0_range):

            W01 = W01_range[i1]
            AVE_p0 = AVE_p0_range[i2]
            VE_p0 = VE_p0_range[i3]
            AVE_v0 = AVE_v0_range[i4]
            lambda_P = lambda_P_range[i5]
            alpha = alpha_range[i6]
            seed = seed_range[j]
            counter = np.flip([i1,i2,i3,i4,i5,i6,j])
            counter = (N**np.arange(len(counter))*counter).sum()

            scan_dict_name = base_name + "_" + "%s"%counter
            df_entry = np.array([counter,W01,AVE_p0,VE_p0,AVE_v0,lambda_P,alpha,seed,scan_dict_name])

            with open("../scan_summary/temp/%s_full_summary_%d.csv"%(base_name,i), "a+") as g:
                g.write(",".join(df_entry.astype(str)) + "\n")

            with open("../scan_summary/temp/%s_path_names_%d.txt"%(base_name,i), "a+") as g:
                g.write(scan_dict_name + "\n")

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

            scan_dict = {"tissue_params":tissue_params,"active_params":active_params,"init_params":init_params,"run_options":run_options,"simulation_params":simulation_params,"grn_params":grn_params,"save_options":save_options}


            pikd = open("../scan_dicts/%s"%scan_dict_name + ".pickle", 'wb')
            pickle.dump(scan_dict, pikd)
            pikd.close()
            print(counter)

