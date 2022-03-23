import numpy as np
import pandas as pd
# import codecs, json
import pickle
"""
For scan for 
differential adhesion
AVE p0
VE p0 
"""
N = 2
N_rep = 2
W01_range = np.logspace(-3,-1,N)
AVE_p0_range = np.linspace(3.2,4.2,N)
VE_p0_range = np.linspace(3.2,4.2,N)
seed_range = 2022 + np.arange(N_rep,dtype=int)

base_name = "23032022_W01_AVEp0_VEp0"

df_save = pd.DataFrame()

counter = 0
for i, W01 in enumerate(W01_range):
    for j, AVE_p0 in enumerate(AVE_p0_range):
        for k, VE_p0 in enumerate(VE_p0_range):
            for l, seed in enumerate(seed_range):
                scan_dict_name = base_name + "_" + "%s"%counter
                df_entry = pd.DataFrame({"sim_id":counter,"W01":W01,"AVE_p0":AVE_p0,"VE_p0":VE_p0,"seed":seed,"scan_dict":scan_dict_name}, index=[0])
                df_save = pd.concat([df_save,df_entry])

                tissue_params = {"L": 20,
                                 "A0": 1,
                                 "P0": 3.6,
                                 "kappa_A": 1,
                                 "kappa_P": 0.1,
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
                simulation_params = {"dt": 0.05,
                                     "tfin": 300,
                                     "tskip": 10,
                                     "dt_grn": 0.025,
                                     "grn_sim": "grn_ave_couple_orientation",
                                     "tinit": 10,
                                     "random_seed": int(seed)}
                grn_params = {"n_AVE_cells": 20,
                              "AVE_alpha_dir": 0.15,
                              "non_AVE_alpha_dir": 0,
                              "AVE_v0": 1e-1,
                              "non_AVE_v0": 0,
                              "AVE_alpha0": -np.pi / 2,
                              "boundary_frac": 0.08,
                              "AVE_A0": 0.7,
                              "exe_frac": 0.0,
                              "AVE_p0": AVE_p0,
                              "nonAVE_p0": VE_p0}
                save_options = {"save": "skeleton",
                                "result_dir": "../scan_results",
                                "name": scan_dict_name,
                                "compressed": True}

                scan_dict = {"tissue_params":tissue_params,"active_params":active_params,"init_params":init_params,"run_options":run_options,"simulation_params":simulation_params,"grn_params":grn_params,"save_options":save_options}


                pikd = open("../scan_dicts/%s"%scan_dict_name + ".pickle", 'wb')
                pickle.dump(scan_dict, pikd)
                pikd.close()

                counter += 1


df_save.to_csv("../scan_summary/%s_full_summary.csv"%base_name)

path_names = open("../scan_summary/%s_path_names.txt"%base_name,"w+")
for nm in df_save["scan_dict"]:
    path_names.write("scan_dicts/%s.pickle"%nm + "\n")
path_names.close()
