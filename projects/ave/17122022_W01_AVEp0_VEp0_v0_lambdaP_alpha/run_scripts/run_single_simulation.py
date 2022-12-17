import sys
import os
sys.dont_write_bytecode = True

SCRIPT_DIR = "../../../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pickle
import time
import synmorph as sm


def run_simulation(path_name):
    pikd = open("../%s"%path_name, 'rb')
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
    j, k, Nper = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    i = j + k * Nper
    path_names = open("../scan_summary/17122022_W01_AVEp0_VEp0_v0_lambdaP_alpha_path_names.txt").readlines()
    path_name = path_names[i].split("\n")[0]

    t0 = time.time()
    run_simulation(path_name)
    t1 = time.time()
    print(t1-t0)
    out_file = open("../scan_summary/17122022_W01_AVEp0_VEp0_v0_lambdaP_alpha_result_log.txt","a")
    out_file.write("%s_%.2f"%(path_name,(t1-t0)) + "\n")
    out_file.close()
