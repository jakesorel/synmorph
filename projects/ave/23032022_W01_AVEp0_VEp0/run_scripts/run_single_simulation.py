import sys
import os

SCRIPT_DIR = "../../../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

import synmorph as sm
import pickle
import time

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

    sim.simulate(progress_bar=True)

if __name__ == "__main__":
    i = int(sys.argv[1])
    # print(i_start,N)
    path_names = open("../scan_summary/23032022_W01_AVEp0_VEp0_path_names.txt").readlines()
    path_name = path_names[i]
    t0 = time.time()
    run_simulation(path_name)
    t1 = time.time()
    out_file = open("../scan_summary/23032022_W01_AVEp0_VEp0_result_log.txt","w+")
    out_file.write("%s_%.2f"%(path_name,(t1-t0)) + "\n")
    out_file.close()