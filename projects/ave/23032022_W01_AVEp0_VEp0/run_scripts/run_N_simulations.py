import sys
import os

SCRIPT_DIR = "../../../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

import synmorph as sm
import pickle

def run_simulation(path_name):
    pikd = open("../%s"%path_name, 'rb')
    scan_dict = pickle.load(pikd)
    pikd.close()

    sim = sm.simulation(tissue_params=scan_dict["tissue_params"],
                        active_params=scan_dict["active_params"],
                        init_params=scan_dict["init_params"],
                        simulation_params=scan_dict["simulation_params"],
                        grn_params=scan_dict["grn_params"],
                        run_options=scan_dict["run_options"],
                        save_options=scan_dict["save_options"])

    sim.simulate(progress_bar=True)

if __name__ == "__main__":
    # print(sys.argv[1],sys.argv[2])
    i_start,N = int(sys.argv[1]),int(sys.argv[2])
    i_start*=N
    print(i_start,N)
    path_names = open("../scan_summary/23032022_W01_AVEp0_VEp0_path_names.txt").readlines()
    path_names = path_names[i_start:i_start+N]
    for path_name in path_names:
        run_simulation(path_name.strip("\n"))
