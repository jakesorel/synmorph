import synmorph as sm
import sys
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
    path_name = sys.argv[0]
    run_simulation(path_name)
    # path_name = "scan_dicts/23032022_W01_AVEp0_VEp0_0.pickle"
