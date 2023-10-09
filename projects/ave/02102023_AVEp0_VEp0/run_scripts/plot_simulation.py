import os
import sys


SCRIPT_DIR = "../../../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np

import synmorph as sm
from synmorph.analysis import spatial
import synmorph.analysis.topological as top
from synmorph import sim_plotting as plot
from joblib import Parallel, delayed

import os

def plot_simulation(sim_name,i):
    pikd = open("../scan_dicts/%s.pickle" % sim_name, 'rb')
    scan_dict = pickle.load(pikd)
    pikd.close()

    L = scan_dict["tissue_params"]["L"]
    sim_dict = load_hdf5_skeleton("../scan_results/%s_simulation.h5.gz" % sim_name, L)

    x_save = np.array(sim_dict["x_save"], dtype=np.float32)
    t_span_save = np.arange(0, scan_dict["simulation_params"]["tfin"],
                            scan_dict["simulation_params"]["dt"] * scan_dict["simulation_params"]["tskip"],
                            dtype=np.float32)
    tri_save = np.array(sim_dict["tri_save"], dtype=np.int32)
    c_types = np.array(sim_dict["c_types"], dtype=np.int32)

    i_range = np.arange(0,len(tri_save),int(len(tri_save)/100)).astype(int)

    def save_fig(i,dir):
        fig, ax = plt.subplots(figsize=(3,3))

        plot.plot_vor(ax,x_save[i].astype(np.float32),L,cols=plot.generate_ctype_cols(c_types,c_type_col_map=["#399cc3", "#e4e4e4","#cbcccc","white"]))
        fig.savefig(dir + "/%d.png"%i,dpi=600)

    save_dir_name = "../analysis_results/plots/%d" % i
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)

    for j in i_range:
        save_fig(j, save_dir_name)
        plt.close("all")


if __name__ == "__main__":
    if not os.path.exists("../analysis_results/plots"):
        os.mkdir("../analysis_results/plots")

    N = 20
    M = 100
    total_sims = N**2 * M
    sims_per_lot = M
    corners = np.array(((4,4),
                        (4,14),
                        (14,4),
                        (14,14)))



    def run_simulation_plotting(i):
        sim_name = "02102023_AVEp0_VEp0_%d" % i
        if not os.path.exists("../analysis_results/plots/%d" % i):
            if not os.path.exists("../analysis_results/plots/%d" % i):
                os.mkdir("../analysis_results/plots/%d" % i)

            try:
                plot_simulation(sim_name,i)
            except:
                print("file not found, skipping")

    for corner in corners:
        range_to_sample = np.array([np.ravel_multi_index((corner[0],corner[1],i),(20,20,100)) for i in range(100)])
        Parallel(n_jobs=-1,backend="loky", prefer="threads")(delayed(run_simulation_plotting)(i) for i in range_to_sample)

