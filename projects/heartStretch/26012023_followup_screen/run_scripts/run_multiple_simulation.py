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
import linecache
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError
from multiprocessing import cpu_count
import threading
import h5py, gzip

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

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def run_simulation(path_name,p_notch_idx,seed_idx,set_idx):
    x = np.load("../scan_initialisation/x_%d.npz"%seed_idx)["arr_0"]
    is_notch = np.load("../scan_initialisation/is_notch_%d.npz"%seed_idx)["arr_0"][set_idx][p_notch_idx]

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
    sim.t.mesh.x = x
    sim.grn.is_notch = is_notch
    sim.simulate(progress_bar=False)

    with gzip.open(scan_dict["save_options"]["name"] + "_simulation.h5.gz", 'a') as gz_file:
        with h5py.File(gz_file, 'a') as hf:
            hf.create_dataset('L_save', data=sim.var_save[:,0,0].tolist())

    return sim


if __name__ == "__main__":
    base_name = "26012023_followup_screen"

    N_set, N_notch, N_seed, N_kappaA,N_kappa = 10, 10, 10, 10, 10

    total_sims_true = N_notch * N_seed * N_kappa * (N_kappa + 1) / 2 * N_kappaA
    total_sims = N_notch * N_seed * N_kappa * N_kappa * N_kappaA

    sims_per_lot = 500
    slurm_index = int(sys.argv[1])
    print("Slurm index", slurm_index)
    range_to_sample = np.arange(slurm_index*sims_per_lot,(slurm_index+1)*sims_per_lot)

    mkdir("../scan_summary")
    mkdir("../scan_results")
    mkdir("../scan_dicts")

    mkdir("../scan_summary/L")

    def run_job(i,equiangulate=True):
        t_0 = time.time()
        if not os.path.exists("../scan_results/%s_%d_simulation.h5.gz"%(base_name,i)):
            print("Simulating %d" % i)
            [i1,i2,i3,i4,i5,i6] = np.unravel_index(i, (N_set, N_notch, N_seed, N_kappaA,N_kappa,N_kappa))
            counter = i
            if i6 >= i5:
                pnotch_range = np.linspace(0, 1, N_notch)
                kappa_P_P_range = np.logspace(-1.5,0.3, N_kappa)
                kappa_P_N_range = np.logspace(-1.5,0.3, N_kappa)
                kappa_A_range = np.logspace(-1.5,0.3, N_kappaA)

                p_notch = pnotch_range[i2]
                kappa_P_P = kappa_P_P_range[i5]
                kappa_P_N = kappa_P_N_range[i6]
                kappa_A = kappa_A_range[i4]


                scan_dict_name = base_name + "_" + "%s" % counter
                # df_entry = np.array([counter, W01, AVE_p0, VE_p0, seed, scan_dict_name])
                n_cell = 250

                tissue_params = {"L": 1.0,
                                 "A0": 1 / n_cell,
                                 "P0": 3.81,
                                 "kappa_A": 1,
                                 "kappa_P": 0.1,
                                 "W": np.array(((0, 0.0), (0.0, 0))).astype(np.float32),
                                 "a": 0.,
                                 "k": 0.}
                active_params = {"v0": 0.,
                                 "Dr": 1e-1}
                init_params = {"init_noise": 0.2,
                               "c_type_proportions": (1.0, 0.0)}
                run_options = {"equiangulate": True,
                               "equi_nkill": 10}
                simulation_params = {"dt": 0.025,
                                     "tfin": 220,
                                     "tskip": 10,
                                     "dt_grn": 0.05,
                                     "grn_sim": "heart_stretch"}
                save_options = {"save": "hdf5",
                                "result_dir": "../scan_results",
                                "name": scan_dict_name,
                                "compressed": True}
                grn_params = {"p_notch": 0.5,
                              "L_init": np.sqrt(n_cell),
                              "L_min": 1.0,
                              "kappa_A_P": kappa_A,
                              "kappa_A_N": kappa_A,
                              "kappa_P_P": kappa_P_P,
                              "kappa_P_N": kappa_P_N,
                              "A0_P": 1.0,
                              "A0_N": 1.0,
                              "P0_P": 3.75,
                              "P0_N": 3.75,
                              "init_pressure": 0.0,
                              "fin_pressure": 0.15,
                              "pressure_start_time": 30.0,
                              "pressure_slope": 0.2,
                              "mu_L": 0.01,
                              "n_t": int(simulation_params["tfin"] / simulation_params["dt"]),
                              "notch_distribution": "random",
                              "osc_level": 1,
                              "heart_period": 80}

                scan_dict = {"tissue_params": tissue_params, "active_params": active_params, "init_params": init_params,
                             "run_options": run_options, "simulation_params": simulation_params, "grn_params": grn_params,
                             "save_options": save_options}

                pikd = open("../scan_dicts/%s" % scan_dict_name + ".pickle", 'wb')
                pickle.dump(scan_dict, pikd)
                pikd.close()
                print(counter)

                path_name = "../scan_dicts/%s" % scan_dict_name + ".pickle"

                sim = run_simulation(path_name,i2,i3,i1)

                file = open("../scan_summary/L/%d.txt"%counter,"w+")
                file.write(str(counter)+","+",".join(np.round(sim.var_save[:,0,0],3).astype(str).tolist()))
                file.close()


        else:
            print("Simulation %d exists, skipping"%i)


    t_tot_0 = time.time()
    Parallel(n_jobs=-1,backend="loky", prefer="threads")(delayed(run_job)(i,True) for i in range_to_sample)
    t_tot_1 = time.time()
    print("simulations completed in ",t_tot_0-t_tot_0,"s")
    sys.exit(0)
