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


if __name__ == "__main__":
    try:

        base_name = "17012024_pilotscreen_kappaP_notch"

        N_notch,N_kappa,N_P0,N_seed = 6,12,4,10

        total_sims = N_notch * N_seed * N_kappa**2 * N_P0
        sims_per_lot = 144
        slurm_index = int(sys.argv[1])
        print("Slurm index", slurm_index)
        range_to_sample = np.arange(slurm_index*sims_per_lot,(slurm_index+1)*sims_per_lot)

        mkdir("../scan_summary")
        mkdir("../scan_results")
        mkdir("../scan_dicts")

        mkdir("../scan_summary/L")

        def run_job(i,equiangulate=True):
            t_0 = time.time()
            if not os.path.exists("../scan_results/22072023_W01_AVEp0_VEp0_%d_simulation.h5.gz"%i):
                print("Simulating %d" % i)
                [i1, i2, i3,i4, j] = np.unravel_index(i, (N_notch,N_kappa,N_kappa,N_P0,N_seed))
                counter = i
                pnotch_range = np.linspace(0, 1, N_notch*2)[:N_notch]
                kappa_P_P_range = np.logspace(-2,0, N_kappa)
                kappa_P_N_range = np.logspace(-2,0, N_kappa)
                P0_range = np.linspace(3.5,4.2, N_P0)

                seed_range = 2024 + np.arange(N_seed, dtype=int)
                p_notch = pnotch_range[i1]
                kappa_P_P = kappa_P_P_range[i2]
                kappa_P_N = kappa_P_N_range[i3]
                P0 = P0_range[i4]

                seed = seed_range[j] ###NOTE, SEED IS NOT USED SO TRUE RANDOM INSTEAD

                scan_dict_name = base_name + "_" + "%s" % counter
                # df_entry = np.array([counter, W01, AVE_p0, VE_p0, seed, scan_dict_name])

                tissue_params = {"L": 1.0,
                                 "A0": 0.00797, ##nb this is just shorthand to force the
                                 "P0": 3.81,
                                 "kappa_A": 1,
                                 "kappa_P": 0.1,
                                 "W": np.array(((0, 0.0), (0.0, 0))).astype(np.float32),
                                 "a": 0.,
                                 "k": 0.}
                active_params = {"v0": 0.,
                                 "Dr": 1e-1}
                init_params = {"init_noise": 0.1,
                               "c_type_proportions": (1.0, 0.0)}
                run_options = {"equiangulate": True,
                               "equi_nkill": 10}
                simulation_params = {"dt": 0.05,
                                     "tfin": 100,
                                     "tskip": 10,
                                     "dt_grn": 0.05,
                                     "grn_sim": "heart_stretch"}
                save_options = {"save": "hdf5",
                                "result_dir": "../scan_results",
                                "name": scan_dict_name,
                                "compressed": True}
                grn_params = {"p_notch": p_notch,
                              "L_init": 10.0,
                              "L_min": 1.0,
                              "kappa_A_P": 0.1,
                              "kappa_A_N": 0.1,
                              "kappa_P_P": kappa_P_P,
                              "kappa_P_N": kappa_P_N,
                              "A0_P": 1.0,
                              "A0_N": 1.0,
                              "P0_P": P0,
                              "P0_N": P0,
                              "init_pressure": 0.0,
                              "fin_pressure": 0.2,
                              "pressure_start_time": 30.0,
                              "pressure_slope": 0.2,
                              "mu_L": 0.02,
                              "n_t": int(simulation_params["tfin"] / simulation_params["dt"])
                              }


                scan_dict = {"tissue_params": tissue_params, "active_params": active_params, "init_params": init_params,
                             "run_options": run_options, "simulation_params": simulation_params, "grn_params": grn_params,
                             "save_options": save_options}

                pikd = open("../scan_dicts/%s" % scan_dict_name + ".pickle", 'wb')
                pickle.dump(scan_dict, pikd)
                pikd.close()
                print(counter)

                path_name = "../scan_dicts/%s" % scan_dict_name + ".pickle"

                t0 = time.time()
                sim = run_simulation(path_name)
                t_1 = time.time()

                L_fin = sim.grn.L_save[-1]
                L_min = sim.grn.L_save.min()
                L_pre = sim.grn.L_save[50]


                file = open("../scan_summary/L/%d.txt"%counter,"w+")
                file.write("%d,%.3f,%.3f,%.3f\n"%(counter,L_fin,L_min,L_pre))
                file.close()

                print("Simulation completed in ", np.round(t_1-t_0),"s")
            else:
                print("Simulation %d exists, skipping"%i)


        t_tot_0 = time.time()
        Parallel(n_jobs=-1,backend="loky", prefer="threads")(delayed(run_job)(i,True) for i in range_to_sample)

        t_tot_1 = time.time()
        print("98 simulations completed in ",t_tot_0-t_tot_0,"s")
        sys.exit(0)

    except TerminatedWorkerError:
        # Handle the error and initiate a restart
        print("TerminatedWorkerError occurred. Restarting...")
        sys.exit(1)  # Or any other action to restart the execution
