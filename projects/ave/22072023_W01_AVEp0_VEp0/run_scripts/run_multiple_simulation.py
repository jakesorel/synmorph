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

        base_name = "22072023_W01_AVEp0_VEp0"

        if not os.path.exists("../scan_summary/%s_full_summary.csv" % base_name):
            with open("../scan_summary/%s_full_summary.csv" % base_name, "w+") as g:
                fcntl.flock(g, fcntl.LOCK_EX)
                g.write("counter,W01,AVE_p0,VE_p0,seed,scan_dict_name\n")
                fcntl.flock(g, fcntl.LOCK_UN)
                # g.close()

        if not os.path.exists("../scan_summary/%s_path_names.txt" % base_name):
            with open("../scan_summary/%s_path_names.txt" % base_name, "w+") as g:
                fcntl.flock(g, fcntl.LOCK_EX)
                # g.write("path_name\n")
                fcntl.flock(g, fcntl.LOCK_UN)

        N = 20
        total_sims = N**3
        sims_per_lot = 400
        slurm_index = int(sys.argv[1])
        print("Slurm index", slurm_index)
        range_to_sample = np.arange(slurm_index*sims_per_lot,(slurm_index+1)*sims_per_lot)

        def run_job(i,equiangulate=True):
            t_0 = time.time()
            if not os.path.exists("../scan_results/22072023_W01_AVEp0_VEp0_%d_simulation.h5.gz"%i):
                print("Simulating %d" % i)
                [i1, i2, i3, j] = np.unravel_index(i, (N, N, N, N))

                W01_range = np.logspace(-3, -1, N)
                AVE_p0_range = np.linspace(3.4, 5, N)
                VE_p0_range = np.linspace(3.4, 5, N)
                seed_range = 2023 + np.arange(N, dtype=int)
                W01 = W01_range[i1]
                AVE_p0 = AVE_p0_range[i2]
                VE_p0 = VE_p0_range[i3]
                AVE_v0 = 1e-1
                lambda_P = 0.1
                seed = seed_range[j]
                counter = np.flip([i1, i2, i3, j])
                counter = (N ** np.arange(len(counter)) * counter).sum()

                scan_dict_name = base_name + "_" + "%s" % counter
                df_entry = np.array([counter, W01, AVE_p0, VE_p0, seed, scan_dict_name])

                with open("../scan_summary/%s_full_summary.csv" % (base_name), "a+") as g:
                    fcntl.flock(g, fcntl.LOCK_EX)
                    g.write(",".join(df_entry.astype(str)) + "\n")
                    fcntl.flock(g, fcntl.LOCK_UN)

                with open("../scan_summary/%s_path_names.txt" % (base_name), "a+") as g:
                    fcntl.flock(g, fcntl.LOCK_EX)
                    g.write(scan_dict_name + "\n")
                    fcntl.flock(g, fcntl.LOCK_UN)

                tissue_params = {"L": 16.8,
                                 "A0": 1.,
                                 "P0": 3.2,
                                 "kappa_A": 1.,
                                 "kappa_P": lambda_P,
                                 "W": (np.array(((0.0, W01, W01, 0.1), (W01, 0, 0, 0.5), (W01, 0, 0, 0.5),
                                                 (0.1, 0.5, 0.5, 0.1))) * 1).astype(np.float32),
                                 "a": 0.,
                                 "k": 0.}
                active_params = {"v0": 2e-1,
                                 "Dr": 5e-3}
                init_params = {"init_noise": 0.1,
                               "c_type_proportions": (1.0, 0)}
                run_options = {"equiangulate": True,
                               "equi_nkill": 10}
                simulation_params = {"dt": 0.10,
                                     "tfin": 300,
                                     "tskip": 10,
                                     "dt_grn": 0.025,
                                     "grn_sim": "grn_ave_couple_orientation",
                                     "tinit": 10,
                                     "random_seed": int(seed)}
                grn_params = {"n_AVE_cells": 20,
                              "AVE_alpha_dir": 0.15,
                              "non_AVE_alpha_dir": 0.,
                              "AVE_v0": AVE_v0,
                              "non_AVE_v0": 0.,
                              "AVE_alpha0": -np.pi / 2,
                              "boundary_frac": 0.20,
                              "AVE_A0": 0.54,
                              "exe_frac": 0.0,
                              "AVE_p0": AVE_p0,
                              "nonAVE_p0": VE_p0}
                save_options = {"save": "hdf5",
                                "result_dir": "../scan_results",
                                "name": "AVE_example_full",
                                "compressed": True}

                scan_dict = {"tissue_params": tissue_params, "active_params": active_params, "init_params": init_params,
                             "run_options": run_options, "simulation_params": simulation_params,
                             "grn_params": grn_params,
                             "save_options": save_options}

                pikd = open("../scan_dicts/%s" % scan_dict_name + ".pickle", 'wb')
                pickle.dump(scan_dict, pikd)
                pikd.close()
                print(counter)

                path_name = "../scan_dicts/%s" % scan_dict_name + ".pickle"

                t0 = time.time()
                run_simulation(path_name)
                t1 = time.time()
                print(t1 - t0)
                out_file = open("../scan_summary/22072023_W01_AVEp0_VEp0_result_log.txt", "a")
                out_file.write("%s_%.2f" % (path_name, (t1 - t0)) + "\n")
                out_file.close()
                t_1 = time.time()
                print("Simulation completed in ", np.round(t_1-t_0),"s")
            else:
                print("Simulation %d exists, skipping"%i)

        # @exit_after(500)
        # def run_job_timed(i):
        #     return run_job(i,True)
        #
        # @exit_after(1800)
        # def run_job_timed_no_equiangulate(i):
        #     return run_job(i,False)
        #

        t_tot_0 = time.time()
        Parallel(n_jobs=-1,backend="loky", prefer="threads")(delayed(run_job)(i,False) for i in range_to_sample)

        #
        # for i in range_to_sample:
        #     run_job(i,equiangulate=True)
        #     # try:
        #     #     run_job_timed(i)
        #     # except:
        #     #     print("Equiangulation timed out")
        #     #     try:
        #     #         run_job_timed_no_equiangulate(i)
        #     #     except:
        #     #         print("Forced triangulation timed out too.. giving up")

        t_tot_1 = time.time()
        print("400 simulations completed in ",t_tot_0-t_tot_0,"s")
        sys.exit(0)

    except TerminatedWorkerError:
        # Handle the error and initiate a restart
        print("TerminatedWorkerError occurred. Restarting...")
        sys.exit(1)  # Or any other action to restart the execution


"""
Note, may need to do some tidying up of the dataset. 

Need to deal with instances where the code crashed mid export or mid zip. 

mid export --> .h5 present, but .gz not present. Here, need to repeat the simulation
mid zip --> .h5 and .gz both present. Here, need to repeat the zipping. 
"""