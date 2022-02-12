import os
import numpy as np
from copy import deepcopy

import psutil
import multiprocessing as mp
from tqdm import tqdm

from sorting_run_one import ex


## Set params values to scan
# W_val (off-diagonals of adhesion matrix W)
min_val = 0.
max_val = 0.015
n_vals  = 15

# Set chunksize for workers
chunksize = 5

# Extract default tissue parameters
tp_dict = deepcopy(ex.configurations[0]._conf["tissue_params"])

def do_sim_in_parallel(_wv):
    """Perform a sim on one worker."""

    # Make adhesion matrix
    _w = [[0., _wv], [_wv, 0.]]
    
    # Update tissue params dict
    tp = deepcopy(tp_dict)
    tp["W"] = _w

    # Run with modified configurations
    cfg_updates = dict(
        tissue_params=tp,
    )
    ex.run(config_updates=cfg_updates)


# Parallelize simulation
if __name__ == '__main__':
    
    # Get param values to change
    W_vals = [float(v) for v in np.linspace(min_val, max_val, n_vals)]

    print("Assembling worker pool")

    # Get worker pool
    pool = mp.Pool(psutil.cpu_count(logical=False))
    
    print("Performing parallel simulations")

    # Run in parallel
    results = pool.imap(do_sim_in_parallel, W_vals, chunksize=chunksize)
    
    pool.close()
    pool.join()


