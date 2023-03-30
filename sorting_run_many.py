import os
import numpy as np
from copy import deepcopy

from sorting_run_one import ex


## Set params values to scan
# W_vals = [float(v) for v in np.linspace(0.,0.015,3)]
W_vals = [0.0, 0.01, 0.05, 0.1]

# Extract default tissue parameters
tp = deepcopy(ex.configurations[0]._conf["tissue_params"])

for _wv in W_vals:

    # Make adhesion matrix
    _w = [[0.0, _wv], [_wv, 0.0]]

    # Update tissue params dict
    tp["W"] = _w

    # Run with modified configurations
    cfg_updates = dict(
        tissue_params=tp,
        animate=True,
    )
    ex.run(config_updates=cfg_updates)
