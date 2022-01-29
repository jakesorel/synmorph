import matplotlib.pyplot as plt
import numpy as np

import synmorph as sm

tissue_params = {"L": 9,
                 "A0": 1,
                 "P0": 3.81,
                 "kappa_A": 1,
                 "kappa_P": 0.1,
                 "W": np.array(((0, 0.0762), (0.0762, 0))),
                 "a": 0,
                 "k": 0}
active_params = {"v0": 1e-1,
                 "Dr": 1e-1}
init_params = {"init_noise": 0.005,
               "c_type_proportions": (1.0, 0)}
run_options = {"equiangulate": True,
               "equi_nkill": 3}
simulation_params = {"dt": 0.05,
                     "tfin": 60,
                     "tskip": 10,
                     "dt_grn": 0.025,
                     "grn_sim": "repressilator_move"}
grn_params = {"init_type": "uniform",
              "init_val_min": np.array((0.9, 0.9, 0.9)),
              "init_val_max": np.array((1.2, 1.2, 1.2)),
              "dt_equilibrate": 0.1,
              "tfin_equilibrate": 10,
              "kappa": 0.05,
              "beta": 40,
              "n": 3,
              "sf": 0.5,
              "mu_move": 0.01}
save_options = {"save": "last",
                "result_dir": "dynamic_activity_example",
                "name": None,
                "compressed": True}
sim = sm.simulation(tissue_params=tissue_params,
                    active_params=active_params,
                    init_params=init_params,
                    simulation_params=simulation_params,
                    grn_params=grn_params,
                    run_options=run_options,
                    save_options=save_options)

sim.simulate(progress_bar=True)

sim.animate_grn(vari=0,
                n_frames=60,
                cmap=plt.cm.inferno)
