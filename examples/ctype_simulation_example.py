import numpy as np
import synmorph as sm
import time




tissue_params = {"L": 9.,
                 "A0": 1.,
                 "P0": 3.81,
                 "kappa_A": 1,
                 "kappa_P": 0.1,
                 "W": np.array(((0, 0.00762), (0.00762, 0))).astype(np.float32),
                 "a": 0.,
                 "k": 0.}
active_params = {"v0": 1e-1,
                 "Dr": 1e-1}
init_params = {"init_noise": 0.00005,
               "c_type_proportions": (0.5,0.5)}
run_options = {"equiangulate": True,
               "equi_nkill": 10}
simulation_params = {"dt": 0.025,
                     "tfin": 100,
                     "tskip": 100,
                     "grn_sim": None,
                     "random_seed":10}
save_options = {"save": "skeleton",
                "result_dir": "results",
                "name": "ctype_example2",
                "compressed": True}
sim = sm.simulation(tissue_params=tissue_params,
                    active_params=active_params,
                    init_params=init_params,
                    simulation_params=simulation_params,
                    run_options=run_options,
                    save_options=save_options)

t0 = time.time()
sim.simulate(progress_bar=True)
t1= time.time()
print(t1-t0)


sim.animate_c_types(n_frames=20,
                    c_type_col_map=["#FFBA40", "#67F5B5"],
                    file_name=None)

