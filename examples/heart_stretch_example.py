import numpy as np
import synmorph as sm
import time
from synmorph.sim_plotting import plot_vor

import matplotlib.pyplot as plt



tissue_params = {"L": 1.0,
                 "A0": 0.0123,
                 "P0": 3.81,
                 "kappa_A": 1,
                 "kappa_P": 0.1,
                 "W": np.array(((0, 0.0), (0.0, 0))).astype(np.float32),
                 "a": 0.,
                 "k": 0.}
active_params = {"v0": 0.,
                 "Dr": 1e-1}
init_params = {"init_noise": 0.00005,
               "c_type_proportions": (1.0,0.0)}
run_options = {"equiangulate": True,
               "equi_nkill": 10}
simulation_params = {"dt": 0.05,
                     "tfin": 100,
                     "tskip": 10,
                     "dt_grn": 0.05,
                     "grn_sim": "heart_stretch"}
save_options = {"save": "skeleton",
                "result_dir": "results",
                "name": "heart_stretch",
                "compressed": True}
grn_params = {"p_notch":0.1,
              "L_init":10.0,
              "L_min":1.0,
              "kappa_A_P":0.1,
              "kappa_A_N":0.1,
              "kappa_P_P":0.1,
              "kappa_P_N": 1.,
              "A0_P": 1.0,
              "A0_N": 1.0,
              "P0_P": 2.5,
              "P0_N": 3.75,
              "init_pressure":0.0,
              "fin_pressure":0.0,
              "pressure_start_time":30.0,
              "pressure_slope":0.2,
              "mu_L":0.1,
              "n_t":int(simulation_params["tfin"]/simulation_params["dt"])
}
sim = sm.simulation(tissue_params=tissue_params,
                    active_params=active_params,
                    init_params=init_params,
                    simulation_params=simulation_params,
                    run_options=run_options,
                    save_options=save_options,
                    grn_params=grn_params)

t0 = time.time()
sim.simulate(progress_bar=True)
t1= time.time()
print(t1-t0)

plt.plot(sim.grn.L_save)
plt.show()

fig, ax = plt.subplots()
plot_vor(ax, sim.x_save[-1], sim.t.mesh.L)
fig.show()

sim.animate_c_types(n_frames=20,
                    c_type_col_map=["#FFBA40", "#67F5B5"],
                    file_name=None)

sim.save_dir_plots = "plots"

sim.animate_grn(vari=0,
                n_frames=20,
                cmap=plt.cm.inferno)

