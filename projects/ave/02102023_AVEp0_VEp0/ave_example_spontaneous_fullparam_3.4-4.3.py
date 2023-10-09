import matplotlib.pyplot as plt
import numpy as np

import synmorph as sm
from synmorph.analysis import spatial
import synmorph.analysis.topological as top
from synmorph import sim_plotting as plot

"""
How many free params:

mechanical
---------
AVE vs rest diff adhesion
AVE p0
EPI-VE p0
ExE-VE p0
kappa_P

frictions 

active
------
v0 
AVE alpha dir
AVE Dr


A first pass: 
AVE-EPI

- diff adhesion 
- AVE p0 
- ExE p0 
- (v0)

"""

W01 = 0.0
AVE_p0 = 3.4
VE_p0 = 4.3
AVE_v0 = 0.05
lambda_P = 0.2
seed = 2023


tissue_params = {"L": 17.0,
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
simulation_params = {"dt": 0.05,
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
              "nonAVE_p0": VE_p0,
              "ExEVE_p0": 4.0}
save_options = {"save": "hdf5",
                "result_dir": "../scan_results",
                "name": "AVE_example_full",
                "compressed": True}

scan_dict = {"tissue_params": tissue_params, "active_params": active_params, "init_params": init_params,
             "run_options": run_options, "simulation_params": simulation_params, "grn_params": grn_params,
             "save_options": save_options}

sim = sm.simulation(tissue_params=tissue_params,
                    active_params=active_params,
                    init_params=init_params,
                    simulation_params=simulation_params,
                    grn_params=grn_params,
                    run_options=run_options,
                    save_options=save_options)

sim.save_dir_plots = "results"
sim.simulate(progress_bar=True)

import os


i_range = np.arange(0,500,5).astype(int)

def save_fig(i,dir):
    fig, ax = plt.subplots(figsize=(3,3))

    plot.plot_vor(ax,sim.x_save[i].astype(np.float32),sim.t.tissue_params["L"],cols=plot.generate_ctype_cols(sim.t.c_types,c_type_col_map=["#399cc3", "#e4e4e4","#cbcccc","white"]))
    fig.savefig(dir + "/%d.png"%i,dpi=600)

save_dir_name = "results/3.4_4.3_"
if not os.path.exists(save_dir_name):
    os.mkdir(save_dir_name)

for i in i_range:
    save_fig(i, save_dir_name)
    plt.close("all")
#

L = sim.t.mesh.L
pos = spatial.displacements(sim.x_save.astype(np.float32),sim.t.mesh.L) + sim.x_save[0]


from scipy.stats import binned_statistic_2d
import numpy as np

# velocity = pos_rel_to_ave[1:,sim.t.c_types==0] - pos_rel_to_ave[:-1,sim.t.c_types==0]
# pos_tm1 = pos_rel_to_ave[:-1,sim.t.c_types==0]
mask = sim.t.c_types<=2

velocity = pos[1:,mask] - pos[:-1,mask]
pos_tm1 = pos[:-1,mask]

# x_bins = np.linspace(-L/2, L/2,15)
# y_bins = np.linspace(-L/2,L/2, 15)
x_bins = np.linspace(0, L,10)
y_bins = np.linspace(0, L,10)

xx, yy = np.meshgrid(x_bins, y_bins)

tmin = 175
tmax = tmin+50

ret_x = binned_statistic_2d(pos_tm1[tmin:tmax,:,0].ravel(),pos_tm1[tmin:tmax,:,1].ravel(), velocity[tmin:tmax,:,0].ravel(), statistic=np.nanmean, bins=[x_bins, y_bins])
ret_y = binned_statistic_2d(pos_tm1[tmin:tmax,:,0].ravel(),pos_tm1[tmin:tmax,:,1].ravel(), velocity[tmin:tmax,:,1].ravel(), statistic=np.nanmean, bins=[x_bins, y_bins])
ret_x_count = binned_statistic_2d(pos_tm1[tmin:tmax,:,0].ravel(),pos_tm1[tmin:tmax,:,1].ravel(), velocity[tmin:tmax,:,0].ravel(), statistic="count", bins=[x_bins, y_bins])
ret_y_count = binned_statistic_2d(pos_tm1[tmin:tmax,:,0].ravel(),pos_tm1[tmin:tmax,:,1].ravel(), velocity[tmin:tmax,:,1].ravel(), statistic="count", bins=[x_bins, y_bins])


av_velocity = np.sqrt(ret_x.statistic**2 + ret_y.statistic**2)
av_velocity /= np.nanmean(av_velocity)

x_bin_centres = (x_bins[1:] + x_bins[:-1])/2
y_bin_centres = (y_bins[1:] + y_bins[:-1])/2

xx, yy = np.meshgrid(x_bin_centres, y_bin_centres)


vx,vy = ret_x.statistic.T,ret_y.statistic.T
n_vx,n_vy = vx/np.sqrt(vx**2 + vy**2),vy/np.sqrt(vx**2 + vy**2)
n_vx[ret_x_count.statistic<=25] = np.nan
n_vy[ret_x_count.statistic<=25] = np.nan

fig, ax = plt.subplots(figsize=(4,4))
# ax.set(aspect=1,xlim=(-L/2,L/2),ylim=(-L/2,L/2))
stream = ax.streamplot(xx,yy,n_vx,n_vy,color="#d0417e",density=1.6,linewidth=2)#start_points=
ax.scatter(xx*n_vx/n_vx,yy*n_vx/n_vx)
alpha = 0.6
stream.lines.set_alpha(alpha)
stream.arrows.set_alpha(alpha)

plot.plot_vor(ax,sim.x_save[int(tmax/2+tmin/2)].astype(np.float32),sim.t.tissue_params["L"],cols=plot.generate_ctype_cols(sim.t.c_types,c_type_col_map=["#399cc3", "lightgrey","lightgrey","white"]))

# ax.quiver(xx,yy,ret_x.statistic,ret_y.statistic)
ax.axis("off")
fig.savefig("results/streamplot.pdf",dpi=300)
# fig.savefig("results/AVE_example2/plots/test.pdf")
