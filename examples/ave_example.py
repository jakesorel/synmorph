import matplotlib.pyplot as plt
import numpy as np

import synmorph as sm
from synmorph.analysis import spatial

tissue_params = {"L": 15,
                 "A0": 1,
                 "P0": 4,
                 "kappa_A": 1,
                 "kappa_P": 0.1,
                 "W": np.array(((0.0, 0.0762,0.0762), (0.0762, 0,0.0762),(0,0,0.0762)))*1,
                 "a": 0,
                 "k": 0}
active_params = {"v0": 2e-1,
                 "Dr": 1e-1}
init_params = {"init_noise": 0.005,
               "c_type_proportions": (1.0, 0)}
run_options = {"equiangulate": True,
               "equi_nkill": 3}
simulation_params = {"dt": 0.05,
                     "tfin": 100,
                     "tskip": 10,
                     "dt_grn": 0.025,
                     "grn_sim": "grn_ave"}
grn_params = {"n_AVE_cells":20,
              "AVE_alpha_dir":0.05,
              "non_AVE_alpha_dir":0,
              "AVE_v0":5e-2,
              "non_AVE_v0":5e-3,
              "AVE_alpha0":-np.pi/2}
save_options = {"save": "last",
                "result_dir": "results",
                "name": "AVE_example2",
                "compressed": True}
sim = sm.simulation(tissue_params=tissue_params,
                    active_params=active_params,
                    init_params=init_params,
                    simulation_params=simulation_params,
                    grn_params=grn_params,
                    run_options=run_options,
                    save_options=save_options)

sim.simulate(progress_bar=True)

sim.animate_c_types(n_frames=20,
                    c_type_col_map=["#108c00", "#fa2111","#666699"],
                    file_name="alpha_dir_AVE0-05")


fig, ax = plt.subplots()
pos = spatial.displacements(sim.x_save,sim.t.mesh.L) + sim.x_save[0]
tmax = 800
pos_sample = pos[:tmax,sim.t.c_types==0]
ax.plot(pos_sample[:,:,0],pos_sample[:,:,1],color='black')
last_dir = pos_sample[-1] - pos_sample[-2]
last_dir = last_dir/np.expand_dims(np.linalg.norm(last_dir,axis=1),1)
ax.quiver(pos_sample[-1,:,0],pos_sample[-1,:,1],last_dir[:,0],last_dir[:,1],color='black')
ax.set(aspect=1)
ax.axis("off")
fig.savefig("results/AVE_example2/plots/tracks.pdf")

plt.show()

fig, ax = plt.subplots()
tmax = -1

pos_ave = pos[:,sim.t.c_types==1]
av_pos_ave = pos_ave.mean(axis=1)
pos_rel_to_ave = pos - np.expand_dims(av_pos_ave,1)
ax.plot(pos_rel_to_ave[:tmax,sim.t.c_types==0,0],pos_rel_to_ave[:tmax,sim.t.c_types==0,1],color='black')
fig.show()

velocity = pos_rel_to_ave[1:] - pos_rel_to_ave[:-1]
pos_tm1 = pos_rel_to_ave[:-1]



from scipy import interpolate
L = sim.t.mesh.L
xx = np.linspace(-L/2, L/2,50)
yy = np.linspace(-L/2,L/2, 50)


xx, yy = np.meshgrid(xx, yy)

# points = np.transpose(np.vstack((x, y)))
u_interp = interpolate.griddata(pos_tm1.reshape(-1,2), velocity[:,:,0].ravel(), (xx, yy), method='cubic')
v_interp = interpolate.griddata(pos_tm1.reshape(-1,2), velocity[:,:,1].ravel(),(xx, yy),  method='cubic')


# plt.figure(2)
plt.quiver(xx, yy, u_interp, v_interp)
plt.show()


fig, ax = plt.subplots()
ax.streamplot(pos_tm1[:,:,0].ravel(),pos_tm1[:,:,1].ravel(),velocity[:,:,0].ravel(),velocity[:,:,1].ravel())
fig.show()


from scipy.stats import binned_statistic_2d
import numpy as np

velocity = pos_rel_to_ave[1:,sim.t.c_types==0] - pos_rel_to_ave[:-1,sim.t.c_types==0]
pos_tm1 = pos_rel_to_ave[:-1,sim.t.c_types==0]

velocity = pos[1:,sim.t.c_types==0] - pos[:-1,sim.t.c_types==0]
pos_tm1 = pos[:-1,sim.t.c_types==0]

# x_bins = np.linspace(-L/2, L/2,15)
# y_bins = np.linspace(-L/2,L/2, 15)
x_bins = np.linspace(0, L,15)
y_bins = np.linspace(0, L,15)

xx, yy = np.meshgrid(x_bins, y_bins)

tmax = 500

ret_x = binned_statistic_2d(pos_tm1[:tmax,:,0].ravel(),pos_tm1[:tmax,:,1].ravel(), velocity[:tmax,:,0].ravel(), statistic=np.mean, bins=[x_bins, y_bins])
ret_y = binned_statistic_2d(pos_tm1[:tmax,:,0].ravel(),pos_tm1[:tmax,:,1].ravel(), velocity[:tmax,:,1].ravel(), statistic=np.mean, bins=[x_bins, y_bins])

av_velocity = np.sqrt(ret_x.statistic**2 + ret_y.statistic**2)
av_velocity /= np.nanmean(av_velocity)

x_bin_centres = (x_bins[1:] + x_bins[:-1])/2
y_bin_centres = (y_bins[1:] + y_bins[:-1])/2

xx, yy = np.meshgrid(x_bin_centres, y_bin_centres)

fig, ax = plt.subplots(figsize=(4,4))
# ax.set(aspect=1,xlim=(-L/2,L/2),ylim=(-L/2,L/2))
ax.streamplot(xx,yy,ret_x.statistic,ret_y.statistic,linewidth=av_velocity,color=plt.cm.inferno(0.3))
# ax.quiver(xx,yy,ret_x.statistic,ret_y.statistic)

ax.axis("off")
fig.savefig("results/AVE_example/plots/streamplot.pdf")