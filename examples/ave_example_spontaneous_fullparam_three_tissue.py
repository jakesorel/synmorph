import matplotlib.pyplot as plt
import numpy as np

import synmorph as sm
from synmorph.analysis import spatial
from synmorph.analysis import geometrical as geo
import synmorph.analysis.topological as top
from synmorph import sim_plotting as plot
import synmorph.tri_functions as trf
from scipy.sparse import coo_matrix
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

"""
It seems like the velocity needs to be not too high such that the AVE stalls. 


"""


W01 = 0.1
AVE_p0 = 3.5
VE_p0 = 4.3
AVE_v0 = 0.05
lambda_P = 0.2
seed = 2023


tissue_params = {"L": 25,
                 "A0": 1.,
                 "P0": 3.4,
                 "kappa_A": 1.,
                 "kappa_P": lambda_P,
                 "W": (np.array(((0.0, W01, W01, 0.1), (W01, 0, 0, 0.5), (W01, 0, 0, 0.5),
                                 (0.1, 0.5,0.5, 0.1))) * 1).astype(np.float32),
                 "a": 0.,
                 "k": 0.}
active_params = {"v0": 2e-1,
                 "Dr": 5e-3}
init_params = {"init_noise": 0.1,
               "c_type_proportions": (1.0, 0)}
run_options = {"equiangulate": True,
               "equi_nkill": 10}
simulation_params = {"dt": 0.25,
                     "tfin": 900,
                     "tskip": 10,
                     "dt_grn": 0.025,
                     "grn_sim": "grn_ave_couple_orientation",
                     "tinit": 10,
                     "random_seed": int(seed)}
grn_params = {"n_AVE_cells": 20,
              "AVE_alpha_dir": 0.2,
              "non_AVE_alpha_dir": 0.,
              "AVE_v0": AVE_v0,
              "non_AVE_v0": 0.,
              "AVE_alpha0": -np.pi / 2,
              "boundary_frac": 0.1,
              "AVE_A0": 0.54,
              "exe_frac": 0.45,
              "AVE_p0": AVE_p0,
              "nonAVE_p0": VE_p0,
              "ExEVE_p0": 3.4}
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


# sim.t.kappa_P[sim.t.c_types==2] = 1.
sim.initialize()
fig, ax = plt.subplots()
plot.plot_vor(ax, sim.t.mesh.x, sim.t.tissue_params["L"],
              cols=plot.generate_ctype_cols(sim.t.c_types,
                                            c_type_col_map=["#399cc3", "lightgrey", "green", "white"]))
fig.show()

sim.save_dir_plots = "results"
sim.simulate(progress_bar=True)

sim.animate_c_types(n_frames=15,
                    c_type_col_map=["#4bdb71", "#ffbb4d","green","white"],
                    file_name="three tissue")
#

i_range = np.arange(0,360,360/6).astype(int)

for j, i in enumerate([0]+list(i_range)):
    fig, ax = plt.subplots(figsize=(4,4))

    plot.plot_vor(ax,sim.x_save[i].astype(np.float32),sim.t.tissue_params["L"],cols=plot.generate_ctype_cols(sim.t.c_types,c_type_col_map=["#399cc3", "#e4e4e4","#cbcccc","white"]))
    fig.savefig("results/dynamics 3 tissue_%d.pdf"%j,dpi=300)




def get_total_t1s():
    tri_save = sim.tri_save

    meshes = geo.mesh_assembler(sim.x_save.astype(np.float32), sim.tri_save.astype(np.int32), sim.t.mesh.L, run_options)

    ave_mask = (sim.t.c_types==0)

    l_int_AVE_save = np.zeros((len(meshes),sum(ave_mask),sum(ave_mask)))
    for i, mesh in enumerate(meshes):
        l_int = mesh.l_int
        l_int_AVE_save[i] = (l_int.toarray()[ave_mask].T[ave_mask].T)

    total_t1s = ((l_int_AVE_save[1:]!=0)!=(l_int_AVE_save[:-1]!=0)).sum()/4

#
# fig, ax = plt.subplots()
pos = spatial.displacements(sim.x_save.astype(np.float32),sim.t.mesh.L) + sim.x_save[0]
# tmax = 800
# pos_sample = pos[:tmax,sim.t.c_types==0]
# ax.plot(pos_sample[:,:,0],pos_sample[:,:,1],color='black')
# last_dir = pos_sample[-1] - pos_sample[-2]
# last_dir = last_dir/np.expand_dims(np.linalg.norm(last_dir,axis=1),1)
# ax.quiver(pos_sample[-1,:,0],pos_sample[-1,:,1],last_dir[:,0],last_dir[:,1],color='black')
# ax.set(aspect=1)
# ax.axis("off")
# fig.savefig("results/AVE_example2/plots/tracks.pdf")
#
# plt.show()
#
# fig, ax = plt.subplots()
# tmax = -1
#
# pos_ave = pos[:,sim.t.c_types==1]
# av_pos_ave = pos_ave.mean(axis=1)
# pos_rel_to_ave = pos - np.expand_dims(av_pos_ave,1)
# ax.plot(pos_rel_to_ave[:tmax,sim.t.c_types==0,0],pos_rel_to_ave[:tmax,sim.t.c_types==0,1],color='black')
# fig.show()
#
# velocity = pos_rel_to_ave[1:] - pos_rel_to_ave[:-1]
# pos_tm1 = pos_rel_to_ave[:-1]
#
#
#
# from scipy import interpolate
# L = sim.t.mesh.L
# xx = np.linspace(-L/2, L/2,50)
# yy = np.linspace(-L/2,L/2, 50)
#
#
# xx, yy = np.meshgrid(xx, yy)
#
# # points = np.transpose(np.vstack((x, y)))
# u_interp = interpolate.griddata(pos_tm1.reshape(-1,2), velocity[:,:,0].ravel(), (xx, yy), method='cubic')
# v_interp = interpolate.griddata(pos_tm1.reshape(-1,2), velocity[:,:,1].ravel(),(xx, yy),  method='cubic')
#
#
# # plt.figure(2)
# plt.quiver(xx, yy, u_interp, v_interp)
# plt.show()
#
#
# fig, ax = plt.subplots()
# x,y = pos_tm1[:,:,0].ravel(),pos_tm1[:,:,1].ravel()
# vx, vy = velocity[:,:,0].ravel(),velocity[:,:,1].ravel()
# order = np.argsort(x)
# x,y = x[order],y[order]
# vx,vy = vx[order],vy[order]
# ax.streamplot(x,y,vx,vy)
# fig.show()


from scipy.stats import binned_statistic_2d
import numpy as np

mask = sim.t.c_types==1
pos_tm1 = pos[:-1,mask]

fig, ax = plt.subplots()
plot.plot_vor(ax,sim.x_save[100].astype(np.float32),sim.t.tissue_params["L"],cols=plot.generate_ctype_cols(sim.t.c_types,c_type_col_map=["darkgrey", "lightgrey","lightgrey","white"]))
for i in range(pos_tm1.shape[1]):
    ax.plot(pos_tm1[:100,i,0],pos_tm1[:100,i,1],color="magenta",alpha=0.6)
fig.show()
# pos_tm1[...,1] = L - pos_tm1[...,1]
velocity = pos[1:,mask] - pos[:-1,mask]
# velocity[...,1] = -velocity[...,1]
# x_bins = np.linspace(-L/2, L/2,15)
# y_bins = np.linspace(-L/2,L/2, 15)
n_bin = 10
x_bins = np.linspace(0, L,n_bin)
y_bins = np.linspace(0, L,n_bin)

xx, yy = np.meshgrid(x_bins, y_bins)

tmin = 0
tmax = 100

vx_av = velocity[tmin:tmax,:,0].ravel().mean()
vy_av = velocity[tmin:tmax,:,1].ravel().mean()

ret_x = binned_statistic_2d(pos_tm1[tmin:tmax,:,0].ravel(),pos_tm1[tmin:tmax,:,1].ravel(), velocity[tmin:tmax,:,0].ravel(), statistic=np.nanmean, bins=[x_bins, y_bins])
ret_y = binned_statistic_2d(pos_tm1[tmin:tmax,:,0].ravel(),pos_tm1[tmin:tmax,:,1].ravel(), velocity[tmin:tmax,:,1].ravel(), statistic=np.nanmean, bins=[x_bins, y_bins])

av_velocity = np.sqrt(ret_x.statistic**2 + ret_y.statistic**2)
av_velocity /= np.nanmean(av_velocity)

vx,vy = ret_x.statistic,ret_y.statistic
n_vx,n_vy = vx/np.sqrt(vx**2 + vy**2),vy/np.sqrt(vx**2 + vy**2)

x_bin_centres = (x_bins[1:] + x_bins[:-1])/2
y_bin_centres = (y_bins[1:] + y_bins[:-1])/2

xx, yy = np.meshgrid(x_bin_centres, y_bin_centres)

fig, ax = plt.subplots(figsize=(4,4))

# ax.set(aspect=1,xlim=(-L/2,L/2),ylim=(-L/2,L/2))
streamlines = ax.streamplot(xx,yy,n_vx,n_vy,linewidth=av_velocity,color="#d0417e",density=1.5)
alpha = 0.6
streamlines.lines.set_alpha(alpha)
streamlines.arrows.set_alpha(alpha)

plot.plot_vor(ax,sim.x_save[0].astype(np.float32),sim.t.tissue_params["L"],cols=plot.generate_ctype_cols(sim.t.c_types,c_type_col_map=["darkgrey", "lightgrey","lightgrey","white"]))
# ax.quiver(xx,yy,ret_x.statistic,ret_y.statistic)
for i in range(pos_tm1.shape[1]):
    ax.plot(pos_tm1[:100,i,0],pos_tm1[:100,i,1])
fig.show()
ax.axis("off")
fig.show()
fig.savefig("results/AVE_example/plots/streamplot.pdf")


# The ret_x_rotated_back and ret_y_rotated_back are the final statistics after the rotation, binned_statistic, and reverse translation.


#
# fig, ax = plt.subplots()
# # ax.quiver(sim.t.mesh.x[:,0],sim.t.mesh.x[:,1],sim.t.F[:,0]/np.linalg.norm(sim.t.F,axis=1),sim.t.F[:,1]/np.linalg.norm(sim.t.F,axis=1),color="red")
# ax.quiver(sim.t.mesh.x[:,0],sim.t.mesh.x[:,1],sim.t.active.active_params["v0"]*sim.t.active.orientation_vector[:,0],sim.t.active.active_params["v0"]*sim.t.active.orientation_vector[:,1],color="green")
# fig.show()
#
fig, ax = plt.subplots()
pos = spatial.displacements(sim.x_save.astype(np.float32),sim.t.mesh.L) + sim.x_save[0]
tmax = 800
pos_sample = pos[:tmax,sim.t.c_types==0]
ax.plot(pos_sample[:,:,0],pos_sample[:,:,1],color='black')
last_dir = pos_sample[-1] - pos_sample[-2]
last_dir = last_dir/np.expand_dims(np.linalg.norm(last_dir,axis=1),1)
ax.quiver(pos_sample[-1,:,0],pos_sample[-1,:,1],last_dir[:,0],last_dir[:,1],color='black')
ax.set(aspect=1)
ax.axis("off")
fig.show()
fig.savefig("results/AVE_example2/plots/tracks.pdf")

t_change = []
lost_edges = []
for t, (tri0,tri1) in enumerate(zip(sim.tri_save[:-1],sim.tri_save[1:])):
    if not (tri0 == tri1).all():
        le = get_lost_edges(tri0,tri1)
        if le is not None:
            lost_edges += [le]
            t_change += [t]

raw_vertex_positions = []
vertex_positions_normalised_to_AVE = []
ave_velocity = []
from synmorph.mesh import Mesh

tmin,tmax = 0,400
delT_vel = 10
d_max = 100
mask = sim.grn.ave_mask
for t,les in zip(t_change[1:],lost_edges[1:]):
    if (t > tmin)*(t<tmax):
        if t < sim.tri_save.shape[0] - delT_vel:
            x = sim.x_save[t]
            ave_x = x[sim.grn.ave_mask]
            ave_x_mean = np.mean(ave_x,axis=0)

            x_dt = sim.x_save[t+delT_vel]
            ave_x_dt = x_dt[sim.grn.ave_mask]
            ave_x_mean_dt = np.mean(ave_x_dt, axis=0)

            ave_vel = (ave_x_mean_dt - ave_x_mean)/delT_vel

            tri = sim.tri_save[t]
            # tri1 = sim.tri_save[t+1]
            # le = get_lost_edges(tri0,tri1)

            vs = Mesh(x, sim.t.mesh.L, tri, run_options=run_options).vs
            for le in les:
                    vtx_id0,vtx_id1 = np.where(((tri == le[0]) + (tri == le[1])).sum(axis=1) == 2)[0]
                    if not (mask[tri[vtx_id0]].any() + mask[tri[vtx_id1]].any()):
                        vtx = vs[vtx_id0]
                        vtx_norm = vtx - ave_x_mean
                        d = np.linalg.norm(vtx_norm)
                        if d <=d_max:
                            raw_vertex_positions += [vtx]
                            vertex_positions_normalised_to_AVE += [vtx_norm]
                            ave_velocity += [ave_vel]

vertex_positions_normalised_to_AVE = np.array(vertex_positions_normalised_to_AVE)
ave_angle = np.arctan2(np.array(ave_velocity)[:,1],np.array(ave_velocity)[:,0])
t1_angle = np.arctan2(vertex_positions_normalised_to_AVE[:,1],vertex_positions_normalised_to_AVE[:,0])

pts = np.array(vertex_positions_normalised_to_AVE)
plt.scatter(pts[:,0],pts[:,1])
plt.show()

normalised_angle = (t1_angle - ave_angle + np.pi)%(2*np.pi) - np.pi
#
t1_rad = np.linalg.norm(vertex_positions_normalised_to_AVE,axis=1)
#

def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


normalised_angle = (t1_angle - ave_angle + np.pi)%(2*np.pi) - np.pi

fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
circular_hist(ax[0],normalised_angle%(2*np.pi),gaps=False)
circular_hist(ax[1],t1_angle%(2*np.pi),gaps=False)

fig.show()

plt.show()
#
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
#
#
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

tmin = 0
tmax = 50

ret_x = binned_statistic_2d(pos_tm1[tmin:tmax,:,0].ravel(),pos_tm1[tmin:tmax,:,1].ravel(), velocity[tmin:tmax,:,0].ravel(), statistic=np.nanmean, bins=[x_bins, y_bins])
ret_y = binned_statistic_2d(pos_tm1[tmin:tmax,:,0].ravel(),pos_tm1[tmin:tmax,:,1].ravel(), velocity[tmin:tmax,:,1].ravel(), statistic=np.nanmean, bins=[x_bins, y_bins])

av_velocity = np.sqrt(ret_x.statistic**2 + ret_y.statistic**2)
av_velocity /= np.nanmean(av_velocity)

x_bin_centres = (x_bins[1:] + x_bins[:-1])/2
y_bin_centres = (y_bins[1:] + y_bins[:-1])/2

xx, yy = np.meshgrid(x_bin_centres, y_bin_centres)


vx,vy = ret_x.statistic.T,ret_y.statistic.T
n_vx,n_vy = vx/np.sqrt(vx**2 + vy**2),vy/np.sqrt(vx**2 + vy**2)

fig, ax = plt.subplots(figsize=(4,4))
# ax.set(aspect=1,xlim=(-L/2,L/2),ylim=(-L/2,L/2))
stream = ax.streamplot(xx,yy,n_vx,n_vy,color="#d0417e",density=1.6,linewidth=2)#start_points=
alpha = 0.6
stream.lines.set_alpha(alpha)
stream.arrows.set_alpha(alpha)

plot.plot_vor(ax,sim.x_save[25].astype(np.float32),sim.t.tissue_params["L"],cols=plot.generate_ctype_cols(sim.t.c_types,c_type_col_map=["#399cc3", "lightgrey","lightgrey","white"]))

# ax.quiver(xx,yy,ret_x.statistic,ret_y.statistic)
ax.axis("off")
fig.savefig("results/streamplot.pdf",dpi=300)
# fig.savefig("results/AVE_example2/plots/test.pdf")


i_range = np.arange(0,250,62.5).astype(int)
fig, ax = plt.subplots(1,len(i_range),figsize=(20,20))

for j, i in enumerate(i_range):
    plot.plot_vor(ax[j],sim.x_save[i].astype(np.float32),sim.t.tissue_params["L"],cols=plot.generate_ctype_cols(sim.t.c_types,c_type_col_map=["#399cc3", "lightgrey","lightgrey","white"]))
fig.savefig("results/dynamics.pdf",dpi=300)


