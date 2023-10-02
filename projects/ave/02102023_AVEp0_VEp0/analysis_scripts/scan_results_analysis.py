import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import mcubes
import plotly.graph_objects as go
import seaborn as sns
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = "Helvetica Neue"


df = pd.read_csv("projects/ave/02102023_AVEp0_VEp0/analysis_results/minimal.csv",header=None)
df.columns = "index,max_AVE_distance,percentile95_distance,t_95_distance,AVE_contiguous".split(",")
df["index"] = df["index"].astype(int)
df.index = df["index"]
df = df.sort_index()
# Create a new index with missing values from 0 to 7999
new_index = pd.RangeIndex(8000)

# Reindex the DataFrame, filling missing values with NaN
df = df.reindex(new_index)

max_AVE_distance = df["max_AVE_distance"].values.reshape(20,20,20)
t_95_distance = df["t_95_distance"].values.reshape(20,20,20)

AVE_contiguous = df["AVE_contiguous"].values.reshape(20,20,20)

AVE_is_contiguous = np.nanmean(AVE_contiguous,axis=-1)<1.5
av_max_AVE_distance = np.nanmean(max_AVE_distance,axis=-1)
AVE_contiguous_and_moved = (av_max_AVE_distance>2)*AVE_is_contiguous



def make_extent(x_range,y_range,xscale="linear",yscale="linear"):
    if xscale == "log":
        x_range = np.log10(x_range)
    if yscale == "log":
        y_range = np.log10(y_range)
    extent = [x_range[0],x_range[-1],y_range[0],y_range[-1]]
    aspect = (extent[1]-extent[0])/(extent[3]-extent[2])
    return extent,aspect

"""
E = ... + lambdaP*(P-P0)^2 
E = ... + lambdaP*P*^2 - 2*lambdaP*P0*P
"""


N = 20
lambda_P = 0.2
W01_range = np.linspace(0, 0.1, N)
AVE_p0_range = np.linspace(3.4, 4.5, N)
VE_p0_range = np.linspace(3.4, 4.5, N)
Gamma_AVE_range = AVE_p0_range*2*lambda_P
Gamma_VE_range = VE_p0_range*2*lambda_P

x,y = np.meshgrid(AVE_p0_range,VE_p0_range)


extent,aspect = make_extent(Gamma_AVE_range,Gamma_VE_range,xscale="linear",yscale="linear")
fig, ax = plt.subplots(figsize=(5,5))
vmin=0
vmax=4.0
ax.imshow(np.flip(np.nanmean(max_AVE_distance,axis=-1).T,axis=0),interpolation="nearest",cmap="inferno",extent=extent,aspect=aspect,vmin=vmin,vmax=vmax)#,vmin=0,vmax=1)
ax.set(xlabel="AVE Line Tension\n"r"$\Gamma_{A}$",ylabel="VE Line Tension\n"r"$\Gamma_{V}$")
sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmax=vmax,vmin=vmin))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
cl.set_label("AVE Displacement\n(Cell Diameters)")
fig.subplots_adjust(left=0.3,right=0.7,bottom=0.3,top=0.8)
fig.show()


extent,aspect = make_extent(Gamma_AVE_range,Gamma_VE_range,xscale="linear",yscale="linear")
fig, ax = plt.subplots(figsize=(5,5))
vmin=0
vmax=0.5
ax.imshow(np.flip(np.nanmean((AVE_contiguous!=1)*AVE_contiguous/AVE_contiguous,axis=-1).T,axis=0),interpolation="nearest",cmap="inferno",extent=extent,aspect=aspect,vmin=vmin,vmax=vmax)#,vmin=0,vmax=1)
ax.set(xlabel="AVE Line Tension\n"r"$\Gamma_{A}$",ylabel="VE Line Tension\n"r"$\Gamma_{V}$")
sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmax=vmax,vmin=vmin))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
cl.set_label("AVE Displacement\n(Cell Diameters)")
fig.subplots_adjust(left=0.3,right=0.7,bottom=0.3,top=0.8)
fig.show()


extent,aspect = make_extent(Gamma_AVE_range,Gamma_VE_range,xscale="linear",yscale="linear")
fig, ax = plt.subplots(figsize=(5,5))
vmin=1
vmax=2
ax.imshow(np.flip(np.nanmean(AVE_contiguous,axis=-1)[0].T,axis=0),interpolation="nearest",cmap="inferno",extent=extent,aspect=aspect,vmin=vmin,vmax=vmax)#,vmin=0,vmax=1)
ax.set(xlabel="AVE Line Tension\n"r"$\Gamma_{A}$",ylabel="VE Line Tension\n"r"$\Gamma_{V}$")
sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmax=vmax,vmin=vmin))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
cl.set_label("AVE Clusters")
fig.show()


extent,aspect = make_extent(W01_range,Gamma_VE_range,xscale="linear",yscale="linear")
fig, ax = plt.subplots(figsize=(5,5))
vmin=0
vmax=4.0
ax.imshow(np.flip(np.nanmean(max_AVE_distance,axis=-1)[:,0].T,axis=0),interpolation="nearest",cmap="inferno",extent=extent,aspect=aspect,vmin=vmin,vmax=vmax)#,vmin=0,vmax=1)
ax.set(xlabel="Differential Tension\n"r"$\Gamma_{A,V}$",ylabel="VE Line Tension\n"r"$\Gamma_{V}$")
# ax.set_xticks([-3,-2,-1],labels=[r"$10^{%d}$"%i for i in [-3,-2,-1]])

sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmax=vmax,vmin=vmin))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
cl.set_label("AVE Displacement\n(Cell Diameters)")
fig.subplots_adjust(left=0.3,right=0.7,bottom=0.3,top=0.8)
fig.savefig("projects/ave/01102023_W01_AVEp0_VEp0/plots/Diff add vs VEp0 AVE_disp.pdf",dpi=300)



df_fluid = pd.read_csv("projects/ave/22072023_W01_AVEp0_VEp0/analysis_results/fluidity.csv",header=None)
pair_labels = ["t1_rate_"+"-".join(val) for val in
               np.mgrid[:4, :4].reshape(2, -1).T.astype(str)]
df_fluid.columns = ["index","D","D_0","D_1","D_2","D_3","t1_rate"] + pair_labels


# df.columns = "index,max_AVE_distance,percentile95_distance,t_95_distance,AVE_contiguous".split(",")
df_fluid["index"] = df_fluid["index"].astype(int)
df_fluid.index = df_fluid["index"]
df_fluid = df_fluid.sort_index()

AVE_D = df_fluid["D_0"].values.reshape(20,20,20,20)
t1_AVE = df_fluid["t1_rate_0-0"].values.reshape(20,20,20,20)


extent,aspect = make_extent(Gamma_AVE_range,Gamma_VE_range,xscale="linear",yscale="linear")
fig, ax = plt.subplots(figsize=(5,5))
vmin=0
vmax=3.5
ax.imshow(np.flip(np.nanmean(t1_AVE,axis=-1)[-1].T,axis=0),interpolation="nearest",cmap="viridis_r",extent=extent,aspect=aspect,vmin=vmin,vmax=vmax)#,vmin=0,vmax=1)
ax.set(xlabel="AVE Line Tension\n"r"$\Gamma_{A}$",ylabel="VE Line Tension\n"r"$\Gamma_{V}$")
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=plt.Normalize(vmax=vmax,vmin=vmin))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
cl.set_label("AVE Neighbour\nExchange Rate")

fig.subplots_adjust(left=0.3,right=0.7,bottom=0.3,top=0.8)
fig.show()
fig.savefig("projects/ave/01102023_W01_AVEp0_VEp0/plots/p0s neighbour exchange rate.pdf")

def convert_contour_value(x_ex,y_ex,contour,N):
    x_eval = x_ex.min() + (x_ex.max()-x_ex.min())*contour[:,0]/N
    y_eval = y_ex.min() + (y_ex.max()-y_ex.min())*contour[:,1]/N
    return np.column_stack((x_eval,y_eval))
from skimage.measure import find_contours
from scipy.interpolate import splprep, splev
def find_contours_expand(arr,level=1,sigma=10):
    arr_large = resize(arr,np.array(arr.shape) * 10)
    smoothed = gaussian_filter(arr_large,sigma)
    return find_contours(smoothed,level=level)[-1]/10



lightblue = "#adbdd3"
darkblue = "#113a58"
sand = "#e1bea2"
red = "#8c2231"

fig, ax = plt.subplots(figsize=(4,4))
arr = np.nanmean(t1_AVE,axis=-1)[-1]
x = W01_range
y = Gamma_AVE_range
z = Gamma_VE_range
x_ex = np.linspace(x.min(), x.max(), t1_AVE.shape[0]*10)
y_ex = np.linspace(y.min(), y.max(), t1_AVE.shape[1]*10)
z_ex = np.linspace(z.min(), z.max(), t1_AVE.shape[2]*10)

t1_y_ev, t1_z_ev = convert_contour_value(y_ex, z_ex, find_contours_expand(arr, level=3), arr.shape[0]).T

ax.plot(t1_y_ev, t1_z_ev,color="black")

arr = np.nanmean(max_AVE_distance,axis=-1)[-1]

d_y_ev, d_z_ev = convert_contour_value(y_ex, z_ex, find_contours_expand(arr, level=2), arr.shape[0]).T
ax.plot(d_y_ev, d_z_ev,color="black")

# plt.plot(*find_contours_expand(np.nanmean(AVE_contiguous,axis=-1)[-1], level=2).T)
ax.set(xlim=(Gamma_AVE_range.min(),Gamma_AVE_range.max()),ylim=(Gamma_AVE_range.min(),Gamma_AVE_range.max()))
ax.set(xlabel="AVE Line Tension\n"r"$\Gamma_{A}$",ylabel="VE Line Tension\n"r"$\Gamma_{V}$")
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)

fig.savefig("projects/ave/01102023_W01_AVEp0_VEp0/plots/phase_diagram_solid_flock.pdf",dpi=300)

w,x,y,z = np.meshgrid(W01_range,Gamma_AVE_range,Gamma_VE_range,np.arange(20),indexing="ij")
df_total = pd.DataFrame({"d":max_AVE_distance.ravel(),"t1":t1_AVE.ravel(),"N_clust":AVE_contiguous.ravel(),"w":w.ravel(),"Ap0":x.ravel(),"Vp0":y.ravel(),"seed":z.ravel()})


fig, ax = plt.subplots(figsize=(5,2.5))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
# ax.plot(np.nanmean(t1_AVE,axis=-1)[-1,:,2])
sns.lineplot(data=df_total[(df_total["w"]==w.max())*(df_total["Vp0"]==Gamma_VE_range[-6])],x="Ap0",y="t1",ax=ax,
             color="#d0417e",label="AVE "r"$p_0$""=%.1f"%AVE_p0_range[10])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend().set_visible(False)
ax.set(xlim=(Gamma_VE_range.min(),Gamma_VE_range.max()))
fig.savefig("projects/ave/01102023_W01_AVEp0_VEp0/plots/t1 rate for phase diagram.pdf",dpi=300)

fig, ax = plt.subplots(figsize=(5,2.5))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
sns.lineplot(data=df_total[(df_total["w"]==w.max())*(df_total["Ap0"]==Gamma_VE_range[2])],x="Vp0",y="d",ax=ax,
             color="#08647c",label="AVE "r"$p_0$""=%.1f"%AVE_p0_range[10])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend().set_visible(False)
ax.set(xlim=(Gamma_VE_range.min(),Gamma_VE_range.max()),ylim=(0,4))
fig.savefig("projects/ave/01102023_W01_AVEp0_VEp0/plots/d rate for phase diagram.pdf",dpi=300)


fig, ax = plt.subplots(figsize=(2.5,2.5))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
# ax.plot(np.nanmean(t1_AVE,axis=-1)[-1,:,2])
sns.lineplot(data=df_total[(df_total["Ap0"]==Gamma_VE_range[2])*(df_total["Vp0"]==Gamma_VE_range[-6])],x="w",y="N_clust",ax=ax,
             color="#505252",label="AVE "r"$p_0$""=%.1f"%AVE_p0_range[10])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend().set_visible(False)
ax.set(ylim=(0,None))
fig.savefig("projects/ave/01102023_W01_AVEp0_VEp0/plots/N_clust for phase diagram.pdf",dpi=300)



def plot_surface(volume, x,y,z, smooth=False,plot_contours=False):
    volume_resized = resize(volume.astype(float), np.array(volume.shape) * 10)
    smoothed = gaussian_filter(volume_resized,10)
    volume_smoothed = smoothed > 0.5

    x_ex = np.linspace(x.min(),x.max(),volume_smoothed.shape[0])
    y_ex = np.linspace(y.min(),y.max(),volume_smoothed.shape[1])
    z_ex = np.linspace(z.min(),z.max(),volume_smoothed.shape[2])

    # Extract the surface using Marching Cubes algorithm.
    if smooth:
        verts, faces = mcubes.marching_cubes(mcubes.smooth(volume_smoothed), 0.85)
    else:
        verts, faces = mcubes.marching_cubes(volume_smoothed, 0.85)

    # Calculate the data range for each axis
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()

    # Determine the minimum range among all three axes
    min_range = min(x_range, y_range, z_range)

    # Scale the vertices using the minimum range
    verts_scaled = np.array(verts)
    verts_scaled[:, 0] = x_range * verts[:, 0] / volume_smoothed.shape[0] + x.min()
    verts_scaled[:, 1] = y_range * verts[:, 1] / volume_smoothed.shape[1] + y.min()
    verts_scaled[:, 2] = z_range * verts[:, 2] / volume_smoothed.shape[2] + z.min()

    # Convert the faces array to integers.
    faces = faces.astype(int)

    # Create the figure for the interactive 3D plot.
    fig = go.Figure()

    # # Add the surface plot to the plot.
    # surface = go.Surface(x=verts_scaled[:, 0], y=verts_scaled[:, 1], z=verts_scaled[:, 2], colorscale='Viridis')
    # fig.add_trace(surface)

    # Add the mesh plot on top of the surface.
    mesh = go.Mesh3d(x=verts_scaled[:, 0], y=verts_scaled[:, 1], z=verts_scaled[:, 2],
                     i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], opacity=1,color='rgb(169,5,5)')

    fig.add_trace(mesh)

    def is_corner_point(boundary_points_sorted):
        mask = (boundary_points_sorted == 0)+(boundary_points_sorted == volume_smoothed.shape[0]-1)
        mask = (mask.sum(axis=1) ==2)
        return boundary_points_sorted[~mask]

    if plot_contours:

        for i in range(0,volume_smoothed.shape[0],20):
            arr = volume_smoothed[i]
            boundary_points = np.argwhere(arr)
            if len(boundary_points)>0:


                contours = find_contours(arr, level=0.5)
                # Get the outer boundary (largest contour)
                if len(contours)>0:
                    outer_boundary = max(contours, key=len)
                    tck, u = splprep([outer_boundary[:, 0], outer_boundary[:, 1]], s=10)
                    u_new = np.linspace(u.min(), u.max(), 500)
                    x_new, y_new = splev(u_new, tck)
                    boundary_points_sorted = np.column_stack((x_new,y_new))
                    y_ev,z_ev = convert_contour_value(y_ex,z_ex,boundary_points_sorted,arr.shape[0]).T
                    fig.add_trace(go.Scatter3d(x=np.repeat(x_ex[i],len(boundary_points_sorted)),
                                               y=y_ev,
                                               z=z_ev,
                                               mode='lines', line=dict(color='white', width=5)))



        for i in range(0,volume_smoothed.shape[1],20):
            arr = volume_smoothed[:,i]
            boundary_points = np.argwhere(arr)
            if len(boundary_points)>0:


                contours = find_contours(arr, level=0.5)
                # Get the outer boundary (largest contour)
                if len(contours)>0:
                    outer_boundary = max(contours, key=len)
                    tck, u = splprep([outer_boundary[:, 0], outer_boundary[:, 1]], s=10)
                    u_new = np.linspace(u.min(), u.max(), 500)
                    x_new, y_new = splev(u_new, tck)
                    boundary_points_sorted = np.column_stack((x_new,y_new))
                    x_ev,z_ev = convert_contour_value(x_ex,z_ex,boundary_points_sorted,arr.shape[0]).T
                    fig.add_trace(go.Scatter3d(x=x_ev,
                                               y=np.repeat(y_ex[i],len(boundary_points_sorted)),
                                               z=z_ev,
                                               mode='lines', line=dict(color='white', width=5)))


        for i in range(0,volume_smoothed.shape[2],20):
            arr = volume_smoothed[:,:,i]
            boundary_points = np.argwhere(arr)
            if len(boundary_points)>0:


                contours = find_contours(arr, level=0.5)
                if len(contours)>0:

                    # Get the outer boundary (largest contour)
                    outer_boundary = max(contours, key=len)
                    tck, u = splprep([outer_boundary[:, 0], outer_boundary[:, 1]], s=10)
                    u_new = np.linspace(u.min(), u.max(), 500)
                    x_new, y_new = splev(u_new, tck)
                    boundary_points_sorted = np.column_stack((x_new,y_new))
                    x_ev,y_ev = convert_contour_value(x_ex,y_ex,boundary_points_sorted,arr.shape[0]).T

                    fig.add_trace(go.Scatter3d(x=x_ev,
                                               y=y_ev,
                                               z=np.repeat(z_ex[i],len(boundary_points_sorted)),
                                               mode='lines', line=dict(color='white', width=5)))



    # Create a Plotly figure
    # fig = go.Figure()

    # Add the square to the figure


            # Set the axis labels and title.

    fig.update_layout(scene_aspectmode='manual',
                      scene_aspectratio=dict(x=1, y=1, z=1))  # Adjust the x value to make the x-axis wider

    # Set lighting and shading
    # fig.update_traces(lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.2))

    # Show contour lines
    # contours = go.surface.Contours(z=dict(show=True, start=verts_scaled[:, 2].min(), end=verts_scaled[:, 2].max(), size=0.5))
    # fig.update_traces(contours=contours)
    # Set the axis labels and title.

    # Show the interactive 3D plot.
    return fig

# # Define the volume as provided
# x = np.linspace(0, 2, 50)
# y = np.linspace(0, 2, 50)
# z = np.linspace(0, 2, 50)
# X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
# volume = (np.sqrt(X**2 + Y**2 + Z**2) <= 1) + (np.sqrt((X - 2)**2 + Y**2 + Z**2) <= 1)


is_solid = np.nanmean(t1_AVE,axis=-1)<3
AVE_is_contiguous = np.nanmean(AVE_contiguous,axis=-1)<2
av_max_AVE_distance = np.nanmean(max_AVE_distance,axis=-1)
AVE_moved = av_max_AVE_distance > 2
AVE_contiguous_and_moved = AVE_moved*AVE_is_contiguous
solid_flock = AVE_moved*AVE_is_contiguous*is_solid

N = 20
# W01_range = np.linspace(-3, -1, N)
# AVE_p0_range = np.linspace(3.4, 5, N)
# VE_p0_range = np.linspace(3.4, 5, N)
# Plot the surface of the volume interactively with custom tick values and labels.
fig = plot_surface(solid_flock.transpose(0,2,1), W01_range, AVE_p0_range, VE_p0_range, True,False)
fig.update_layout(scene=dict(xaxis_title="Differential tension", yaxis_title="VE p0",zaxis_title="AVE p0"),
                  title="Approximated Surface of Filled Volume")

fig.update_layout(
    scene=dict(
        xaxis=dict(range=[W01_range.min(), W01_range.max()]),   # Set x-axis range
        yaxis=dict(range=[AVE_p0_range.min(), AVE_p0_range.max()]),  # Set y-axis range
        zaxis=dict(range=[VE_p0_range.min()-0.1, VE_p0_range.max()])    # Set z-axis range
    )
)
fig.show()

