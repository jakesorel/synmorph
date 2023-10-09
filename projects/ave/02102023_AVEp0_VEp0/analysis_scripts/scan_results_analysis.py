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
new_index = pd.RangeIndex(40000)

# Reindex the DataFrame, filling missing values with NaN
df = df.reindex(new_index)

max_AVE_distance = df["max_AVE_distance"].values.reshape(20,20,100)
# t_95_distance = df["t_95_distance"].values.reshape(20,20,100)
max_AVE_distance = max_AVE_distance[:18,:18]

AVE_contiguous = df["AVE_contiguous"].values.reshape(20,20,100)
AVE_contiguous = AVE_contiguous[:18,:18]

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
Gamma_AVE_range = np.flip(-AVE_p0_range*2*lambda_P)*0.73484695
Gamma_AVE_range = Gamma_AVE_range[:18]
Gamma_VE_range = np.flip(-VE_p0_range*2*lambda_P)*1.0379211
Gamma_VE_range = Gamma_VE_range[:18]



x,y = np.meshgrid(AVE_p0_range,VE_p0_range)


extent,aspect = make_extent(Gamma_AVE_range,Gamma_VE_range,xscale="linear",yscale="linear")
fig, ax = plt.subplots(figsize=(5,5))
vmin=0
vmax=4.0
ax.imshow(np.flip(np.nanmean(max_AVE_distance,axis=-1).T,axis=1),interpolation="nearest",cmap="inferno",extent=extent,aspect=aspect,vmin=vmin,vmax=vmax)#,vmin=0,vmax=1)
ax.set(xlabel="AVE Line Tension\n"r"$\Gamma_{A}$",ylabel="VE Line Tension\n"r"$\Gamma_{V}$")
sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmax=vmax,vmin=vmin))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
cl.set_label("AVE Displacement\n(Cell Diameters)")
fig.subplots_adjust(left=0.3,right=0.7,bottom=0.3,top=0.8)
fig.savefig("projects/ave/02102023_AVEp0_VEp0/plots/AVE displacement.pdf",dpi=300)


extent,aspect = make_extent(Gamma_AVE_range,Gamma_VE_range,xscale="linear",yscale="linear")
fig, ax = plt.subplots(figsize=(5,5))
vmin=0
vmax=30
im = np.flip(np.nanmean((AVE_contiguous!=1)*AVE_contiguous/AVE_contiguous,axis=-1).T,axis=1)*100
# im[np.isnan(im)] = 0
ax.imshow(im,interpolation="nearest",cmap="magma",extent=extent,aspect=aspect,vmin=vmin,vmax=vmax)#,vmin=0,vmax=1)
ax.set(xlabel="AVE Line Tension\n"r"$\Gamma_{A}$",ylabel="VE Line Tension\n"r"$\Gamma_{V}$")
sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(vmax=vmax,vmin=vmin))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
cl.set_label("Frequency of\nAVE splitting (%)")
fig.subplots_adjust(left=0.3,right=0.7,bottom=0.3,top=0.8)
fig.savefig("projects/ave/02102023_AVEp0_VEp0/plots/frequency breakup.pdf",dpi=300)



#
# extent,aspect = make_extent(Gamma_AVE_range,Gamma_VE_range,xscale="linear",yscale="linear")
# fig, ax = plt.subplots(figsize=(5,5))
# vmin=1
# vmax=np.flip(np.nanmean(AVE_contiguous,axis=-1).T,axis=1).max()
# ax.imshow(np.flip(np.nanmean(AVE_contiguous,axis=-1).T,axis=1),interpolation="nearest",cmap="Reds",extent=extent,aspect=aspect,vmin=vmin,vmax=vmax)#,vmin=0,vmax=1)
# ax.set(xlabel="AVE Line Tension\n"r"$\Gamma_{A}$",ylabel="VE Line Tension\n"r"$\Gamma_{V}$")
# sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmax=vmax,vmin=vmin))
# fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
# cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
# cl.set_label("AVE Clusters")
# fig.show()



df_fluid = pd.read_csv("projects/ave/02102023_AVEp0_VEp0/analysis_results/fluidity.csv",header=None)
pair_labels = ["t1_rate_"+"-".join(val) for val in
               np.mgrid[:4, :4].reshape(2, -1).T.astype(str)]
df_fluid.columns = ["index","D","D_0","D_1","D_2","D_3","t1_rate"] + pair_labels


# df.columns = "index,max_AVE_distance,percentile95_distance,t_95_distance,AVE_contiguous".split(",")
df_fluid["index"] = df_fluid["index"].astype(int)
df_fluid.index = df_fluid["index"]
df_fluid = df_fluid.sort_index()
# Create a new index with missing values from 0 to 7999
new_index = pd.RangeIndex(40000)

# Reindex the DataFrame, filling missing values with NaN
df_fluid = df_fluid.reindex(new_index)


# AVE_D = df_fluid["D_0"].values.reshape(20,20,100)
t1_AVE = df_fluid["t1_rate_0-0"].values.reshape(20,20,100)
t1_AVE = t1_AVE[:18,:18]

extent,aspect = make_extent(Gamma_AVE_range,Gamma_VE_range,xscale="linear",yscale="linear")
fig, ax = plt.subplots(figsize=(5,5))

im = np.log10(np.flip(np.nanmean(t1_AVE,axis=-1).T,axis=1))
im[np.isnan(im)+np.isinf(im)] = np.nanmin(im[~np.isinf(im)])
vmin=-1
vmax = 0.1
ax.imshow(im,interpolation="nearest",cmap="cividis",extent=extent,aspect=aspect,vmin=vmin,vmax=vmax)#,vmin=0,vmax=1)
ax.set(xlabel="AVE Line Tension\n"r"$\Gamma_{A}$",ylabel="VE Line Tension\n"r"$\Gamma_{V}$")
sm = plt.cm.ScalarMappable(cmap="cividis", norm=plt.Normalize(vmax=vmax,vmin=vmin))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=18, orientation="vertical")
cl.set_label("AVE Neighbour\nExchange Rate")
cl.set_ticks([np.log10(0.1),np.log10(0.25),np.log10(0.5),np.log10(1)])
cl.set_ticklabels([r"$\leq 0.1$",0.25,0.5,1])

fig.subplots_adjust(left=0.3,right=0.7,bottom=0.3,top=0.8)
fig.show()
fig.savefig("projects/ave/02102023_AVEp0_VEp0/plots/fluidity.pdf",dpi=300)



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


fig, ax = plt.subplots(1,4)
ax[0].imshow(np.flip(np.nanmean(t1_AVE,axis=-1).T,axis=1),vmin=0.1,vmax=0.11)
ax[1].imshow(np.flip(np.nanmean(AVE_contiguous,axis=-1).T,axis=1))
ax[2].imshow(np.flip(np.nanmean(max_AVE_distance,axis=-1).T,axis=1))
arr = np.sum((t1_AVE<=0.25),axis=-1)/np.nansum(~np.isnan(t1_AVE),axis=-1)
ax[3].imshow(np.flip(arr.T,axis=1))

fig.show()

fig, ax = plt.subplots(figsize=(4,4))
# arr = np.sum((t1_AVE<=0.25),axis=-1)/np.nansum(~np.isnan(t1_AVE),axis=-1)
arr = np.nanmean(t1_AVE,axis=-1)
x = Gamma_AVE_range
y = Gamma_VE_range
x_ex = np.linspace(x.min(), x.max(), t1_AVE.shape[0]*10)
y_ex = np.linspace(y.min(), y.max(), t1_AVE.shape[1]*10)

t1_x_ev, t1_y_ev = convert_contour_value(x_ex, y_ex, find_contours_expand(np.flip(np.flip(arr,axis=1),axis=0), level=0.1,sigma=10), arr.shape[0]).T

ax.plot(t1_x_ev, t1_y_ev,color="black")

# fig.show()
arr = np.nanmean((max_AVE_distance>2)*max_AVE_distance/max_AVE_distance,axis=-1)
#
d_x_ev, d_y_ev = convert_contour_value(x_ex, y_ex, find_contours_expand(np.flip(np.flip(arr,axis=1),axis=0), level=0.5,sigma=15), arr.shape[0]).T
ax.plot(d_x_ev, d_y_ev,color="black")

# plt.plot(*find_contours_expand(np.nanmean(AVE_contiguous,axis=-1)[-1], level=2).T)
ax.set(xlim=(Gamma_AVE_range.min(),Gamma_AVE_range.max()),ylim=(Gamma_VE_range.min(),Gamma_VE_range.max()))
ax.set(xlabel="AVE Line Tension\n"r"$\Gamma_{A}$",ylabel="VE Line Tension\n"r"$\Gamma_{V}$")
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)

fig.savefig("projects/ave/02102023_AVEp0_VEp0/plots/phase_diagram_solid_flock.pdf",dpi=300)


### T1 vs breakup

fig, ax = plt.subplots(figsize=(4,3))
x,y = np.log10(np.nanmean(t1_AVE,axis=-1)),np.nanmean((AVE_contiguous!=1)*AVE_contiguous/AVE_contiguous,axis=-1)
x = np.flip(np.flip(x,axis=0),axis=1)
y = np.flip(np.flip(y,axis=0),axis=1)*100

ax.scatter(x,y,s=5,color=plt.cm.inferno(0.2))

ax.spines[['right', 'top']].set_visible(False)
ax.set_xticks([-4,-2,0],labels=[r"$10^{%d}$"%i for i in [-4,-2,0]])
ax.set(xlabel="AVE Neighbour\nExchange Rate")
ax.set(ylabel="Frequency of\nAVE splitting (%)")

fig.subplots_adjust(bottom=0.3,left=0.3,right=0.7,top=0.7)
fig.savefig("projects/ave/02102023_AVEp0_VEp0/plots/neighbour exchange vs splitting.pdf",dpi=300)

###Summary plots
freq_split = np.nanmean((AVE_contiguous!=1)*AVE_contiguous/AVE_contiguous,axis=-1)

x,y,z = np.meshgrid(np.flip(Gamma_AVE_range),np.flip(Gamma_VE_range),np.arange(100),indexing="ij")
df_total = pd.DataFrame({"d":max_AVE_distance.ravel(),"t1":t1_AVE.ravel(),"Gamma_AVE":x.ravel(),"Gamma_VE":y.ravel(),"seed":z.ravel()})


fig, ax = plt.subplots(figsize=(5,2.5))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
# ax.plot(np.nanmean(t1_AVE,axis=-1)[-1,:,2])
sns.lineplot(data=df_total[(df_total["Gamma_VE"]==Gamma_VE_range[4])],x="Gamma_AVE",y="t1",ax=ax,
             color="#d0417e",label="AVE "r"$p_0$""=%.1f"%AVE_p0_range[10])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend().set_visible(False)
ax.set(xlim=(Gamma_AVE_range.min(),Gamma_AVE_range.max()))
fig.savefig("projects/ave/02102023_AVEp0_VEp0/plots/t1 rate for phase diagram Gamma AVE.pdf",dpi=300)


fig, ax = plt.subplots(figsize=(5,2.5))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
# ax.plot(np.nanmean(t1_AVE,axis=-1)[-1,:,2])
sns.lineplot(data=df_total[(df_total["Gamma_AVE"]==Gamma_AVE_range[4])],x="Gamma_VE",y="t1",ax=ax,
             color="#d0417e",label="AVE "r"$p_0$""=%.1f"%AVE_p0_range[10])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend().set_visible(False)
ax.set(xlim=(Gamma_VE_range.min(),Gamma_VE_range.max()))
fig.savefig("projects/ave/02102023_AVEp0_VEp0/plots/t1 rate for phase diagram.pdf",dpi=300)


fig, ax = plt.subplots(figsize=(5,2.5))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
sns.lineplot(data=df_total[(df_total["Gamma_AVE"]==Gamma_AVE_range[4])],x="Gamma_VE",y="d",ax=ax,
             color="#08647c",label="AVE "r"$p_0$""=%.1f"%AVE_p0_range[10])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend().set_visible(False)
ax.set(xlim=(Gamma_VE_range.min(),Gamma_VE_range.max()),ylim=(0,4.3))
fig.savefig("projects/ave/02102023_AVEp0_VEp0/plots/d rate for phase diagram.pdf",dpi=300)




fig, ax = plt.subplots(figsize=(5,2.5))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
# ax.plot(np.nanmean(t1_AVE,axis=-1)[-1,:,2])
ax.scatter(Gamma_AVE_range,np.flip(freq_split[:,-4]),color=plt.cm.inferno(0.2))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set(xlabel="AVE Gamma",ylabel="Freq breakup")
ax.legend().set_visible(False)
fig.savefig("projects/ave/02102023_AVEp0_VEp0/plots/Freq breakup for phase diagram AVE.pdf",dpi=300)

fig, ax = plt.subplots(figsize=(5,2.5))
fig.subplots_adjust(bottom=0.3, top=0.8, left=0.3, right=0.8, wspace=0.7)
# ax.plot(np.nanmean(t1_AVE,axis=-1)[-1,:,2])
ax.scatter(Gamma_VE_range,np.flip(freq_split[-4]),color=plt.cm.inferno(0.2))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set(xlabel="VE Gamma",ylabel="Freq breakup")
ax.legend().set_visible(False)
fig.savefig("projects/ave/02102023_AVEp0_VEp0/plots/Freq breakup for phase diagram VE.pdf",dpi=300)



### Average breakup sorting and not

from scipy.spatial import ConvexHull


x,y = t1_x_ev, t1_y_ev
x = np.concatenate((t1_x_ev,(-2,)))
y = np.concatenate((t1_y_ev,(-2,)))

t1_points = np.array((x,y)).T

x = np.concatenate((d_x_ev,(-2,2)))
y = np.concatenate((d_y_ev,(-2,-2)))

d_points = np.array((x,y)).T


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

X,Y = np.meshgrid(Gamma_AVE_range,Gamma_VE_range,indexing="ij")
Pts_flat = np.array((X.ravel(),Y.ravel())).T

in_hull_t1 = np.array([in_hull(pt,t1_points) for pt in Pts_flat])
in_hull_d = np.array([in_hull(pt,d_points) for pt in Pts_flat])

solid = (~in_hull_t1)*(~in_hull_d)
fluid = (in_hull_t1)*(~in_hull_d)
solid_flock = (~in_hull_t1)*(in_hull_d)
fluid_flock = (in_hull_t1)*(in_hull_d)

plt.imshow(np.flip(in_hull_t1.reshape(18,18).T,axis=0))
# plt.imshow(np.flip(np.nanmean((AVE_contiguous!=1)*AVE_contiguous/AVE_contiguous,axis=-1).T,axis=1))
plt.imshow(np.log10(np.flip(np.nanmean(t1_AVE,axis=-1).T,axis=1)))
plt.show()

im = np.flip(np.nanmean((AVE_contiguous!=1)*AVE_contiguous/AVE_contiguous,axis=-1).T,axis=1)



df_dict = {"Class":[],"Frequency AVE Splitting (%)":[]}
for i in range(len(freq_split.ravel())):
    f_split = np.flip(np.nanmean((AVE_contiguous!=1)*AVE_contiguous/AVE_contiguous,axis=-1).T,axis=1).ravel()[i]*100
    classification = ""
    if np.flip(in_hull_t1.reshape(18,18).T,axis=0).ravel()[i]:
        classification += "Fluid"
    else:
        classification += "Solid"
    if np.flip(in_hull_d.reshape(18,18).T,axis=0).ravel()[i]:
        classification += "\nFlock"
    df_dict["Frequency AVE Splitting (%)"] += [f_split]
    df_dict["Class"] += [classification]

df_class = pd.DataFrame(df_dict)

fig, ax = plt.subplots(figsize=(3,4))
sns.barplot(data=df_class,x="Class",y="Frequency AVE Splitting (%)",order = ["Solid","Fluid","Solid\nFlock","Fluid\nFlock"],
            palette=["#646f87","#cad4e3","#b16e69","#edd5c3"])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend().set_visible(False)
ax.set(xlabel="")
fig.subplots_adjust(bottom=0.3,left=0.3,right=0.8,top=0.8)
fig.savefig("projects/ave/02102023_AVEp0_VEp0/plots/AVE_splitting rates.pdf",dpi=300)
