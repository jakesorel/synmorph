import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

files = []
for fl in sorted(os.listdir("projects/ave/22072023_W01_AVEp0_VEp0/analysis_results/compiled/selected/by_position_class/W01minAVEmaxVEmax")):
    if ".csv" in fl:
        files.append(fl)

df = pd.DataFrame()
for i, file in enumerate(files):
    dfi = pd.read_csv("projects/ave/22072023_W01_AVEp0_VEp0/analysis_results/compiled/selected/by_position_class/W01minAVEmaxVEmax/"+file)
    dfi["index"] = i
    df = pd.concat((df,dfi[dfi["t"] == 120.]))

values = {}
for key in df.columns:
    values[key] = df[key].values.reshape(20,16)




def plot_polar_bar(vals,R,ax,cmap,vmin=0,vmax=1,zorder=0):
    theta = np.linspace(0.0, 2 * np.pi, len(vals), endpoint=False)
    width = np.pi*2/len(vals)
    radii = np.ones_like(vals)*R
    if isinstance(ax, plt.PolarAxes):
        ax_new = ax
    else:
        # Get the position of the current axis
        pos = ax.get_position()

        # Remove the current axis
        ax.remove()
        # Create a new polar axis in the same position
        ax_new = plt.gcf().add_axes(pos, projection='polar')
    bars = ax_new.bar(theta, radii, width=width, bottom=0.0,zorder=zorder)
    for i, (r, bar) in enumerate(zip(theta, bars)):
        bar.set_facecolor(cmap((vals[i]-vmin)/(vmax-vmin)))
        bar.set_alpha(0.8)

    return ax_new

def make_polar_heatmap(ax,vals,cmap,label="Feature",alpha_percentile=0):
    vmin,vmax = np.percentile(vals,alpha_percentile),np.percentile(vals,100-alpha_percentile)
    ax = plot_polar_bar(vals[:8],1,ax,cmap,vmin,vmax)
    ax = plot_polar_bar(vals[8:16],2,ax,cmap,vmin,vmax,zorder=-1)
    ax.set_yticks([])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax, vmin=vmin))

    cl = plt.colorbar(sm, ax=ax, pad=0.1, fraction=0.02, aspect=10, orientation="vertical")
    cl.set_label(label)
    ax.set_xticks([0,np.pi/2,np.pi,np.pi*3/2])
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(-1)

    return ax

fig, ax = plt.subplots()
cmap = plt.cm.Reds
ax = make_polar_heatmap(ax,values["cum_ant_speed"].mean(axis=-1),cmap)

fig.show()


SI_t = dfi.SI.values.reshape(24,16)