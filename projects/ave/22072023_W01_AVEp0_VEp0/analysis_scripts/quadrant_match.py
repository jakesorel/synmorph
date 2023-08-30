import numpy as np
import matplotlib.pyplot as plt


data = np.array([2.83333333, 1.6, 1.75, 2.6, 0.875, 1.75])

N = len(data)
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = data
width = 2 * np.pi / N

ax = plt.subplot(111, polar=True)


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
    ax = plot_polar_bar(vals[:4],1,ax,cmap,vmin,vmax)
    ax = plot_polar_bar(vals[4:8],2,ax,cmap,vmin,vmax,zorder=-1)
    ax.set_yticks([])
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmax=vmax, vmin=vmin))

    cl = plt.colorbar(sm, ax=ax, pad=0.1, fraction=0.02, aspect=10, orientation="vertical")
    cl.set_label(label)
    ax.set_xticks([0,np.pi/2,np.pi,np.pi*3/2])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    return ax

fig, ax = plt.subplots()
cmap = plt.cm.plasma
ax = make_polar_heatmap(ax,[1,2,3,4,5,6,7,8],cmap)

fig.show()
