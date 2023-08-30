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
        print("is_polar")
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

fig, ax = plt.subplots()
ax = plot_polar_bar([0.1,0.2,0.3,0.4],1,ax,plt.cm.plasma)
ax = plot_polar_bar([0.5,0.6,0.7,0.8],2,ax,plt.cm.plasma,zorder=-1)
ax.set_yticks([])
ax.xaxis.grid(False)
ax.yaxis.grid(False)
fig.show()