import os
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_hex
from matplotlib.patches import Polygon
from scipy.spatial import Voronoi

"""
Plotting funcitons
------------------

Sets of plotting functions for the voronoi model, allowing to generate static images and animations. Colours and other kwargs can be parsed. 
"""


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def hex_to_rgb(value):
    """
    Convert a hex to an rgb value.
    :param value:
    :return:
    """
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) / 255 for i in range(0, lv, lv // 3)) + (1,)


def plot_vor(ax, x, L, cols=None, cbar=None, **kwargs):
    """
    Plot the Voronoi.

    Takes in a set of cell locs (x), tiles these 9-fold, plots the full voronoi, then crops to the field-of-view

    :param L: Domain size
    :param x: Cell locations (nc x 2)
    :param ax: matplotlib axis
    :param cols: array of strings (e.g. hex) defining the colour of each cell, in the order of x
    :param cbar: dictionary defining the options of the colorbar. cbar["cmap"] is the colormap. cbar["vmax],cbar["vmin] are the max and min vals in the cmap. cbar["label"] is the colorbar label.
    """

    if cols is None:
        cols = np.repeat("grey", x.shape[0])
    grid_x, grid_y = np.mgrid[-1:2, -1:2]
    grid_x[0, 0], grid_x[1, 1] = grid_x[1, 1], grid_x[0, 0]
    grid_y[0, 0], grid_y[1, 1] = grid_y[1, 1], grid_y[0, 0]
    y = np.vstack([x + np.array([i * L, j * L]) for i, j in np.array([grid_x.ravel(), grid_y.ravel()]).T])

    cols_print = np.tile(cols, 9)
    bleed = 0.1
    cols_print = cols_print[(y < L * (1 + bleed)).all(axis=1) + (y > -L * bleed).all(axis=1)]
    y = y[(y < L * (1 + bleed)).all(axis=1) + (y > -L * bleed).all(axis=1)]
    regions, vertices = voronoi_finite_polygons_2d(Voronoi(y))
    patches = []
    for i, region in enumerate(regions):
        patches.append(Polygon(vertices[region], True, facecolor=cols_print[i], ec=(1, 1, 1, 1), **kwargs))

    p = PatchCollection(patches, match_original=True)
    # p.set_array(c_types_print)
    ax.add_collection(p)
    ax.set(xlim=(0, L), ylim=(0, L), aspect=1)
    ax.axis("off")
    if cbar is not None:
        sm = plt.cm.ScalarMappable(cmap=cbar["cmap"], norm=plt.Normalize(vmax=cbar["vmax"], vmin=cbar["vmin"]))
        sm._A = []
        cl = plt.colorbar(sm, ax=ax, pad=0.05, fraction=0.05, aspect=10, orientation="vertical")
        cl.set_label(cbar["label"])


def animate(x_save, L, cols, n_frames=100, file_name=None, dir_name="plots", cbar=None, **kwargs):
    """
    Animate the simulation

    :param x_save: nt x nc x 2 array of positions
    :param L: domain size
    :param cols: either a nc array of strings defining fixed colors, or a nt x nc array if colors are to vary.
    :param n_frames: number of frames ot plot
    :param file_name: file name
    :param dir_name: directory name into which the animation is saved
    :param cbar: see above.
    :param kwargs: other arguments parsed into plot_vor
    :return:
    """

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    skip = int((x_save.shape[0]) / n_frames)

    if cols.shape == x_save.shape[:-1]:
        dynamic_col = True
    else:
        dynamic_col = False

    if cbar is not None:
        sm = plt.cm.ScalarMappable(cmap=cbar["cmap"], norm=plt.Normalize(vmax=cbar["vmax"], vmin=cbar["vmin"]))
        sm._A = []
        cl = plt.colorbar(sm, ax=ax1, pad=0.05, fraction=0.05, aspect=10, orientation="vertical")
        cl.set_label(cbar["label"])

    def animate(i):
        ax1.cla()
        cbar_a = None
        if dynamic_col:
            plot_vor(ax1, x_save[skip * i], L, cols[skip * i], cbar=cbar_a, **kwargs)
        else:
            plot_vor(ax1, x_save[skip * i], L, cols, cbar=cbar_a, **kwargs)
        ax1.set(aspect=1, xlim=(0, L), ylim=(0, L))

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)
    if file_name is None:
        file_name = "animation %d" % time.time()
    an = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
    an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)


def generate_ctype_cols(c_types, c_type_col_map=["#00c8f0", "#f0a800"]):
    """
    Generates an array of color strings given the colormap and the cell types.
    :param c_types: array of cell types (0,1,...) of length nc
    :param c_type_col_map: corresponding colours for each of the cell types. e.g. in hex format.
    :return:
    """
    cols = np.empty(len(c_types), dtype=object)
    cols[:] = "grey"
    # cols = np.repeat("grey", len(c_types))
    for i in range(len(c_type_col_map)):
        cols[c_types == i] = c_type_col_map[i]
    return cols


def rgba_to_hex(cols):
    """
    Convert rgb to hex
    :param cols: 1, 2, or 3 dimensional array (e.g. generated from a cmap function)
    :return:
    """
    if cols.ndim == 1:
        return to_hex(cols)
    if cols.ndim == 2:
        cols_out = np.empty((cols.shape[0]), dtype=object)
        for i in range(cols.shape[0]):
            cols_out[i] = to_hex(cols[i])
        return cols_out
    if cols.ndim == 3:
        cols_out = np.empty((cols.shape[0], cols.shape[1]), dtype=object)
        for i in range(cols.shape[0]):
            for j in range(cols.shape[1]):
                cols_out[i, j] = to_hex(cols[i, j])
        return cols_out
