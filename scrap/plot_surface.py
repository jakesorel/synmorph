import numpy as np
import plotly.graph_objects as go
import mcubes
import pandas as pd

gz_names = pd.read_csv("scrap/gz_names.txt",header=None)
gz_names["index"] = [int(nm.split("_")[4]) for nm in gz_names[0]]

is_simulated = np.ones((20**4))*np.nan
for i in gz_names["index"]:
    is_simulated[i] = 1

is_simulated = is_simulated.reshape(20,20,20,20)

is_sim_repeats = np.nansum(is_simulated,axis=-1)

df = pd.read_csv("scrap/t_hit_boundary.txt",header=None)
df["index"] = [int(nm.split("_")[4]) for nm in df[0]]

t_hit_boundary = np.ones((20**4))*np.nan
for i,t in zip(df["index"].values,df[1].values):
    t_hit_boundary[i] = t

t_hit_boundary = t_hit_boundary.reshape(20,20,20,20)

average_hit = np.mean((t_hit_boundary!=-1)*(t_hit_boundary<290),axis=-1)

def plot_surface(volume, x, y, z, smooth=False):
    # Extract the surface using Marching Cubes algorithm.
    if smooth:
        verts, faces = mcubes.marching_cubes(mcubes.smooth(volume), 0.5)
    else:
        verts, faces = mcubes.marching_cubes(volume, 0.5)

    # Scale the vertices with the grid spacing of the X, Y, and Z arrays.
    verts_scaled = np.array(verts)
    verts_scaled[:, 0] = (x.max() - x.min()) * verts[:, 0] / volume.shape[0] + x.min()
    verts_scaled[:, 1] = (y.max() - y.min()) * verts[:, 1] / volume.shape[1] + y.min()
    verts_scaled[:, 2] = (z.max() - z.min()) * verts[:, 2] / volume.shape[2] + z.min()

    # Convert the faces array to integers.
    faces = faces.astype(int)

    # Create the figure for the interactive 3D plot.
    fig = go.Figure()

    # Add the mesh surface to the plot.
    fig.add_trace(go.Mesh3d(x=verts_scaled[:, 0], y=verts_scaled[:, 1], z=verts_scaled[:, 2],
                            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2]))

    # fig.add_trace(go.Surface(x=verts_scaled[:, 0], y=verts_scaled[:, 1], z=verts_scaled[:, 2]))


    # Set the axis labels and title.
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                      title="Approximated Surface of Filled Volume")

    # Flip the x-axis by reversing the range of xaxis.
    fig.update_layout(scene=dict(xaxis=dict(range=[x.max(), x.min()])))

    # Show the interactive 3D plot.
    fig.show()

# # Define the volume as provided
# x = np.linspace(0, 2, 50)
# y = np.linspace(0, 2, 50)
# z = np.linspace(0, 2, 50)
# X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
# volume = (np.sqrt(X**2 + Y**2 + Z**2) <= 1) + (np.sqrt((X - 2)**2 + Y**2 + Z**2) <= 1)

N = 20
W01_range = np.linspace(-3, -1, N)
AVE_p0_range = np.linspace(3.4, 5, N)
VE_p0_range = np.linspace(3.4, 5, N)
# Plot the surface of the volume interactively with custom tick values and labels.
plot_surface(average_hit > 0.5, W01_range, AVE_p0_range, VE_p0_range, False)


import numpy as np


def intersect_surface_with_plane(surface_faces, surface_vertices, alpha,axis=0):
    def intersect_triangle_with_plane(triangle_vertices, alpha):
        intersection_points = []

        for i in range(3):
            v1 = surface_vertices[triangle_vertices[i]]
            v2 = surface_vertices[triangle_vertices[(i + 1) % 3]]

            if (v1[axis] < alpha and v2[axis] > alpha) or (v1[axis] > alpha and v2[axis] < alpha):
                t = (alpha - v1[axis]) / (v2[axis] - v1[axis])
                intersection_point = np.array([
                    v1[0] + t * (v2[0] - v1[0]),
                    v1[1] + t * (v2[1] - v1[1]),
                    v1[2] + t * (v2[2] - v1[2])
                ])
                intersection_points.append(intersection_point)

        return intersection_points

    intersection_points = []

    for face in surface_faces:
        intersection_points.extend(intersect_triangle_with_plane(face, alpha))

    if len(intersection_points) >= 2:
        intersection_line = np.array(intersection_points)
    else:
        intersection_line = None

    return intersection_line


intersection_line = intersect_surface_with_plane(faces, verts, alpha,1)
print("Intersection Line:", intersection_line)
