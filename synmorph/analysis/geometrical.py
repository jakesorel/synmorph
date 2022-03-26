import numpy as np
from numba import jit
import synmorph.tri_functions as trf
from synmorph.mesh import Mesh

def mesh_assembler(x_save,tri_save,L,run_options):
    meshes = [None]*len(x_save)
    for i, (x,tri) in enumerate(zip(x_save,tri_save)):
        meshes[i] = Mesh(x=x,tri=tri,fill=False,L=L,run_options=run_options)
    return meshes

def heterotypic_perimeter(mesh,c_types):
    het_mat = c_types!=np.expand_dims(c_types,1)
    return (mesh.l_int.toarray()*het_mat).sum(axis=1)

def heterotypic_perimeter_proportion(mesh,c_types):
    het_perim = heterotypic_perimeter(mesh,c_types)
    return het_perim/mesh.P

def average_heterotypic_perimeter_proportion(mesh,c_types):
    het_perim_prop = heterotypic_perimeter_proportion(mesh,c_types)
    return het_perim_prop.mean()

def average_perimeter_by_cell_type(mesh,c_types,return_c_types=False):
    unique_c_types = np.unique(c_types)
    np.sort(unique_c_types)
    av_perimeter = np.zeros(len(unique_c_types))
    for i, ctp in enumerate(unique_c_types):
        av_perimeter[i] = mesh.P[c_types == ctp].mean()
    if return_c_types:
        return unique_c_types,av_perimeter
    else:
        return av_perimeter

def average_area_by_cell_type(mesh,c_types,return_c_types=False):
    unique_c_types = np.unique(c_types)
    np.sort(unique_c_types)
    av_area = np.zeros(len(unique_c_types))
    for i, ctp in enumerate(unique_c_types):
        av_area[i] = mesh.A[c_types == ctp].mean()
    if return_c_types:
        return unique_c_types,av_area
    else:
        return av_area

def average_shape_index_by_cell_type(mesh,c_types,return_c_types=False):
    unique_c_types = np.unique(c_types)
    np.sort(unique_c_types)
    av_p0 = np.zeros(len(unique_c_types))
    for i, ctp in enumerate(unique_c_types):
        mask = c_types == ctp
        a,p = mesh.A[mask],mesh.P[mask]
        p0 = p/np.sqrt(a)
        av_p0[i] = p0.mean()
    if return_c_types:
        return unique_c_types,av_p0
    else:
        return av_p0

def apply_c_type_fn_to_meshes(fn,meshes,c_types):
    return np.array(list(map(fn, meshes, [c_types] * len(meshes))))

def apply_fn_to_meshes(fn,meshes):
    return np.array(list(map(fn, meshes)))


def get_eccentricity_of_cell(v_x_of_cell):
    """https://stackoverflow.com/questions/52406661/eccentricity-of-a-2d-convex-hull-in-python

    """
    points = v_x_of_cell
    small_latwise = np.min(points[points[:, 0] == np.min(points[:, 0])], 0)
    small_lonwise = np.min(points[points[:, 1] == np.min(points[:, 1])], 0)
    big_latwise = np.max(points[points[:, 0] == np.max(points[:, 0])], 0)
    big_lonwise = np.max(points[points[:, 1] == np.max(points[:, 1])], 0)
    distance_lat = np.sqrt(((big_latwise-small_latwise)**2).sum())
    distance_lon = np.sqrt(((big_lonwise-small_lonwise)**2).sum())
    if distance_lat >= distance_lon:
        major_axis_length = distance_lat
        minor_axis_length = distance_lon
    else:
        major_axis_length = distance_lon
        minor_axis_length = distance_lat
    a = major_axis_length / 2
    b = minor_axis_length / 2
    ecc = np.sqrt(np.square(a) - np.square(b)) / a
    return ecc

def get_vectors_of_cell(cid,vec,tri):
    # vertex_mask = trf.sum_tri(tri == cid)
    return vec[tri==cid]

def get_vectors_of_cells(vec,tri):
    cids = np.unique(tri)
    np.sort(cids)
    n_c = len(cids)
    return list(map(get_vectors_of_cell,cids,[vec]*n_c,[tri]*n_c))

def get_eccentricities(mesh):
    cell_vertices = get_vectors_of_cells(mesh.v_x,mesh.tri)
    eccs = np.array(list(map(get_eccentricity_of_cell,cell_vertices)))
    return eccs

def average_eccentricies_by_cell_type(mesh,c_types):
    unique_c_types = np.unique(c_types)
    np.sort(unique_c_types)
    average_ecc = np.zeros(len(unique_c_types))
    eccs = get_eccentricities(mesh)
    for i, ct in enumerate(unique_c_types):
        average_ecc[i] = eccs[c_types == ct].mean()
    return average_ecc

#geo.average_perimeter_by_cell_type = average_perimeter_by_cell_type