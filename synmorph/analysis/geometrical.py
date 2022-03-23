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

