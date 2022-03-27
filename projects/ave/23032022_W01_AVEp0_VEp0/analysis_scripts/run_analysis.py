import sys
import os


SCRIPT_DIR = "../../../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

import json
import numpy as np
from synmorph.analysis import geometrical as geo
from synmorph.analysis import topological as top
from synmorph.analysis import spatial as sp
import pickle
import pandas as pd

def run(sim_name):

    def open_json(json_path):
        with open(json_path) as json_file:
            data = json.load(json_file)
        return data

    # process_skip = 6
    # sim_name = "23032022_W01_AVEp0_VEp0_0"
    sim_json = open_json("../scan_results/%s/pickled/%s"%(sim_name,sim_name)+"_simulation.json")
    pikd = open("../scan_dicts/%s.pickle" % sim_name, 'rb')
    scan_dict = pickle.load(pikd)
    pikd.close()

    run_options = scan_dict["run_options"]

    x = np.array(sim_json["x_save"])
    t = np.array(sim_json["t_span_save"])
    tri = np.array(sim_json["tri_save"])
    L = sim_json["L"]
    c_types = np.array(sim_json["c_types"])
    x_unwrapped = sp.unwrap_positions(x,L)
    meshes = geo.mesh_assembler(x,tri,L,run_options)

    def get_AVE_x(x,c_types,c_type=0):
        mask = c_types == c_type
        AVE_x = x[:,mask]
        return AVE_x

    def get_average_centroid(x_sample):
        return x_sample.mean(axis=1)

    def get_average_velocity(x_sample,t):
        dt = t[1] - t[0]
        v = (x_sample[1:] - x_sample[:-1])/dt
        return v.mean(axis=1)

    def get_scalar_from_vector(v):
        return np.sqrt(v[...,0]**2 + v[...,1]**2)

    AVE_x = get_AVE_x(x,c_types)
    av_AVE_x = get_average_centroid(AVE_x)
    av_AVE_d = get_scalar_from_vector(av_AVE_x-av_AVE_x[0])
    av_AVE_v = get_average_velocity(AVE_x,t)
    av_AVE_speed = get_scalar_from_vector(av_AVE_v)
    bulk_AVE_velocity = ((av_AVE_x-av_AVE_x[0])[1:] - (av_AVE_x-av_AVE_x[0])[:-1])/(t[1]-t[0])
    bulk_AVE_speed = get_scalar_from_vector(bulk_AVE_velocity)

    ##Geometrical
    average_P = geo.apply_c_type_fn_to_meshes(geo.average_perimeter_by_cell_type,meshes,c_types)
    average_A = geo.apply_c_type_fn_to_meshes(geo.average_area_by_cell_type,meshes,c_types)
    average_p0 = geo.apply_c_type_fn_to_meshes(geo.average_shape_index_by_cell_type,meshes,c_types)
    # average_eccentricity = geo.apply_c_type_fn_to_meshes(geo.average_eccentricies_by_cell_type,meshes,c_types)

    def get_ave_connected_components(mesh):
        return top.count_connected_components(mesh.tri, c_types, len(mesh.x))[0]

    ave_connected_components = np.array(list(map(get_ave_connected_components,meshes)))

    def append_NaN(x):
        return np.concatenate((x,(np.nan,)))


    grn_params = scan_dict["grn_params"]
    AVE_p0 = np.repeat(grn_params["AVE_p0"],len(AVE_x[:,0]))
    VE_p0 = np.repeat(grn_params["nonAVE_p0"],len(AVE_x[:,0]))
    W01 = np.repeat(scan_dict["tissue_params"]["W"][0,1],len(AVE_x[:,0]))

    df = pd.DataFrame({"t":t,"PARAM_AVE_p0":AVE_p0,"PARAM_VE_p0":VE_p0,"PARAM_W01":W01,
                       "v_AVE_x":av_AVE_x[:,0],"av_AVE_y":av_AVE_x[:,1],"av_av_AVE_d":av_AVE_d,
                       "av_AVE_vx":append_NaN(av_AVE_v[:,0]),"av_AVE_vy":append_NaN(av_AVE_v[:,1]),
                       "av_AVE_speed":append_NaN(av_AVE_speed),"bulk_AVE_speed":append_NaN(bulk_AVE_speed),
                       "average_P_AVE":average_P[:,0],"average_P_VE":average_P[:,1],
                       "average_A_AVE":average_A[:,0],"average_A_VE":average_A[:,1],
                       "average_p0_AVE":average_p0[:,0],"average_p0_VE":average_p0[:,1],
                       # "average_ecc_AVE":average_eccentricity[:,0],"average_ecc_VE":average_eccentricity[:,1],
                       "AVE_connected_components":ave_connected_components})
    if not os.path.exists("../analysis_results"):
        os.mkdir("../analysis_results")
    if not os.path.exists("../analysis_results/all"):
        os.mkdir("../analysis_results/all")

    df.to_csv("../analysis_results/all/%s_analysis.csv"%(sim_name),index=None)

if __name__ == "__main__":
    i = int(sys.argv[1])
    path_names = open("../scan_summary/23032022_W01_AVEp0_VEp0_path_names.txt").readlines()
    path_name = path_names[i].split("\n")[0]
    out_file = open("../scan_summary/23032022_W01_AVEp0_VEp0_result_log.txt")
    out_file_lines = out_file.readlines()
    if any([path_name in o for o in out_file_lines]):
        sim_name = path_name.split(".pickle")[0].split("scan_dicts/")[1]
        run(sim_name)
