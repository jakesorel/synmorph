import os
import sys


SCRIPT_DIR = "../../../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

import json
import numpy as np
from synmorph.analysis import geometrical as geo
from synmorph.analysis import topological as top
from synmorph.analysis import spatial as sp
import pickle
import pandas as pd

# sim_name = "23032022_W01_AVEp0_VEp0_504"


def run(sim_name):
    def open_json(json_path):
        with open(json_path) as json_file:
            data = json.load(json_file)
        return data

    # process_skip = 6
    # sim_name = "23032022_W01_AVEp0_VEp0_0"
    sim_json = open_json("../scan_results/%s/pickled/%s" % (sim_name, sim_name) + "_simulation.json")
    pikd = open("../scan_dicts/%s.pickle" % sim_name, 'rb')
    scan_dict = pickle.load(pikd)
    pikd.close()

    run_options = scan_dict["run_options"]

    x = np.array(sim_json["x_save"])
    t = np.array(sim_json["t_span_save"])
    tri = np.array(sim_json["tri_save"])
    L = sim_json["L"]
    c_types = np.array(sim_json["c_types"])
    x_unwrapped = sp.unwrap_positions(x, L)
    meshes = geo.mesh_assembler(x, tri, L, run_options)

    def get_AVE_x(x, c_types, c_type=0):
        mask = c_types == c_type
        AVE_x = x[:, mask]
        return AVE_x

    def get_average_centroid(x_sample):
        return x_sample.mean(axis=1)

    def get_average_velocity(x_sample, t):
        dt = t[1] - t[0]
        v = (x_sample[1:] - x_sample[:-1]) / dt
        return v.mean(axis=1)

    def get_scalar_from_vector(v):
        return np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)

    AVE_x = get_AVE_x(x, c_types)
    av_AVE_x = get_average_centroid(AVE_x)
    av_AVE_d = get_scalar_from_vector(av_AVE_x - av_AVE_x[0])
    av_AVE_v = get_average_velocity(AVE_x, t)
    av_AVE_speed = get_scalar_from_vector(av_AVE_v)
    bulk_AVE_velocity = ((av_AVE_x - av_AVE_x[0])[1:] - (av_AVE_x - av_AVE_x[0])[:-1]) / (t[1] - t[0])
    bulk_AVE_speed = get_scalar_from_vector(bulk_AVE_velocity)

    ##Geometrical
    average_P = geo.apply_c_type_fn_to_meshes(geo.average_perimeter_by_cell_type, meshes, c_types)
    average_A = geo.apply_c_type_fn_to_meshes(geo.average_area_by_cell_type, meshes, c_types)
    average_p0 = geo.apply_c_type_fn_to_meshes(geo.average_shape_index_by_cell_type, meshes, c_types)

    # average_eccentricity = geo.apply_c_type_fn_to_meshes(geo.average_eccentricies_by_cell_type,meshes,c_types)

    def get_ave_connected_components(mesh):
        return top.count_connected_components(mesh.tri, c_types, len(mesh.x))[0]

    ave_connected_components = np.array(list(map(get_ave_connected_components, meshes)))

    def append_NaN(x):
        return np.concatenate((x, (np.nan,)))

    grn_params = scan_dict["grn_params"]
    AVE_p0 = np.repeat(grn_params["AVE_p0"], len(AVE_x[:, 0]))
    VE_p0 = np.repeat(grn_params["nonAVE_p0"], len(AVE_x[:, 0]))
    W01 = np.repeat(scan_dict["tissue_params"]["W"][0, 1], len(AVE_x[:, 0]))

    df = pd.DataFrame({"t": t, "PARAM_AVE_p0": AVE_p0, "PARAM_VE_p0": VE_p0, "PARAM_W01": W01,
                       "v_AVE_x": av_AVE_x[:, 0], "av_AVE_y": av_AVE_x[:, 1], "av_av_AVE_d": av_AVE_d,
                       "av_AVE_vx": append_NaN(av_AVE_v[:, 0]), "av_AVE_vy": append_NaN(av_AVE_v[:, 1]),
                       "av_AVE_speed": append_NaN(av_AVE_speed), "bulk_AVE_speed": append_NaN(bulk_AVE_speed),
                       "average_P_AVE": average_P[:, 0], "average_P_VE": average_P[:, 1],
                       "average_A_AVE": average_A[:, 0], "average_A_VE": average_A[:, 1],
                       "average_p0_AVE": average_p0[:, 0], "average_p0_VE": average_p0[:, 1],
                       # "average_ecc_AVE":average_eccentricity[:,0],"average_ecc_VE":average_eccentricity[:,1],
                       "AVE_connected_components": ave_connected_components})
    if not os.path.exists("../analysis_results"):
        os.mkdir("../analysis_results")
    if not os.path.exists("../analysis_results/all"):
        os.mkdir("../analysis_results/all")

    df.to_csv("../analysis_results/all/%s_analysis.csv" % (sim_name), index=None)


def run_time_binned(sim_name, n_time_point=101):
    def open_json(json_path):
        with open(json_path) as json_file:
            data = json.load(json_file)
        return data

    # process_skip = 6
    # sim_name = "23032022_W01_AVEp0_VEp0_0"
    sim_json = open_json("../scan_results/%s/pickled/%s" % (sim_name, sim_name) + "_simulation.json")
    pikd = open("../scan_dicts/%s.pickle" % sim_name, 'rb')
    scan_dict = pickle.load(pikd)
    pikd.close()

    run_options = scan_dict["run_options"]

    t_span = np.array(sim_json["t_span_save"])
    nti = t_span.size
    ti_interval = int(nti / n_time_point)

    def get_AVE_x(x, c_types, c_type=0):
        mask = c_types == c_type
        AVE_x = x[:, mask]
        return AVE_x

    def get_average_centroid(x_sample):
        return x_sample.mean(axis=1)

    def get_average_velocity(x_sample, t):
        dt = t[1] - t[0]
        v = (x_sample[1:] - x_sample[:-1]) / dt
        return v.mean(axis=1)

    def get_scalar_from_vector(v):
        return np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)

    def apply_rotation(x_save, mid, rotation_matrix):
        _x = x_save - mid
        return (rotation_matrix @ _x.reshape(-1, 2).T).T.reshape(_x.shape)

    c_types = np.array(sim_json["c_types"])
    L = sim_json["L"]
    mid_point = np.array((L / 2, L / 2))
    AVE_x_start = get_AVE_x(np.array(sim_json["x_save"])[:1], c_types)[0]

    AVE_x_end = get_AVE_x(np.array(sim_json["x_save"])[-2:], c_types)[-1]
    # AVE_x_start_centre = AVE_x_start.mean(axis=0)

    AVE_vector = np.mean(AVE_x_end - AVE_x_start, axis=0)
    ###re-orient all positions such that AVE moves to (0,1).
    rotation_angle = np.pi / 2 - np.arctan2(AVE_vector[1], AVE_vector[0])
    rotation_matrix = np.array(
        ((np.cos(rotation_angle), -np.sin(rotation_angle)), (np.sin(rotation_angle), np.cos(rotation_angle))))

    rng = np.arange(0, ti_interval * n_time_point, ti_interval)
    t = t_span[rng]
    x_original = np.array(sim_json["x_save"])[rng]
    x = apply_rotation(x_original, mid_point, rotation_matrix)
    tri = np.array(sim_json["tri_save"])[rng]
    meshes = geo.mesh_assembler(x_original, tri, L, run_options)
    eccentricities, P, A, N_neighbours = np.zeros_like(x[:, :, 0]), np.zeros_like(x[:, :, 0]), np.zeros_like(
        x[:, :, 0]), np.zeros_like(x[:, :, 0], dtype=np.int64)
    for i, mesh in enumerate(meshes):
        eccentricities[i] = geo.get_eccentricities(mesh)
        P[i] = mesh.P
        A[i] = mesh.A
        N_neighbours[i] = np.array((mesh.l_int != 0).sum(axis=0).ravel())

    SI = P / np.sqrt(A)

    x0 = x[0]
    dx_cum = x[:-1] - x0
    v_cum = dx_cum / np.expand_dims(np.expand_dims(t[:-1], axis=1), axis=2)
    v_cum[np.isnan(v_cum)] = 0
    speed_cum = np.sqrt(v_cum[..., 0] ** 2 + v_cum[..., 1] ** 2)
    ant_speed_cum = v_cum[..., 1]
    dt = t[1] - t[0]
    dx_inst = x[1:] - x[:-1]
    v_inst = dx_inst / dt
    speed_inst = np.sqrt(v_inst[..., 0] ** 2 + v_inst[..., 1] ** 2)
    ant_speed_inst = v_inst[..., 1]

    c_type_mask = c_types < 2

    df = pd.DataFrame()

    for i in range(0, len(t)-1):
        b = meshes[i].l_int.tocsr()
        dfi = pd.DataFrame({"t": np.float32(np.repeat(t[i], len(P[i]))),
                            "c_types": c_types,
                            "cell_id": np.arange(len(P[i])),
                            "x": np.float32(x[i, :, 0]),
                            "y": np.float32(x[i, :, 1]),
                            "P": np.float32(P[i]),
                            "A": np.float32(A[i]),
                            "SI": np.float32(SI[i]),
                            "ecc": np.float32(eccentricities[i]),
                            "neighbours": [":".join(vec.astype(str)) for vec in np.split(b.indices, b.indptr[1:-1])],
                            "N_neighbours": N_neighbours[i],
                            "inst_speed": np.float32(speed_inst[i]),
                            "inst_ant_speed": np.float32(ant_speed_inst[i]),
                            "cum_speed": np.float32(speed_cum[i]),
                            "cum_ant_speed": np.float32(ant_speed_cum[i])})
        dfi = dfi[dfi["c_types"] < 2]
        df = pd.concat([df, dfi])

    if not os.path.exists("../analysis_results"):
        os.mkdir("../analysis_results")
    if not os.path.exists("../analysis_results/compiled"):
        os.mkdir("../analysis_results/compiled")
    if not os.path.exists("../analysis_results/compiled/by_cell"):
        os.mkdir("../analysis_results/compiled/by_cell")
    if not os.path.exists("../analysis_results/compiled/by_position_class"):
        os.mkdir("../analysis_results/compiled/by_position_class")
    if not os.path.exists("../analysis_results/compiled/by_position_class_and_ctype"):
        os.mkdir("../analysis_results/compiled/by_position_class_and_ctype")

    df.to_csv("../analysis_results/compiled/by_cell/%s_analysis_by_cell.csv" % (sim_name), index=None)

    ###note that distances are normalized to sqrt(A0) or sqrt(mean(A)).

    outer_radius = (1 - scan_dict["grn_params"]["boundary_frac"]) * L / 2
    inner_radius = outer_radius / 2
    dangle = np.pi / 4
    angle_bins = np.arange(-np.pi, np.pi, dangle)
    radius_bins = np.array([0, inner_radius])

    dist_from_mid = np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)

    angle_classes = np.digitize(np.arctan2(x[..., 1], x[..., 0]), angle_bins)
    angle_classes = np.mod(angle_classes - 7,8)
    radius_classes = np.digitize(dist_from_mid, radius_bins)

    classes = 8 * (radius_classes - 1) + (angle_classes)

    df["position_class"] = classes[:-1][:, c_types < 2].ravel()

    df_position_class = pd.DataFrame()
    for i in range(0, len(t)-1):
        dfi = df[df["t"] == np.float32(t[i])].groupby("position_class").mean()
        df_position_class = pd.concat([df_position_class, dfi])
    df_position_class["position_class"] = df_position_class.index

    df_position_class.to_csv(
        "../analysis_results/compiled/by_position_class/%s_analysis.csv" % (sim_name), index=False)

    df["position_and_cell_type_class"] = df["position_class"] + 64 * df["c_types"]

    df_position_and_ctype_class = pd.DataFrame()
    for i in range(0, len(t)-1):
        dfi = df[df["t"] == t[i]].groupby("position_and_cell_type_class").mean()
        df_position_and_ctype_class = pd.concat([df_position_and_ctype_class, dfi])

    df_position_and_ctype_class.to_csv(
        "../analysis_results/compiled/by_position_class_and_ctype/%s_analysis.csv" % (
            sim_name), index=False)



if __name__ == "__main__":
    j, k, Nper = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    i = j + k * Nper
    # i = int(sys.getenv('SLURM_ARRAY_TASK_ID'))
    path_names = open("../scan_summary/23032022_W01_AVEp0_VEp0_path_names.txt").readlines()
    path_name = path_names[i].split("\n")[0]
    out_file = open("../scan_summary/23032022_W01_AVEp0_VEp0_result_log.txt")
    out_file_lines = out_file.readlines()
    if any([path_name in o for o in out_file_lines]):
        sim_name = path_name.split(".pickle")[0].split("scan_dicts/")[1]
        # run(sim_name)

        run_time_binned(sim_name)

