import sys
import os
sys.dont_write_bytecode = True

SCRIPT_DIR = "../../../../"
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pickle
import time
import synmorph as sm


def run_simulation(path_name):
    pikd = open("../%s"%path_name, 'rb')
    scan_dict = pickle.load(pikd)
    pikd.close()

    print(scan_dict)
    sim = sm.simulation(tissue_params=scan_dict["tissue_params"],
                        active_params=scan_dict["active_params"],
                        init_params=scan_dict["init_params"],
                        simulation_params=scan_dict["simulation_params"],
                        grn_params=scan_dict["grn_params"],
                        run_options=scan_dict["run_options"],
                        save_options=scan_dict["save_options"])

    sim.simulate(progress_bar=False)


if __name__ == "__main__":
    # j, k, Nper = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    # i = j + k * Nper
    # path_names = open("../scan_summary/23032022_W01_AVEp0_VEp0_path_names.txt").readlines()
    # path_name = path_names[i].split("\n")[0]
    #
    path_name = "scan_dicts/23032022_W01_AVEp0_VEp0_12.pickle"
    t0 = time.time()
    run_simulation(path_name)
    t1 = time.time()
    print(t1-t0)
    out_file = open("../scan_summary/23032022_W01_AVEp0_VEp0_result_log.txt","a")
    out_file.write("%s_%.2f"%(path_name,(t1-t0)) + "\n")
    out_file.close()


    #
    # import _pickle as cPickle
    # import bz2
    # import importlib
    # import os
    # import pickle
    # from datetime import datetime
    # import codecs, json
    #
    #
    # def load(fname):
    #     """
    #     Load an instance of the simulation class.
    #     :param fname:
    #     :return:
    #     """
    #     if fname.split(".")[1] == "pbz2":
    #         fdict = cPickle.load(bz2.BZ2File(fname, 'rb'))
    #
    #     else:
    #         pikd = open(fname, 'rb')
    #         fdict = pickle.load(pikd)
    #         pikd.close()
    #     return fdict
    #
    # skeleton_dict = load("/Users/cornwaj/PycharmProjects/synmorph/projects/ave/23032022_W01_AVEp0_VEp0/scan_results/23032022_W01_AVEp0_VEp0_12/pickled/23032022_W01_AVEp0_VEp0_12_simulation.pbz2")
    #
    # dir_path = '../scan_results/23032022_W01_AVEp0_VEp0_12/pickled'
    # name = '23032022_W01_AVEp0_VEp0_12'
    #
    # with bz2.BZ2File(dir_path + "/" + name + "_simulation_resaved" + '.pbz2', 'w') as f:
    #     cPickle.dump(skeleton_dict, f)
    #
    # import pickle, gzip, lzma, bz2
    #
    # import hickle as hkl
    # import h5py
    # import numpy as np
    # import gzip
    # import shutil
    #
    # keys = ["x_save","tri_save","c_types"]
    # dict_reduced = {}
    # for key in keys:
    #     dict_reduced[key] = skeleton_dict[key]
    #
    #
    # L = skeleton_dict["L"]
    #
    #
    #
    # def convert_to_unsigned_int(arr,L):
    #     arr_div = arr/L
    #     arr_int = np.round(arr_div*2**16).astype(np.uint16)
    #     return arr_int
    #
    # def convert_to_float(arr_int,L):
    #     return (arr_int.astype(np.float64)*L/(2**16))
    #
    #
    # dict_reduced["x_save"] = convert_to_unsigned_int(np.array(dict_reduced["x_save"]),L)
    # dict_reduced["tri_save"] = np.array(dict_reduced["tri_save"]).astype(np.uint16)
    # dict_reduced["c_types"] = np.array(dict_reduced["c_types"]).astype(np.uint8)
    #
    #
    #
    # file_path = dir_path + "/" + name + "_simulation_resaved"+'.h5'
    #
    # f = h5py.File(file_path, 'w')
    # for i, key in enumerate(dict_reduced.keys()):
    #     f.create_dataset(key,data=dict_reduced[key],compression="gzip")
    # f.close()
    #
    #
    # with open(file_path, 'rb') as f_in:
    #     with gzip.open(file_path + ".gz", 'wb') as f_out:
    #         shutil.copyfileobj(f_in, f_out)
    #
    #
    # with gzip.open(file_path + ".gz", 'rb') as f_out:
    #     with open(file_path.replace(".h5","_reopened.h5"), 'wb') as f_in:
    #         shutil.copyfileobj(f_out, f_in)
    #
    # f = h5py.File(file_path.replace(".h5","_reopened.h5"), 'r')
    # re_opened_dict = {}
    # for key in f.keys():
    #     re_opened_dict[key] = np.array(f[key])
    #
    # re_opened_dict["x_save"] = convert_to_float(re_opened_dict["x_save"],L)