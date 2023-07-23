import numpy
import h5py
import gzip
import shutil
import os
import numpy as np
_latex = {"dA": r"$\Delta A$",
          "dP": r"$\Delta P$",
          "kappa_A": r"$\kappa_A$",
          "kappa_P": r"$\kappa_P$",
          "A0": r"$A_0$",
          "P0": r"$P_0$"}


def serialise_dict(dictionary):
    serialised_dictionary = dictionary.copy()
    for key,value in dictionary.items():
        if type(value) is numpy.ndarray:
            serialised_dictionary[key] = value.tolist()
        else:
            serialised_dictionary[key] = value
    return serialised_dictionary

def convert_to_unsigned_int(arr,L):
    arr_div = arr/L
    arr_int = numpy.round(arr_div*2**16).astype(numpy.uint16)
    return arr_int

def convert_to_float(arr_int,L):
    return arr_int.astype(numpy.float64)*L/(2**16)


def load_hdf5_skeleton(file_path,L):
    # Unzip the .h5.gz file.
    unzipped_file_path = file_path.rstrip('.gz')
    with gzip.open(file_path, 'rb') as f_in:
        with open(unzipped_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Load the pickled dictionary from the .h5 file.
    with h5py.File(unzipped_file_path, 'r') as f:
        skeleton_dict = {}
        for key in f.keys():
            skeleton_dict[key] = np.array(f[key])

    # Remove the temporary unzipped .h5 file.
    os.remove(unzipped_file_path)

    # Convert back to the original data types if needed.
    skeleton_dict["x_save"] = convert_to_float(skeleton_dict["x_save"],L)  # Assuming L is defined somewhere
    skeleton_dict["tri_save"] = skeleton_dict["tri_save"].astype(int)

    return skeleton_dict

##could also put the default params in here...
