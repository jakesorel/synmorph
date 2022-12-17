import numpy
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

##could also put the default params in here...
