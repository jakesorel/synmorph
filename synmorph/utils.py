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


##could also put the default params in here...
