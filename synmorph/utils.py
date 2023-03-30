import numpy as np

_latex = {"dA": r"$\Delta A$",
          "dP": r"$\Delta P$",
          "kappa_A": r"$\kappa_A$",
          "kappa_P": r"$\kappa_P$",
          "A0": r"$A_0$",
          "P0": r"$P_0$"}


grid_x, grid_y = np.mgrid[-1:2, -1:2]
grid_x[0, 0], grid_x[1, 1] = grid_x[1, 1], grid_x[0, 0]
grid_y[0, 0], grid_y[1, 1] = grid_y[1, 1], grid_y[0, 0]
grid_xy = np.array([grid_x.ravel(), grid_y.ravel()]).T


def serialise_dict(dictionary):
    serialised_dictionary = dictionary.copy()
    for key,value in dictionary.items():
        if isinstance(value, np.ndarray):
            serialised_dictionary[key] = value.tolist()
        else:
            serialised_dictionary[key] = value
    return serialised_dictionary


##could also put the default params in here...
