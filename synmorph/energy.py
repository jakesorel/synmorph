import numpy as np
from numba import jit

import synmorph.periodic_functions as per
import synmorph.tri_functions as trf
import synmorph.force as force

class Energy:
    """
    Energy class
    -----------

    This class is used to calculate the energy under the SPV model.

    Takes in an instance of the Tissue class, and within it, the Mesh class, and uses information about the geometry to calculate the energy per cell.

    Energy functional is given by:

        E = Sum_i (kappa_A/2) * (Ai - A0)^2 + (kappa_P/2) * (Pi - P0)^2 + Sum_j J_ij*lij
    
    These forces are accessible in self.E
    """

    def __init__(self, tissue):

        self.t = tissue
        self.Jp, self.Jm = None, None  ##triangular forms of the J matrix, considering J_ij for CW (Jp) and CCW (Jm) neighbours.
        self.get_J()

        self.get_E_per_cell()
        
        self.E = self.E_v.sum()

    def get_J(self):
        self.Jp = force.get_J(self.t.W, self.t.tc_types, self.t.tc_typesp, self.t.nc_types)
        self.Jm = force.get_J(self.t.W, self.t.tc_types, self.t.tc_typesm, self.t.nc_types)
    

    def get_E_v(self):
        self.E_v = get_E_per_cell
            self.t.mesh.A,
            self.t.mesh.P,
            self.Jm,
            self.Jp,
            self.t.mesh.lm1,
            self.t.mesh.lp1,
            self.t.kappa_A,
            self.t.kappa_P,
            self.t.A0,
            self.t.P0,
        )

def get_E_per_cell(
    A,
    P,
    Jm,
    Jp,
    lm1,
    lp1,
    kappa_A,
    kappa_P,
    A0,
    P0,
):
    
    E = 0
    E += kappa_A / 2 * (A - A0) ** 2
    E += kappa_P / 2 * (P - P0) ** 2
#    E += Jm * lm1 + Jp * lp1    ## This is wrong - looks like J is defined at vertices?
    return E_A + E_P + E_J


