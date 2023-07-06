import _pickle as cPickle
import bz2
import pickle

import numpy as np
from numba import jit,f4,i4

import synmorph.tri_functions as trf
from synmorph.active_force import ActiveForce
from synmorph.force import Force
from synmorph.mesh import Mesh
from synmorph import utils


class Tissue:
    """
    Tissue class
    ------------

    This class sits above the mesh, force, active_force,grn classes and integrates information about geometry and other properties of cells to determine forces.

    This class is used to initialize the mesh, force and active_force classes.

    """

    def __init__(self, tissue_params=None, active_params=None, init_params=None, initialize=True, calc_force=True, meshfile=None,
                 run_options=None,tissue_file=None):

        if tissue_file is None:
            assert tissue_params is not None, "Specify tissue params"
            assert active_params is not None, "Specify active params"
            assert init_params is not None, "Specify init params"
            assert run_options is not None, "Specify run options"

            self.tissue_params = tissue_params

            self.init_params = init_params
            self.mesh = None

            self.c_types = None
            self.nc_types = None
            self.c_typeN = None
            self.tc_types, self.tc_typesp, self.tc_typesm = None, None, None

            if meshfile is None:
                assert init_params is not None, "Must provide initialization parameters unless a previous mesh is parsed"
                if initialize:
                    self.initialize(run_options)
            else:
                self.mesh = Mesh(load=meshfile, run_options=run_options)
                assert self.L == self.mesh.L, "The L provided in the params dict and the mesh file are not the same"

            for par in ["A0", "P0", "kappa_A", "kappa_P"]:
                self.tissue_params[par] = _vectorify(self.tissue_params[par], self.mesh.n_c)

            self.active = ActiveForce(self, active_params)

            if calc_force:
                self.get_forces()
            else:
                self.F = None

            self.time = None

            self.name = None
            self.id = None

        else:
            self.load(tissue_file)

    def set_time(self, time):
        """
        Set the time and date at which the simulation was performed. For logging.
        :param time:
        :return:
        """
        self.time = time

    def update_tissue_param(self, param_name, val):
        """
        Short-cut for updating a tissue parameter
        :param param_name: dictionary key
        :param val: corresponding value
        :return:
        """
        self.tissue_params[param_name] = val

    def initialize(self, run_options=None):
        """
        Initialize the tissue. Here, the mesh is initialized, and cell types are assigned.
        In the future, this may want to be generalized.

        :param run_options:
        :return:
        """
        self.initialize_mesh(run_options=run_options)
        self.assign_ctypes()

    def initialize_mesh(self, run_options=None):
        """
        Make initial condition. Currently, this is a hexagonal lattice + noise

        Makes reference to the self.hexagonal_lattice function, then crops down to the reference frame

        If x is supplied, this is over-ridden

        :param run_options:
        :param L: Domain size/length (np.float32)
        :param noise: Gaussian noise added to {x,y} coordinates (np.float32)
        """

        x = trf.hexagonal_lattice(int(self.L), int(np.ceil(self.L)), noise=self.init_noise, A=self.A0)
        x += 1e-3
        np.argsort(x.max(axis=1))

        x = x[np.argsort(x.max(axis=1))[:int(self.L ** 2 / self.A0)]].astype(np.float32)
        self.mesh = Mesh(x, self.L, run_options=run_options)

    def assign_ctypes(self):
        assert sum(self.c_type_proportions) == 1.0, "c_type_proportions must sum to 1.0"
        assert (np.array(self.c_type_proportions) >= 0).all(), "c_type_proportions values must all be >=0"
        self.nc_types = len(self.c_type_proportions)
        self.c_typeN = [int(pr * self.mesh.n_c) for pr in self.c_type_proportions[:-1]]
        self.c_typeN += [self.mesh.n_c - sum(self.c_typeN)]

        c_types = np.zeros(self.mesh.n_c, dtype=np.int32)
        j = 0
        for k, ctN in enumerate(self.c_typeN):
            c_types[j:j + ctN] = k
            j += ctN
        np.random.shuffle(c_types)
        self.c_types = c_types
        self.c_type_tri_form()

    def c_type_tri_form(self):
        """
        Convert the nc x 1 c_type array to a nv x 3 array -- triangulated form.
        Here, the CW and CCW (p,m) cell types can be easily deduced by the roll function.
        :return:
        """
        self.tc_types = trf.tri_call_int(self.c_types, self.mesh.tri)
        self.tc_typesp = trf.roll_int(self.tc_types, -1)
        self.tc_typesm = trf.roll_int(self.tc_types, 1)

    def get_forces(self):
        """
        Calculate the forces by calling the Force class.
        :return:
        """
        self.F = Force(self).F
        return sum_forces(self.F, self.active.aF)

    def update(self, dt):
        """
        Wrapper for update functions.
        :param dt: time-step.
        :return:
        """
        self.update_active(dt)
        self.update_mechanics()

    def update_active(self, dt):
        """
        Wrapper for update of active forces
        :param dt: time-step
        :return:
        """
        self.active.update_active_force(dt)

    def update_mechanics(self):
        """
        Wrapper of update of the mesh. The mesh is retriangulated and the geometric properties are recalculated.

        Then the triangulated form of the cell types are reassigned.
        :return:
        """
        self.mesh.update()
        self.c_type_tri_form()

    def update_x_mechanics(self, x):
        """
        Like update_mechanics, apart from x is explicitly provided.
        :param x:
        :return:
        """
        self.mesh.x = x
        self.update_mechanics()

    @property
    def init_noise(self):
        return self.init_params["init_noise"]

    @property
    def c_type_proportions(self):
        return self.init_params["c_type_proportions"]

    @property
    def L(self):
        return self.tissue_params["L"]

    @property
    def A0(self):
        return self.tissue_params["A0"]

    @property
    def P0(self):
        return self.tissue_params["P0"]

    @property
    def kappa_A(self):
        return self.tissue_params["kappa_A"]

    @property
    def kappa_P(self):
        return self.tissue_params["kappa_P"]

    @property
    def W(self):
        return self.tissue_params["W"]

    @property
    def a(self):
        return self.tissue_params["a"]

    @property
    def k(self):
        return self.tissue_params["k"]

    ###More properties, for plotting primarily.

    @property
    def dA(self):
        return self.mesh.A - self.A0

    @property
    def dP(self):
        return self.mesh.P - self.P0

    def get_latex(self, val):
        if val in utils._latex:
            return utils._latex[val]
        else:
            print("No latex conversion in the dictionary.")
            return val

    def save(self, name, id=None, dir_path="", compressed=False):
        self.name = name
        if id is None:
            self.id = {}
        else:
            self.id = id
        if compressed:
            with bz2.BZ2File(dir_path + "/" + self.name + "_tissue" + '.pbz2', 'w') as f:
                cPickle.dump(self.__dict__, f)
        else:
            pikd = open(dir_path + "/" + self.name + "_tissue" + '.pickle', 'wb')
            pickle.dump(self.__dict__, pikd)
            pikd.close()

    def load(self, fname):
        if fname.split(".")[1] == "pbz2":
            fdict = cPickle.load(bz2.BZ2File(fname, 'rb'))

        else:
            pikd = open(fname, 'rb')
            fdict = pickle.load(pikd)
            pikd.close()
        self.__dict__ = fdict


@jit(nopython=True)
def sum_forces(F, aF):
    return F + aF


@jit(f4[:](f4,i4))
def _vectorify(x, n):
    return (x * np.ones(n)).astype(np.float32)
