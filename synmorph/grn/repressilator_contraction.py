import numpy as np
from scipy.integrate import odeint

"""
Modification to repressilator class whereby increase in x1 induces a contraction in A0

A0_i = (A0max - A0min)*(xmax - x1)/(xmax) + A0min
"""


class Repressilator_contraction:
    def __init__(self, tissue, grn_params=None):
        assert grn_params is not None, "Specify grn params"
        self.t = tissue
        self.var = None
        self.nvar = None
        self.params = grn_params
        self.initialize()

    def initialize(self):
        if self.params["init_type"] == "normal":
            self.initialize_var_normal(self.params["init_vals"], self.params["init_noise"])
        elif self.params["init_type"] == "uniform":
            self.initialize_var_uniform(self.params["init_val_min"], self.params["init_val_max"])
        else:
            print("Define init_type")
        self.equilibrate()

    def initialize_var_normal(self, init_vals, noise=None):
        """
        Initialize variable values according to normal distribution. Init_vals is the mean, noise is the SD.

        :param init_vals:
        :param noise:
        :return:
        """
        if noise is None:
            noise = np.zeros((init_vals.size))
        elif type(noise) is float:
            noise = np.ones(init_vals) * noise
        self.nvar = len(init_vals)
        self.var = np.empty((self.t.mesh.n_c, self.nvar))
        for i, init_val in enumerate(init_vals):
            self.var[:, i] = init_val + np.random.normal(0, noise[i], self.var[:, i].shape)

    def initialize_var_uniform(self, init_val_min, init_val_max):
        """
        As above, but a uniform distribution, defining minimum and maximum values
        :param init_val_min:
        :param init_val_max:
        :return:
        """
        self.nvar = len(init_val_min)
        self.var = np.empty((self.t.mesh.n_c, self.nvar))
        for i, (minval, maxval) in enumerate(zip(init_val_min, init_val_max)):
            self.var[:, i] = np.random.uniform(minval, maxval, self.var[:, i].shape)

    def f_repressilator(self, x, t, params):
        """
        Elowitz repressilator, but with an additional term parameterized by kappa, where x_1 is inhibited by the sum of x_3s in neighbouring cells weighted by the edge lengths of the corresponding contacts.
        :param x:
        :param t:
        :param params:
        :return:
        """
        x_1, x_2, x_3 = x[::3], x[1::3], x[2::3]
        sum_x_3_neighbours = self.t.mesh.l_int @ x_3  ##self.t.mesh.l_int is a sparse nc x nc matrix of cell-cell contact lengths.
        dtx_1 = params["beta"] / (
                    1 + (x_3) ** params["n"] + (params["kappa"] * sum_x_3_neighbours) ** params["n"]) - x_1
        dtx_2 = params["beta"] / (1 + x_1 ** params["n"]) - x_2
        dtx_3 = params["beta"] / (1 + x_2 ** params["n"]) - x_3
        dtx = np.zeros_like(x)
        dtx[::3], dtx[1::3], dtx[2::3] = dtx_1.ravel(), dtx_2.ravel(), dtx_3.ravel()
        return dtx * params["sf"]

    def equilibrate(self):
        """
        Run the simulation without coupling. For the repressilator, this is setting kappa to 0
        :return:
        """

        params_equilibrate = self.params.copy()
        params_equilibrate["kappa"] = 0
        ysol_init = odeint(self.f_repressilator, self.var.ravel(),
                           np.arange(0, self.params["tfin_equilibrate"], self.params["dt_equilibrate"]),
                           args=(params_equilibrate,))
        self.var = ysol_init[-1].reshape(-1, 3)

    def update_grn(self, dt, dt_grn):
        ysol = odeint(self.f_repressilator,
                      self.var.ravel(),
                      np.arange(0, dt + dt_grn, dt_grn),
                      args=(self.params,))
        self.var = ysol[-1].reshape(-1, 3)
        self.t.tissue_params["A0"] = self.params["A0min"] + (self.params["A0max"] - self.params["A0min"]) * (
                    self.params["xmax"] - self.var[:, 0]) / self.params["xmax"]
