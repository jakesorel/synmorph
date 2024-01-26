import numpy as np

class Heart_stretch:
    """
    Skeleton instance of GRN-like class.

    Called if simulation_options["grn_type"] == "grn_dummy"

    All the GRN simulation parameters can be parsed in grn_params. This allows the programmer to feed in the specifications of the GRN model via the instantiation of the overarching Simulation class.

    In principle, this class can also modify the tissue class (e.g. if a gene initiates the change in the physical property of the cell).

    """

    def __init__(self, tissue, grn_params=None):
        assert grn_params is not None, "Specify grn params"
        self.t = tissue

        self.params = grn_params
        self.true_tissue_params = {}
        self.x0 = None
        self.x0_bound = None
        self.x = None
        self.L_true = self.params["L_init"]
        self.var = self.L_true
        self.nvar = 1
        self.is_notch = None
        self.time_point = 0.
        self.time_step = 0
        self.pressure = 0.
        self.L_save = np.zeros(self.params["n_t"])
        self.initialize()



    def initialize(self):
        """
        Define your initialization of the values of self.var here. This should be of size (nc x nvar). Also include a statement to save nvar.
        :return:
        """
        self.set_notch()
        self.set_true_params()
        self.set_renormalised_params()
        self.update_pressure()


    def set_notch(self):
        self.is_notch = np.zeros(self.t.mesh.n_c, dtype=bool)
        self.is_notch[:int(np.round(self.params["p_notch"] * self.t.mesh.n_c))] = True
        if "notch_distribution" in self.params:
            if self.params["notch_distribution"] == "circular":
                self.is_notch = np.zeros(self.t.mesh.n_c, dtype=bool)
                centre = np.array((self.t.mesh.L / 2, self.t.mesh.L / 2))
                n_positive = np.zeros(100,dtype=int)
                masks = []
                for i, r in enumerate(np.linspace(0,self.t.mesh.L,100)):
                    x = self.t.mesh.x
                    mask = (x[:, 0] - centre[0]) ** 2 + (x[:, 1] - centre[1]) ** 2 <= r **2
                    masks.append(mask)
                    n_positive[i] = mask.sum()
                if np.any(n_positive < (self.params["p_notch"] * self.t.mesh.n_c)):
                    self.is_notch = masks[np.where(n_positive < self.params["p_notch"] * self.t.mesh.n_c)[0][-1]]
            else:
                np.random.shuffle(self.is_notch)
        else:
            np.random.shuffle(self.is_notch)

    def set_true_params(self):
        self.true_tissue_params["A0"] = np.ones(self.t.mesh.n_c, dtype=np.float32)*self.params["A0_N"]
        self.true_tissue_params["A0"][self.is_notch] = self.params["A0_P"]
        self.true_tissue_params["P0"] = np.ones(self.t.mesh.n_c, dtype=np.float32)*self.params["P0_N"]
        self.true_tissue_params["P0"][self.is_notch] = self.params["P0_P"]
        self.true_tissue_params["kappa_A"] = np.ones(self.t.mesh.n_c, dtype=np.float32)*self.params["kappa_A_N"]
        self.true_tissue_params["kappa_A"][self.is_notch] = self.params["kappa_A_P"]
        self.true_tissue_params["kappa_P"] = np.ones(self.t.mesh.n_c, dtype=np.float32)*self.params["kappa_P_N"]
        self.true_tissue_params["kappa_P"][self.is_notch] = self.params["kappa_P_P"]

    def set_renormalised_params(self):
        self.t.A0[:] = self.true_tissue_params["A0"]/self.L_true**2
        self.t.P0[:] = self.true_tissue_params["P0"]/self.L_true
        self.t.kappa_P[:]= self.true_tissue_params["kappa_P"]  ##normalised by 1/L to account for the adjusted damping coefficient.
        self.t.kappa_A[:] = self.true_tissue_params["kappa_A"] * self.L_true**2 ##normalised by 1/L to acc for adj. damping coef.

    def update_pressure(self):
        # norm_pressure = 0.5*(1+np.tanh((self.time_point-self.params["pressure_start_time"])*self.params["pressure_slope"]))
        norm_pressure = 0.5*(1+np.tanh((self.time_point-self.params["pressure_start_time"])*self.params["pressure_slope"]))*(1+self.params["osc_level"]*np.sin((self.time_point)*np.pi*2/(self.params["heart_period"])))/(1+self.params["osc_level"])

        self.pressure = self.params["init_pressure"] + (self.params["fin_pressure"]-self.params["init_pressure"])*norm_pressure

    def update_grn(self, dt, dt_grn):
        self.update_pressure()
        self.set_renormalised_params()
        dE_dL = 4*self.true_tissue_params["kappa_A"]*self.t.mesh.A*self.L_true*(self.t.mesh.A*self.L_true**2 - self.true_tissue_params["A0"]) \
              + 2*self.true_tissue_params["kappa_P"]*self.t.mesh.P*(self.t.mesh.P*self.L_true - self.true_tissue_params["P0"])
        dE_dL = dE_dL.sum() - self.pressure*self.L_true**2
        dtL = - self.params["mu_L"]*dE_dL
        self.L_true += dtL*dt

        if self.L_true < self.params["L_min"]:
            self.L_true = self.params["L_min"]
        self.L_save[self.time_step] = self.L_true ##may be unnecessary
        self.var = self.L_true

        self.time_point += dt
        self.time_step += 1



    def update_grn_initialisation(self, dt, dt_grn):
        """
        GRN update strategy. Typically utilizing odeint.

        dt is the time-step of the SPV algorithm,
        dt_grn is the time-step of the GRN updating algorithm. Typically these can just be the same.
        Note that at the moment, the following must hold: dt_grn <= dt
        """
        self.update_pressure()
        self.set_renormalised_params()


