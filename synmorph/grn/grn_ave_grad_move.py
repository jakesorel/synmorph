import numpy as np

class Grn_ave_grad_move:
    """
    Skeleton instance of GRN-like class.

    Called if simulation_options["grn_type"] == "grn_dummy"

    All the GRN simulation parameters can be parsed in grn_params. This allows the programmer to feed in the specifications of the GRN model via the instantiation of the overarching Simulation class.

    In principle, this class can also modify the tissue class (e.g. if a gene initiates the change in the physical property of the cell).

    """

    def __init__(self, tissue, grn_params=None):
        assert grn_params is not None, "Specify grn params"
        self.t = tissue
        self.var = None
        self.nvar = None
        self.params = grn_params
        self.x0 = None
        self.x0_bound = None
        self.initialize()


    def initialize(self):
        """
        Define your initialization of the values of self.var here. This should be of size (nc x nvar). Also include a statement to save nvar.
        :return:
        """
        self.nvar = 2
        self.var = np.zeros((self.t.mesh.n_c,2))
        self.set_initial_condition()
        self.set_motility_params()
        self.x0 = self.t.mesh.x.copy()
        self.x0_bound = self.x0[self.t.c_types==2]

    def set_initial_condition(self):
        centre = np.array((self.t.mesh.L/2,self.t.mesh.L/2))
        radius = np.sqrt(self.params["n_AVE_cells"]*self.t.mesh.A.mean()/np.pi)
        x = self.t.mesh.x
        mask = (x[:,0]-centre[0])**2 + (x[:,1]-centre[1])**2 <= radius**2
        self.ave_mask = mask
        # ave_ids = np.nonzero(mask)[0]
        self.t.c_types[mask] = 1
        self.t.c_types[~mask] = 0

        frac = 1/10
        top_bottom_mask = (x[:,1]<(self.t.mesh.L*frac)) + (x[:,1]>(self.t.mesh.L*(1-frac)))
        self.t.c_types[top_bottom_mask] = 2

    def set_motility_params(self):
        self.t.active.active_params["alpha_dir"] = self.params["AVE_alpha_dir"]*self.ave_mask + self.params["non_AVE_alpha_dir"]*(~self.ave_mask)
        self.t.active.active_params["v0"] = self.params["AVE_v0"]*self.ave_mask + self.params["non_AVE_v0"]*(~self.ave_mask)
        self.t.active.active_params["angle0"] = self.params["AVE_alpha0"]

    def update_grn(self, dt, dt_grn):
        """
        GRN update strategy. Typically utilizing odeint.

        dt is the time-step of the SPV algorithm,
        dt_grn is the time-step of the GRN updating algorithm. Typically these can just be the same.
        Note that at the moment, the following must hold: dt_grn <= dt
        """
        self.t.mesh.x[self.t.c_types==2] = self.x0_bound
        if type(self.t.F) is np.ndarray:
            if self.t.F.ndim == 2:

                Fnorm = self.t.F/np.expand_dims(np.linalg.norm(self.t.F,axis=1),1)
                Fangle = np.arctan2(Fnorm[:,1],Fnorm[:,0])
                self.t.active.active_params["angle0"] = Fangle
            else:
                self.t.active.active_params["angle0"] = np.random.uniform(0,np.pi*2,self.t.mesh.n_c)
        else:
            self.t.active.active_params["angle0"] = np.random.uniform(0, np.pi * 2, self.t.mesh.n_c)

        # self.nvar = 2
        # self.var = np.zeros((self.t.mesh.n_c,2))
        # return

