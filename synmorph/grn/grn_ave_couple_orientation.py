import numpy as np

class Grn_ave_couple_orientation:
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
        self.x = None


    def initialize(self):
        """
        Define your initialization of the values of self.var here. This should be of size (nc x nvar). Also include a statement to save nvar.
        :return:
        """
        self.nvar = 2
        self.var = np.zeros((self.t.mesh.n_c,2))
        self.set_initial_condition()
        self.set_motility_params()
        self.set_preferred_area()
        self.set_shape_index()
        self.x0 = self.t.mesh.x.copy()
        self.x0_bound = self.x0[self.t.c_types==3]
        self.x = self.x0.copy()

    def set_initial_condition(self):
        self.centre = np.array((self.t.mesh.L/2,self.t.mesh.L/2))
        self.radius = np.sqrt(self.params["n_AVE_cells"]*self.t.mesh.A.mean()/np.pi)
        x = self.t.mesh.x
        mask = (x[:,0]-self.centre[0])**2 + (x[:,1]-self.centre[1])**2 <= self.radius**2
        self.ave_mask = mask
        # ave_ids = np.nonzero(mask)[0]
        self.t.c_types[mask] = 0
        self.t.c_types[~mask] = 1

        boundary_frac = self.params["boundary_frac"]
        exe_frac = self.params["exe_frac"]
        # top_bottom_mask = (x[:,1]<(self.t.mesh.L*frac)) + (x[:,1]>(self.t.mesh.L*(1-frac)))
        # self.t.c_types[top_bottom_mask] = 2
        self.exe_rad = (1-exe_frac)*self.t.mesh.L/2
        self.bound_rad = (1-boundary_frac)*self.t.mesh.L/2

        self.exe_mask = ((x[:,0]-self.centre[0])**2 + (x[:,1]-self.centre[1])**2 >= self.exe_rad**2)*((x[:,0]-self.centre[0])**2 + (x[:,1]-self.centre[1])**2 < self.bound_rad**2)
        self.boundary_mask = ((x[:,0]-self.centre[0])**2 + (x[:,1]-self.centre[1])**2 >= self.bound_rad**2)
        self.epi_mask = ~(self.ave_mask + self.exe_mask + self.boundary_mask)
        self.t.c_types[self.exe_mask] = 2
        self.t.c_types[self.boundary_mask] = 3



    def set_motility_params(self):
        self.t.active.active_params["alpha_dir"] = self.params["AVE_alpha_dir"]*self.ave_mask + self.params["non_AVE_alpha_dir"]*(~self.ave_mask)
        self.t.active.active_params["v0"] = self.params["AVE_v0"]*self.ave_mask + self.params["non_AVE_v0"]*(~self.ave_mask)
        self.t.active.active_params["angle0"] = self.params["AVE_alpha0"]


    def set_preferred_area(self):
        proportion_ave = np.sum(self.ave_mask)/np.sum(~self.boundary_mask)
        A0_rest = (1 - self.params["AVE_A0"]*proportion_ave)/(1-proportion_ave)
        self.t.A0[self.ave_mask] = self.params["AVE_A0"]
        self.t.A0[(~self.boundary_mask)*(~self.ave_mask)] = A0_rest

    def set_shape_index(self):
        self.t.P0[self.ave_mask] = np.sqrt(self.t.A0[self.ave_mask])*self.params["AVE_p0"]
        self.t.P0[self.epi_mask] = np.sqrt(self.t.A0[self.epi_mask])*self.params["nonAVE_p0"]
        self.t.P0[self.exe_mask] = np.sqrt(self.t.A0[self.exe_mask])*self.params["ExEVE_p0"]

    #
    def get_average_orientation(self):
        neighbourhood_mat = self.t.mesh.l_int != 0
        neighbourhood_mat = neighbourhood_mat.toarray() * np.outer(self.ave_mask,self.ave_mask)
        average_vec = neighbourhood_mat@self.t.active.orientation_vector / np.expand_dims(neighbourhood_mat.sum(axis=1),1)
        average_vec[~self.ave_mask] = 0
        average_orientation = np.arctan2(average_vec[:,1],average_vec[:,0])
        mask_nan = np.isnan(average_orientation)
        average_orientation[mask_nan] = 0
        self.t.active.active_params["alpha_dir"][mask_nan] = 0
        return average_orientation
    #
    # def get_average_orientation(self):
    #     ##couple instead to the instantaneous speeds of the neighbouring cells (rather than purely their active components).
    #     ## weighted average of neighbouring velocities.
    #     if self.x is None:
    #         self.x = self.t.mesh.x
    #     current_vec = self.t.mesh.x - self.x
    #     # if (current_vec == 0).any(): ##set random if there was no movement.
    #     #     current_vec[current_vec == 0] = np.random.random((current_vec == 0).sum())
    #     # current_vec = current_vec / np.expand_dims(np.sqrt(current_vec[...,0]**2 + current_vec[...,1]**2),1)
    #     #
    #
    #     neighbourhood_mat = self.t.mesh.l_int != 0
    #     neighbourhood_mat = neighbourhood_mat.toarray() * np.outer(self.ave_mask,self.ave_mask)
    #     average_vec = neighbourhood_mat@current_vec / np.expand_dims(neighbourhood_mat.sum(axis=1),1)
    #     # average_vec[~self.ave_mask] = 0
    #     average_orientation = np.arctan2(average_vec[:,1],average_vec[:,0])
    #     mask_nan = np.isnan(average_orientation)
    #     average_orientation[mask_nan] = 0
    #     self.t.active.active_params["alpha_dir"][mask_nan] = 0
    #     return average_orientation

    def update_grn(self, dt, dt_grn):
        """
        GRN update strategy. Typically utilizing odeint.

        dt is the time-step of the SPV algorithm,
        dt_grn is the time-step of the GRN updating algorithm. Typically these can just be the same.
        Note that at the moment, the following must hold: dt_grn <= dt
        """
        self.t.mesh.x[self.t.c_types==3] = self.x0_bound
        if type(self.t.F) is np.ndarray:
            if self.t.F.ndim == 2:

                self.t.active.active_params["angle0"] = self.get_average_orientation()
            else:
                self.t.active.active_params["angle0"] = np.random.uniform(0,np.pi*2,self.t.mesh.n_c)
        else:
            self.t.active.active_params["angle0"] = np.random.uniform(0, np.pi * 2, self.t.mesh.n_c)
        self.x = self.t.mesh.x.copy()

    def update_grn_initialisation(self, dt, dt_grn):
        """
        GRN update strategy. Typically utilizing odeint.

        dt is the time-step of the SPV algorithm,
        dt_grn is the time-step of the GRN updating algorithm. Typically these can just be the same.
        Note that at the moment, the following must hold: dt_grn <= dt
        """
        self.t.mesh.x[self.t.c_types == 3] = self.x0_bound
        self.x = self.t.mesh.x
        self.t.active.active_params["angle0"] = np.random.uniform(0, np.pi * 2, self.t.mesh.n_c)

        # self.nvar = 2
        # self.var = np.zeros((self.t.mesh.n_c,2))
        # return

