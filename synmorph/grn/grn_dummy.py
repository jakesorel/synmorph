class Grn_dummy:
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
        self.initialize()

    def initialize(self):
        """
        Define your initialization of the values of self.var here. This should be of size (nc x nvar). Also include a statement to save nvar.
        :return:
        """

    def update_grn(self, dt, dt_grn):
        """
        GRN update strategy. Typically utilizing odeint.

        dt is the time-step of the SPV algorithm,
        dt_grn is the time-step of the GRN updating algorithm. Typically these can just be the same.
        Note that at the moment, the following must hold: dt_grn <= dt
        """
