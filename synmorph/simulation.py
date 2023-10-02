import _pickle as cPickle
import bz2
import importlib
import os
import pickle
from datetime import datetime
import codecs, json
import h5py
import shutil
import gzip


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from synmorph.utils import *

import synmorph.periodic_functions as per
import synmorph.sim_plotting as plot
from synmorph.tissue import Tissue


class Simulation:
    """
    Simulation class
    ----------------

    This is the wrapper for the simulation itself. It will autoinstantiate all of the relevant subclasses (e.g. tissue, mesh, active_force, grn etc.)


    """

    def __init__(self,
                 tissue=None,
                 tissue_params=None,
                 active_params=None,
                 init_params=None,
                 simulation_params=None,
                 grn_params=None,
                 run_options=None,
                 save_options=None):

        assert tissue_params is not None, "Specify tissue params"
        assert active_params is not None, "Specify active params"
        assert init_params is not None, "Specify init params"
        assert simulation_params is not None, "Specify simulation params"
        assert run_options is not None, "Specify run options"
        assert save_options is not None, "Specify save options"

        self.simulation_params = simulation_params
        self.save_options = save_options

        if "random_seed" in self.simulation_params:
            np.random.seed(self.simulation_params["random_seed"])

        if tissue is None:
            self.t = Tissue(tissue_params=tissue_params,
                            active_params=active_params,
                            init_params=init_params,
                            run_options=run_options)
        else:
            self.t = tissue


        ###To do: include functionality to load a tissue pickle file.
        self.t_span = None
        self.t_span_save = None
        self.nt = None
        self.nts = None
        self.x_save = None
        self.tri_save = None
        self.var_save = None
        self.active_save = None
        self.grn = None

        self.initialize()

        ##initialize GRN
        if self.simulation_params["grn_sim"] is not None:
            self.grn = getattr(importlib.import_module("synmorph.grn.%s" % self.simulation_params["grn_sim"].lower()),
                               self.simulation_params[
                                   "grn_sim"].capitalize())  ##import grn class of interest using the string "grn_sim" if this is not None.
            self.grn = self.grn(self.t, grn_params)
            self.var_save = np.zeros((self.x_save.shape[0], self.x_save.shape[1], self.grn.nvar))
        ###


        self.date = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        self.name = self.save_options["name"] if self.save_options["name"] is not None else self.date
        self.id = {"Date": self.date}

        if (self.save_options["result_dir"] is None) or ("result_dir" not in self.save_options):
            self.save_options["result_dir"] = self.date

        if not os.path.exists(self.save_options["result_dir"]):
            os.mkdir(self.save_options["result_dir"])

        # #Remove the excessive directory structure for large scale simualtions:
        # self.save_dir = self.save_options["result_dir"] + "/" + self.name
        # if not os.path.exists(self.save_dir):
        #     os.mkdir(self.save_dir)
        # #
        # self.save_dir_pickled = self.save_options["result_dir"] + "/" + self.name + "/pickled"
        # if not os.path.exists(self.save_dir_pickled):
        #     os.mkdir(self.save_dir_pickled)
        #
        # self.save_dir_plots = self.save_options["result_dir"] + "/" + self.name + "/plots"
        # if not os.path.exists(self.save_dir_plots):
        #     os.mkdir(self.save_dir_plots)
        #
        self.save_dir_pickled = self.save_options["result_dir"]

        if "tinit" in self.simulation_params:
            if self.simulation_params["tinit"] >0:
                self.passive_initialisation()

    def initialize(self):
        """
        Initialize the simulation
        """
        self.get_t_span()
        self.initialize_x_save()
        self.initialize_tri_save()
        self.initialize_active_save()

    def get_t_span(self):
        """
        Get the time-stepping values. t_span is the time points, nt is the number of time-points, t_span_save is the time-points at which data is saved, nts is the corresponding number.
        """
        self.t_span = np.arange(0, self.tfin, self.dt)
        self.nt = self.t_span.size
        self.t_span_save = self.t_span[::self.tskip]
        self.nts = self.t_span_save.size

    def initialize_x_save(self):
        """
        Generate an empty array of positions for saving (at the times defined by self.t_span_save)

        Of size nts x nc x 2
        :return:
        """
        self.x_save = np.zeros(((self.nts,) + self.t.mesh.x.shape))

    def initialize_active_save(self):
        self.active_save = np.zeros(((self.nts,) + self.t.mesh.x.shape))

    def initialize_tri_save(self):
        """
        Generate an empty array of triangulations for saving (at the times defined by self.t_span_save)

        Of size nts x nv x 3 x 2
        :return:
        """
        self.tri_save = np.zeros(((self.nts,) + self.t.mesh.tri.shape),dtype=np.int64)


    def passive_initialisation(self,progress_bar=True):
        """
        Initialise the mesh without active forces.
        :return:
        """
        dt = self.dt  ##repeatedly called to saved here temporarily.
        k = 0  ##dummy variable to count the number of saved time-points.
        v0 = self.t.active.active_params["v0"].copy()
        self.t.active.active_params["v0"] *= 0.

        t_span_init = np.arange(0,self.simulation_params["tinit"],self.dt)
        nt_init = t_span_init.size
        grn = True if self.grn is not None else False  # bool for whether to perform the grn calculation if such a model is specified.
        def update_with_grn(dt):  # short-hand for updating tissue and grn.
            self.grn.update_grn_initialisation(dt, self.simulation_params["dt_grn"])
            self.t.update(dt)
        update = update_with_grn if grn else self.t.update  # define the update rule depending on whether the grn has been specified or not.

        if progress_bar:  # set up the progress bar if wanted.
            iterator = tqdm(range(0, nt_init), ncols=100, desc="Initialisation progress")
        else:
            iterator = range(nt_init)
        for i in iterator:
            t = t_span_init[i]  # present time-point
            update(dt)  # update the tissue and the grn.
            F = self.t.get_forces()  # calculate the forces.
            self.t.mesh.x += F * dt  # execute the movements.
            self.t.mesh.x = per.mod2(self.t.mesh.x, self.t.mesh.L, self.t.mesh.L)  # enforce periodicity
            # fig, ax = plt.subplots()
            # plot.plot_vor(ax,self.t.mesh.x,self.t.mesh.L)
            # fig.savefig("results/current_state.pdf")
        self.t.active.active_params["v0"] = v0



    def simulate(self, progress_bar=True):
        """
        Perform the simulation
        :param progress_bar: bool. Whether to show or hide the progress bar.
        """
        dt = self.dt  ##repeatedly called to saved here temporarily.
        k = 0  ##dummy variable to count the number of saved time-points.

        grn = True if self.grn is not None else False  # bool for whether to perform the grn calculation if such a model is specified.
        saveall = True if self.save_options[
                              "save"] == "all" else False  # bool for whether to dump all instances of the Tissue class into unique pickle files.

        def update_with_grn(dt):  # short-hand for updating tissue and grn.
            self.grn.update_grn(dt, self.simulation_params["dt_grn"])
            self.t.update(dt)

        update = update_with_grn if grn else self.t.update  # define the update rule depending on whether the grn has been specified or not.

        if progress_bar:  # set up the progress bar if wanted.
            iterator = tqdm(range(0, self.nt), ncols=100, desc="Simulation progress")
        else:
            iterator = range(self.nt)
        for i in iterator:
            t = self.t_span[i]  # present time-point
            update(dt)  # update the tissue and the grn.
            F = self.t.get_forces()  # calculate the forces.
            self.t.mesh.x += F * dt  # execute the movements.

            self.t.mesh.x = per.mod2(self.t.mesh.x, self.t.mesh.L, self.t.mesh.L)  # enforce periodicity
            if not i % self.tskip:
                ## for the saving time-points, copy over to x_save (and also var_save)
                self.x_save[k] = self.t.mesh.x
                try: ##debugging code
                    self.tri_save[k] = self.t.mesh.tri
                except:
                    print("k,i",k,i)
                    print("previous_tri")
                    print(self.tri_save[k-1])
                    print("current tri")
                    print(self.t.mesh.tri)
                    print("shapes")
                    print(self.tri_save[k-1].shape,self.t.mesh.tri.shape)
                    self.tri_save[k] = self.t.mesh.tri
                self.active_save[k] = self.t.active.aF
                if grn:
                    self.var_save[k] = self.grn.var
                k += 1
                if saveall:  # if saveall is true, then save the corresponding tissue class to a pickle file.
                    self.t.set_time(t)
                    self.t.save("%s_f%d" % (self.name, i),
                                id={"Date": self.date},
                                dir_path=self.save_dir_pickled,
                                compressed=self.save_options["compressed"])
        if self.save_options["save"] == "last":  # minimal saving option, as an alternative. Saves the simulation class instantiation to a pickle file. This includes x_save and var_save, so most attributes can be reconstructed for further use (bar the active terms, which are currently not saved).
            self.save(self.name,
                      id=self.id,
                      dir_path=self.save_dir_pickled,
                      compressed=self.save_options["compressed"])

        if self.save_options["save"] == "skeleton":
            self.save_skeleton(self.name,id=self.id,dir_path=self.save_dir_pickled)

        if self.save_options["save"] == "hdf5":
            self.save_hdf5_skeleton(self.name,id=self.id,dir_path=self.save_dir_pickled)

    def animate_c_types(self,
                        n_frames=20,
                        c_type_col_map=["#f0a800", "#4287f5"],
                        file_name=None,
                        dir_name=None):
        """
        Generate an animation for the cell-types. Wrapper for plot.animate
        """
        if dir_name is None:
            dir_name = self.save_dir_plots
        if file_name is None:
            file_name = self.name + "_c_types"
        plot.animate(self.x_save,
                     self.t.mesh.L,
                     plot.generate_ctype_cols(self.t.c_types,
                                              c_type_col_map=c_type_col_map),
                     n_frames=n_frames,
                     file_name=file_name,
                     dir_name=dir_name)

    def animate_property(self, tissue_property="dA", cmap=plt.cm.plasma, n_frames=20, file_name=None, dir_name=None,
                         vmid=None):
        """
        Animate a tissue property, defined by the eponymous string. e.g. dA, dP
        :param tissue_property:
        :param cmap:
        :param n_frames:
        :param file_name:
        :param dir_name:
        :param vmid:
        :return:
        """
        if dir_name is None:
            dir_name = self.save_dir_plots
        if file_name is None:
            file_name = self.name + "_" + tissue_property
        skip = int((self.x_save.shape[0]) / n_frames)
        x_save_plot = self.x_save[::skip]
        prop_val = np.zeros((x_save_plot.shape[0], x_save_plot.shape[1]))
        for i in range(n_frames):
            self.t.update_x_mechanics(x_save_plot[i])
            prop_val[i] = getattr(self.t, tissue_property)
        pvmax, pvmin = np.max(prop_val), np.min(prop_val)
        if vmid is None:
            nprop_val = (prop_val - pvmin) / (pvmax - pvmin)
        else:
            if vmid > pvmax:
                print("Mid value greater than the calculated max. Proceeding anyway... ")
            if vmid < pvmin:
                print("Mid value greater than the calculated max. Proceeding anyway... ")
            half = np.max(((pvmax - vmid), (vmid - pvmin)))
            nprop_val = 0.5 + (prop_val - vmid) / (2 * half)
            pvmax = vmid + half
            pvmin = vmid - half
        cols = plot.rgba_to_hex(cmap(nprop_val))
        plot.animate(x_save_plot,
                     self.t.mesh.L,
                     cols,
                     n_frames=n_frames,
                     file_name=file_name,
                     dir_name=dir_name,
                     cbar={"cmap": cmap, "vmin": pvmin, "vmax": pvmax, "label": self.t.get_latex(tissue_property)})

    def animate_grn(self, vari=0, cmap=plt.cm.plasma, n_frames=20, file_name=None, dir_name=None):
        """
        Animate one of the variables in the grn. Which one is defined by vari (an int).
        """
        if dir_name is None:
            dir_name = self.save_dir_plots
        if file_name is None:
            file_name = self.name + "_" + "grn" + "_var_%d" % vari
        skip = int((self.x_save.shape[0]) / n_frames)
        x_save_plot = self.x_save[::skip]
        vart = self.var_save[::skip, :, vari]
        nvart = (vart - vart.min()) / (vart.max() - vart.min())
        cols = plot.rgba_to_hex(cmap(nvart))
        plot.animate(x_save_plot,
                     self.t.mesh.L,
                     cols,
                     n_frames=n_frames,
                     file_name=file_name,
                     dir_name=dir_name,
                     cbar={"cmap": cmap, "vmin": vart.min(), "vmax": vart.max(), "label": "var_%d" % vari})

    def save(self, name, id=None, dir_path="", compressed=False):
        """
        Save the instance of the simulation class to a pickle file.
        :param name:
        :param id:
        :param dir_path:
        :param compressed:
        :return:
        """
        self.name = name
        if id is None:
            self.id = {}
        else:
            self.id = id
        if compressed:
            with bz2.BZ2File(dir_path + "/" + self.name + "_simulation" + '.pbz2', 'w') as f:
                cPickle.dump(self.__dict__, f)
        else:
            pikd = open(dir_path + "/" + self.name + "_simulation" + '.pickle', 'wb')
            pickle.dump(self.__dict__, pikd)
            pikd.close()

    def save_skeleton(self, name, id=None, dir_path=""):
        """
        Save the bare-bones results to a json file
        :param name:
        :param id:
        :param dir_path:
        :param compressed:
        :return:
        """
        self.name = name
        if id is None:
            self.id = {}
        else:
            self.id = id
        skeleton_dict = {"c_types": self.t.c_types.tolist(),
                         "x_save": self.x_save.tolist(),
                         "tri_save": self.tri_save.tolist(),
                         "active_save":self.active_save.tolist(),
                         "t_span_save": self.t_span_save.tolist(),
                         "tissue_params": serialise_dict(self.t.tissue_params),
                         "active_params": serialise_dict(self.t.active.active_params),
                         "run_options": self.t.mesh.run_options,
                         "simulation_params": self.simulation_params,
                         "L":self.t.mesh.L}
        if self.save_options["compressed"] is True:
            with bz2.BZ2File(dir_path + "/" + self.name + "_simulation" + '.pbz2', 'w') as f:
                cPickle.dump(skeleton_dict, f)
        else:
            json.dump(skeleton_dict, codecs.open(dir_path + "/" + self.name + "_simulation" + '.json', 'w', encoding='utf-8'),
                      separators=(',', ':'),
                      sort_keys=True,
                      indent=4)
        ### this saves the array in .json format

    def save_hdf5_skeleton(self, name, id=None, dir_path=""):
        """Note that this drops active_save"""
        self.name = name
        if id is None:
            self.id = {}
        else:
            self.id = id
        skeleton_dict = {"c_types": self.t.c_types,
                         "x_save": self.x_save,
                         "tri_save": self.tri_save}

        skeleton_dict["x_save"] = convert_to_unsigned_int(np.array(skeleton_dict["x_save"]), self.t.mesh.L)
        skeleton_dict["tri_save"] = np.array(skeleton_dict["tri_save"]).astype(np.uint16)
        skeleton_dict["c_types"] = np.array(skeleton_dict["c_types"]).astype(np.uint8)

        file_path = dir_path + "/" + self.name + "_simulation" + '.h5'

        f = h5py.File(file_path, 'w')
        for i, key in enumerate(skeleton_dict.keys()):
            f.create_dataset(key, data=skeleton_dict[key], compression="gzip")
        f.close()

        with open(file_path, 'rb') as f_in:
            with gzip.open(file_path + ".gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(file_path)


    def load(self, fname):
        """
        Load an instance of the simulation class.
        :param fname:
        :return:
        """
        if fname.split(".")[1] == "pbz2":
            fdict = cPickle.load(bz2.BZ2File(fname, 'rb'))

        else:
            pikd = open(fname, 'rb')
            fdict = pickle.load(pikd)
            pikd.close()
        self.__dict__ = fdict

    @property
    def dt(self):
        return self.simulation_params["dt"]

    @property
    def tfin(self):
        return self.simulation_params["tfin"]

    @property
    def tskip(self):
        return self.simulation_params["tskip"]

