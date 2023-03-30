from uuid import uuid4
import os
import h5py
from glob import glob

from synmorph.simulation import Simulation

#### This script runs one simulation and optionally stores
####   the experiment information (outputs and metadata) in
####   an experiment (`ex`) object that is managed by the
####   provenance system Sacred. Sacred will then store the
####   run metadata/info for subsequent retrieval. The
####   default parameters for the run are specified in a
####   configuration file in the "*run_one.py" script.


# Use a unique directory name for this run
uid = str(uuid4())

# Write to temporary directory of choice (fast read/write)
data_dir = os.path.abspath(f"/tmp/{uid}")  # Use root temp dir (Linux/MacOS)
# data_dir = f"/home/pbhamidi/scratch/lateral_signaling/tmp/{uid}"  # Scratch dir on Caltech HPC
os.makedirs(data_dir, exist_ok=True)


def do_one_simulation(ex=None, save_data=False, animate=False, **cfg):

    # Create simulation with defined configuration
    sim = Simulation(**cfg)

    # Run
    sim.simulate(progress_bar=False)

    if ex is not None:

        # Save any source code dependencies to Sacred
        source_files = glob(os.path.join("synmorph", "*.py"))
        source_files = [os.path.abspath(f) for f in source_files]
        for sf in source_files:
            ex.add_source_file(sf)

        # Initialize stuff to save
        artifacts = []

        # Dump data to file
        if save_data:

            print("Writing data")

            # Dump data to an HDF5 file
            data_dump_fname = os.path.join(data_dir, "results.hdf5")
            with h5py.File(data_dump_fname, "w") as f:
                f.create_dataset("c_types", data=sim.t.c_types)
                f.create_dataset("t_span_save", data=sim.t_span_save)
                f.create_dataset("tri_save", data=sim.tri_save)
                f.create_dataset("x_save", data=sim.x_save)

            # Add to Sacred artifacts
            artifacts.append(data_dump_fname)

        # Make animation
        if animate:

            print("Making animation")

            anim_fname = os.path.join(data_dir, "animation.mp4")
            sim.animate_c_types(
                dir_name=data_dir,
                file_name=anim_fname,
                n_frames=cfg["n_frames"],
                fps=cfg["fps"],
                dpi=cfg["dpi"],
            )

            # Add to Sacred artifacts
            artifacts.append(anim_fname)

        # Add all artifacts to Sacred
        for _a in artifacts:
            ex.add_artifact(_a)

    else:
        return sim
