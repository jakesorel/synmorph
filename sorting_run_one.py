import os
import sacred
from glob import glob
from sacred.observers import FileStorageObserver
from sorting_simulation_logic import do_one_simulation

# Set up Sacred experiment
ex = sacred.Experiment("sorting_test")

# Save any source code dependencies to Sacred
source_files = glob(os.path.join("synmorph", "*.py"))
source_files = [os.path.abspath(f) for f in source_files]
for sf in source_files:
    ex.add_source_file(sf)

# Set storage location for all Sacred results
res_dir = "./sacred"                          # Local
# res_dir = "/home/pbhamidi/scratch/lateral_signaling/sacred"  # Scratch dir on Caltech HPC (fast read/write)

# Use this dir for storage
sacred_storage_dir = os.path.abspath(res_dir)
# os.makedirs(sacred_storage_dir)   # Make dir if it doesn't exist
ex.observers.append(
    FileStorageObserver(sacred_storage_dir)
)

# Set default experimental configuration
config_file = os.path.abspath("default_config.json")
ex.add_config(config_file)

@ex.main  # Use ex as our provenance system and call this function as __main__()
def run_one_simulation(_config, _run, seed):
    """Simulates SPV given a single parameter configuration"""
    # _config contains all the variables in the configuration
    # _run contains data about the run
    
    do_one_simulation(
        ex=ex, 
        **_config
    )


