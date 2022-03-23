#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH -J "AV_jakecs"   # job name
#SBATCH --output=../bash_out/output.out
#SBATCH --error=../bash_out/error.out

source activate synmorph

python run_N_simulations.py "$1" "$2"