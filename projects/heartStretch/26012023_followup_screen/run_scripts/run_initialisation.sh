#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=23:59:00   # walltime
#SBATCH -J "heart_simulations_initialisation"   # job name
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=../bash_out/error_initialisation.out
#SBATCH --error=../bash_out/output_initialisation.out

eval "$(conda shell.bash hook)"
source activate synmorph

python mcmc_randomiser.py


