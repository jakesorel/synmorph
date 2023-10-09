#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=01-00:00:00   # walltime
#SBATCH -J "cpu_count"   # job name
#SBATCH -n 1
#SBATCH --partition=hmem
#SBATCH --cpus-per-task=12
#SBATCH --output=../bash_out/plot_sim_output.out
#SBATCH --error=../bash_out/plot_sim_error.out
#SBATCH --mem=128G

eval "$(conda shell.bash hook)"
source activate synmorph

python plot_simulation.py
