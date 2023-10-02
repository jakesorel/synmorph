#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=03-00:00:00   # walltime
#SBATCH -J "cpu_count"   # job name
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --output=cpu_count_out.out
#SBATCH --error=cpu_count_error.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G

eval "$(conda shell.bash hook)"
source activate synmorph

python print_cpu_count.py
