#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=00:05:00   # walltime
#SBATCH -J "AVE_build_list"   # job name
#SBATCH --output=../bash_out/build_list_output.out
#SBATCH --error=../bash_out/build_list_error.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G


eval "$(conda shell.bash hook)"
source activate synmorph

python build_scan_list.py  ${SLURM_ARRAY_TASK_ID} "$1" "$2"
