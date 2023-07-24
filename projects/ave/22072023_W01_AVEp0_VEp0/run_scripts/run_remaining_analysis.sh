#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=01:00:00   # walltime
#SBATCH -J "AVE_analysis"   # job name
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G

eval "$(conda shell.bash hook)"
source activate synmorph

python ../analysis_scripts/run_remaining_analysis.py ${SLURM_ARRAY_TASK_ID}
