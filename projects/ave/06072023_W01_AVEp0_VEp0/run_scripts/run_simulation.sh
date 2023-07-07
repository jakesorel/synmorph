#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=03-00:00:00   # walltime
#SBATCH -J "AVE_simulations"   # job name
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G

eval "$(conda shell.bash hook)"
source activate synmorph

python run_multiple_simulation.py ${SLURM_ARRAY_TASK_ID}
#python ../analysis_scripts/run_analysis.py ${SLURM_ARRAY_TASK_ID}