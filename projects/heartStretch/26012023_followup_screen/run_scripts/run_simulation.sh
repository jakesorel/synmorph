#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=23:59:00   # walltime
#SBATCH -J "heart_simulations"   # job name
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=32G


eval "$(conda shell.bash hook)"
source activate synmorph

python run_multiple_simulation.py ${SLURM_ARRAY_TASK_ID}




#python ../analysis_scripts/run_analysis.py ${SLURM_ARRAY_TASK_ID}



#python ../analysis_scripts/run_analysis.py ${SLURM_ARRAY_TASK_ID}