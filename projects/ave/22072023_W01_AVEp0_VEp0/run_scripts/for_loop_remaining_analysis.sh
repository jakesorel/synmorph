#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=01:00:00   # walltime
#SBATCH -J "AVE2_analysis_for_loop"   # job name
#SBATCH --output=../bash_out/AVE2_analysis_for_loop.out
#SBATCH --error=../bash_out/AVE2_analysis_for_loop.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G


sbatch --array [0-6347] --output=../bash_out/output_%A_%a.out --error=../bash_out/error_%A_%a.out run_remaining_analysis.sh ${SLURM_ARRAY_TASK_ID}


