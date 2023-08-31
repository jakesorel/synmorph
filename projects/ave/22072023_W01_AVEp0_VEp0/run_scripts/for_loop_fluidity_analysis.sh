#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=01:00:00   # walltime
#SBATCH -J "AVE2_analysis_for_loop"   # job name
#SBATCH --output=../bash_out/AVE2_analysis_for_loop.out
#SBATCH --error=../bash_out/AVE2_analysis_for_loop.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G



sbatch --array [208-209] run_fluidity_analysis.sh ${SLURM_ARRAY_TASK_ID}


