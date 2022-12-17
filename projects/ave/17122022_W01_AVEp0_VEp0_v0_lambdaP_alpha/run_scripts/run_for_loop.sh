#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=1:00:00   # walltime
#SBATCH -J "AVE_for_loop_jobs"   # job name
#SBATCH --output=../bash_out/for_loop_jobs_output.out
#SBATCH --error=../bash_out/for_loop_jobs_error.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G

./for_loop.sh 10 10