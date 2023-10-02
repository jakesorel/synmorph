#!/bin/bash
#Submit this script with: sbatch thefilename

#SBATCH --time=20:00:00   # walltime
#SBATCH -J "AVE_for_loop_build_lists"   # job name
#SBATCH --output=../bash_out/for_loop_build_lists_output.out
#SBATCH --error=../bash_out/for_loop_build_lists_error.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G

rm -R ../scan_dicts_old
rm -R ../scan_dicts_old_2
