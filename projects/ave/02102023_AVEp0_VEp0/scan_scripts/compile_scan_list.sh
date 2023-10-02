#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=01:00:00   # walltime
#SBATCH -J "AVE_build_list"   # job name
#SBATCH --output=../bash_out/build_list_output.out
#SBATCH --error=../bash_out/build_list_error.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G

cat ../scan_summary/temp/17122022_W01_AVEp0_VEp0_v0_lambdaP_alpha_full_summary_*.csv > ../scan_summary/17122022_W01_AVEp0_VEp0_v0_lambdaP_alpha_full_summary.csv
cat ../scan_summary/temp/17122022_W01_AVEp0_VEp0_v0_lambdaP_alpha_path_names_*.txt > ../scan_summary/17122022_W01_AVEp0_VEp0_v0_lambdaP_alpha_path_names.txt
