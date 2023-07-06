#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=2:10:00   # walltime
#SBATCH -J "AVE_compile"   # job name
#SBATCH --output=../bash_out/output_compile_analysis.out
#SBATCH --error=../bash_out/error_compile_analysis.out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=2G


eval "$(conda shell.bash hook)"
source activate synmorph

python compile_analysis.py
